"""Exact native-eight-region annotations, display only; import has no Qt/SDK IO.

Attach only while constructing a new GUI. A transparent child paints on top
of the measured CCD widget; neither its QImage nor any numeric buffer changes.
The mask's sensor ROI is explicit: a moved/resized ROI suppresses annotation.
"""
from __future__ import annotations

import copy
import hashlib
import numbers
from pathlib import Path
import re

import numpy as np


def _roi(value):
    if not isinstance(value, dict):
        raise ValueError('explicit native sensor ROI required')
    result = {}
    for key in ('OffsetX', 'OffsetY', 'Width', 'Height'):
        item = value.get(key)
        if isinstance(item, str) and re.fullmatch(r'[0-9]+', item):
            item = int(item)
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Integral):
            raise ValueError('integer native ROI required: ' + key)
        result[key] = int(item)
    if min(result.values()) < 0 or min(result['Width'], result['Height']) < 1:
        raise ValueError('nonnegative offsets and positive native extent required')
    return result


def layout_from_weights(weights, roi):
    """Derive exact half-open rectangles; never invent an equal-size grid."""
    roi = _roi(roi)
    a = np.asarray(weights)
    if (a.shape != (8, roi['Height'], roi['Width']) or a.dtype.kind not in 'buif'
            or not np.isfinite(a).all() or not np.isin(a, (0, 1)).all()
            or not np.all(a.sum(axis=0) == 1)):
        raise ValueError('eight binary regions must cover the native ROI exactly once')
    regions = []
    for index, mask in enumerate(a):
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError('empty region is not a valid readout')
        x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
        if len(xx) != (x1-x0)*(y1-y0):
            raise ValueError('this overlay requires exact rectangular regions, not bounding-box approximations')
        regions.append(dict(index=index, bounds=[x0, y0, x1, y1], pixels=len(xx)))
    return dict(roi=roi, shape=[roi['Height'], roi['Width']], regions=regions, index_base=0)


def load_layout(path, sha256, roi):
    """Read only an explicitly named, checksum-bound native mask."""
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path or path.is_symlink()
            or not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256)):
        raise ValueError('absolute regular mask and explicit SHA256 required')
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != sha256:
        raise ValueError('native eight-region mask SHA256 differs')
    # Decode the same verified bytes rather than reopening a possibly replaced file.
    import io
    layout = layout_from_weights(np.load(io.BytesIO(blob), allow_pickle=False), roi)
    layout['source'] = dict(path=str(path), sha256=sha256)
    return layout


def project_regions(layout, active_roi, image_rect):
    if _roi(active_roi) != layout['roi']:
        raise ValueError('ROI位置或尺寸与固定八区不一致；不显示错误划分')
    if (not isinstance(image_rect, (tuple, list)) or len(image_rect) != 4
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real)
                   or not np.isfinite(v) for v in image_rect)
            or min(image_rect[2:]) <= 0):
        raise ValueError('valid current image display rectangle required')
    ox, oy, width, height = image_rect
    sx, sy = width/layout['roi']['Width'], height/layout['roi']['Height']
    return [dict(index=r['index'], bounds=[ox+r['bounds'][0]*sx, oy+r['bounds'][1]*sy,
            (r['bounds'][2]-r['bounds'][0])*sx, (r['bounds'][3]-r['bounds'][1])*sy])
            for r in layout['regions']]


def attach(window, layout):
    """Startup-only composition on manual measured CCD; no new device owner."""
    from PyQt5 import QtCore, QtGui, QtWidgets

    if hasattr(window, 'region_overlay'):
        raise ValueError('native region overlay already attached')
    view = window.view
    controls = window.manual_display.controls
    grid = controls.layout()
    if not isinstance(grid, QtWidgets.QGridLayout):
        raise ValueError('original manual display controls grid required')
    frozen = copy.deepcopy(layout)

    class Overlay(QtWidgets.QWidget):
        def __init__(self, controller):
            super().__init__(view)
            self.controller = controller
            self.setObjectName('manual_native8_overlay')
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
            self.setFocusPolicy(QtCore.Qt.NoFocus)
            self.setAutoFillBackground(False)
            self.setStyleSheet('background: transparent;')
            self.hide()

        def paintEvent(self, event):
            try:
                plan = self.controller.plan()
            except ValueError:
                return  # A just-cleared/fault/changed-ROI image must not show old lines.
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
            painter.setClipRect(QtCore.QRectF(*view.image_rect))
            font = painter.font(); font.setPointSize(9); font.setBold(True); painter.setFont(font)
            metrics = painter.fontMetrics()
            for region in plan:
                color = QtGui.QColor({4: '#ffad33', 5: '#ff55d0'}.get(region['index'], '#66e5ff'))
                pen = QtGui.QPen(color); pen.setWidthF(2.0 if region['index'] in (4, 5) else 1.0)
                pen.setCosmetic(True); painter.setPen(pen); painter.setBrush(QtCore.Qt.NoBrush)
                rect = QtCore.QRectF(*region['bounds'])
                painter.drawRect(rect.adjusted(.5, .5, -.5, -.5))
                title = 'Region ' + str(region['index'])
                if metrics.horizontalAdvance(title)+16 > rect.width():
                    title = 'R' + str(region['index'])  # Preserve the index on narrow regions.
                label = QtCore.QRectF(rect.left()+4, rect.top()+4,
                                     min(metrics.horizontalAdvance(title)+8, max(1, rect.width()-8)),
                                     min(metrics.height()+4, max(1, rect.height()-8)))
                painter.fillRect(label, QtGui.QColor(0, 0, 0, 175))
                painter.drawText(label, QtCore.Qt.AlignCenter, title)
            painter.end()

    class Controller(QtCore.QObject):
        def __init__(self):
            super().__init__(controls)
            self.layout = frozen
            self.last_error = None
            self.checkbox = QtWidgets.QCheckBox('显示八区划分（Region 0–7）', controls)
            self.checkbox.setObjectName('manual_show_native8_regions')
            self.checkbox.setToolTip('当前研究积分的真实分区；只显示在实测CCD，不改变RAW、滤波、积分或控制。')
            self.note = QtWidgets.QLabel(controls)
            self.note.setWordWrap(True)
            self.note.setFixedHeight(self.note.fontMetrics().lineSpacing()*2+6)
            self.note.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            row = grid.rowCount()
            grid.addWidget(self.checkbox, row, 0, 1, 2)
            grid.addWidget(self.note, row, 2, 1, 4)
            self.overlay = Overlay(self)
            self.checkbox.toggled.connect(self.refresh)
            view.installEventFilter(self)
            # Reuse the manual display refresh timer; no new per-frame processing.
            timer = getattr(window.manual_display, 'timer', None)
            if timer is not None:
                timer.timeout.connect(self.refresh)
            self.refresh()

        def plan(self):
            if not self.checkbox.isChecked():
                raise ValueError('八区划分已关闭')
            if getattr(window, 'fault', None):
                raise ValueError('CCD故障／旧帧：暂不显示八区划分')
            image = getattr(view, 'image', None)
            if image is None or image.isNull():
                raise ValueError('等待当前CCD图像')
            if (image.width(), image.height()) != (frozen['roi']['Width'], frozen['roi']['Height']):
                raise ValueError('CCD图像尺寸与固定八区不一致')
            return project_regions(frozen, getattr(view, 'active_roi', None), getattr(view, 'image_rect', None))

        def refresh(self, *args):
            self.overlay.setGeometry(view.rect())
            if not self.checkbox.isChecked():
                self.last_error = None
                self.overlay.hide()
                self.note.setText('仅实测CCD显示；编号从0开始。Region 4橙色、5粉色。')
                return
            try:
                self.plan()
            except ValueError as error:
                self.last_error = str(error)
                self.overlay.hide()
                self.note.setText('八区划分未显示：' + str(error))
                return
            self.last_error = None
            self.note.setText('原固定积分分区；Region 4／5为左下上下相邻区。仅显示，不修改数值。')
            self.overlay.show(); self.overlay.raise_(); self.overlay.update()
            band = getattr(view, 'band', None)
            if band is not None:
                band.raise_()  # Keep the original ROI-selection rubber band above annotations.

        def eventFilter(self, obj, event):
            if obj is view and event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
                self.overlay.setGeometry(view.rect())
                QtCore.QTimer.singleShot(0, self.refresh)
            return False

    controller = Controller()
    window.region_overlay = controller
    return controller
