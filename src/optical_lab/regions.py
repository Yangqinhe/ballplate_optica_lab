"""Nested 8/16/32/64-region annotations, display only; import has no Qt/SDK IO.

Attach only while constructing a new GUI. A transparent child paints on top
of the measured CCD widget; neither its QImage nor any numeric buffer changes.
The original nonuniform partition is mapped proportionally to the current ROI.
Overlay and test integration use this SAME layout; changed geometry is diagnostic,
not a claim that an old control head remains qualified.
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
    if ((a.ndim != 3 or a.shape[0] not in (8,16,32,64) or a.shape[1:] != (roi['Height'], roi['Width'])) or a.dtype.kind not in 'buif'
            or not np.isfinite(a).all() or not np.isin(a, (0, 1)).all()
            or not np.all(a.sum(axis=0) == 1)):
        raise ValueError('8/16/32/64 binary regions must cover the native ROI exactly once')
    regions = []
    for index, mask in enumerate(a):
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError('empty region is not a valid readout')
        x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
        if len(xx) != (x1-x0)*(y1-y0):
            raise ValueError('this overlay requires exact rectangular regions, not bounding-box approximations')
        regions.append(dict(index=index, bounds=[x0, y0, x1, y1], pixels=len(xx)))
    return dict(roi=roi, shape=[roi['Height'], roi['Width']], regions=regions, index_base=0, region_count=len(regions))


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
        raise ValueError('ROI位置或尺寸与读出区域不一致；不显示错误划分')
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


def layout_for_roi(layout, roi):
    """Preserve original region topology/indexing, not obsolete sensor offsets."""
    roi = _roi(roi)
    if roi['Width'] > 2592 or roi['Height'] > 1944:
        raise ValueError('ROI exceeds native camera dimensions')
    result = copy.deepcopy(layout)
    ow, oh = layout['roi']['Width'], layout['roi']['Height']
    w, h = roi['Width'], roi['Height']
    for region in result['regions']:
        x0,y0,x1,y1 = region['bounds']
        region['bounds'] = [round(x0*w/ow),round(y0*h/oh),round(x1*w/ow),round(y1*h/oh)]
        a,b,c,d = region['bounds']
        if c <= a or d <= b:
            raise ValueError('ROI过小，无法保留所有非空读出区域')
        region['pixels'] = (c-a)*(d-b)
    result.update(roi=roi, shape=[h,w], original_roi=copy.deepcopy(layout['roi']),
                  coordinate_policy='ROI-relative proportional native8',
                  geometry_changed=roi != layout['roi'], control_qualified=False)
    return result


def weights_for_layout(layout):
    weights = np.zeros((len(layout['regions']), *layout['shape']), np.uint8)
    for r in layout['regions']:
        x0,y0,x1,y1 = r['bounds']
        weights[r['index'],y0:y1,x0:x1] = 1
    if not np.all(weights.sum(0) == 1):
        raise ValueError('读出划分存在重叠或缺口')
    return weights


def subdivide_layout(layout, count):
    """Split each original rectangle at its longest side; exact nested coverage.

    Rescale the ORIGINAL eight first, then split using integer midpoints.
    Child ordering is parent-major; summing each consecutive child group
    reproduces that parent's pixel sum exactly.
    """
    if type(count) is not int or count not in (8,16,32,64):
        raise ValueError('读出分区数只能为8、16、32、64')
    if len(layout['regions']) != 8:
        raise ValueError('分区必须从原八区生成，不能累计重复细分')
    result=copy.deepcopy(layout)
    for region in result['regions']:
        region['parent8']=region['index']
    while len(result['regions']) < count:
        children=[]
        for region in result['regions']:
            x0,y0,x1,y1=region['bounds'];w,h=x1-x0,y1-y0
            if max(w,h)<2:raise ValueError('当前ROI太小，不能细分为非空区域')
            if w>=h:
                mid=(x0+x1)//2
                boxes=([x0,y0,mid,y1],[mid,y0,x1,y1])
            else:
                mid=(y0+y1)//2
                boxes=([x0,y0,x1,mid],[x0,mid,x1,y1])
            for box in boxes:
                a,b,c,d=box
                children.append(dict(index=len(children),parent8=region['parent8'],
                                     bounds=box,pixels=(c-a)*(d-b)))
        result['regions']=children
    result.update(region_count=count,
        partition_policy='native8-longest-side-midpoint-nested-v1',
        control_qualified=False)
    return result


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
            font = painter.font(); font.setBold(True)
            for region in plan:
                color = QtGui.QColor('#66e5ff')
                pen = QtGui.QPen(color); pen.setWidthF(1.0)
                pen.setCosmetic(True); painter.setPen(pen); painter.setBrush(QtCore.Qt.NoBrush)
                rect = QtCore.QRectF(*region['bounds'])
                painter.drawRect(rect.adjusted(.5, .5, -.5, -.5))
                font.setPixelSize(12);painter.setFont(font)
                metrics=painter.fontMetrics()
                title = str(region['index']) if len(plan)>16 else 'Region ' + str(region['index'])
                if metrics.horizontalAdvance(title)+16 > rect.width():
                    title = str(region['index']) if len(plan)>16 else 'R' + str(region['index'])
                while metrics.horizontalAdvance(title)+4>rect.width()-2 and font.pixelSize()>6:
                    font.setPixelSize(font.pixelSize()-1);painter.setFont(font);metrics=painter.fontMetrics()
                label = QtCore.QRectF(rect.left()+1, rect.top()+1,
                                     min(metrics.horizontalAdvance(title)+4, max(1, rect.width()-2)),
                                     min(metrics.height()+2, max(1, rect.height()-2)))
                painter.fillRect(label, QtGui.QColor(0, 0, 0, 175))
                painter.drawText(label, QtCore.Qt.AlignCenter, title)
            painter.end()

    class Controller(QtCore.QObject):
        def __init__(self):
            super().__init__(controls)
            self.layout = frozen
            self.count = 8
            self.selector = QtWidgets.QComboBox(controls)
            self.selector.setObjectName('main_readout_regions')
            for count in (8,16,32,64):self.selector.addItem(f'{count}区 → 2输出（{2*count+2}参数）',count)
            self.last_error = None
            self.checkbox = QtWidgets.QCheckBox('显示读出区域（R0–R7）', controls)
            self.checkbox.setObjectName('manual_show_native8_regions')
            self.checkbox.setToolTip('原八区内逐级细分；预览与测试积分使用同一划分。不改变RAW、DMD或SLM。')
            self.note = QtWidgets.QLabel(controls)
            self.note.setWordWrap(True)
            self.note.setFixedHeight(self.note.fontMetrics().lineSpacing()*2+6)
            self.note.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            row = grid.rowCount()
            grid.addWidget(QtWidgets.QLabel('读出区域',controls),row,0)
            grid.addWidget(self.selector,row,1,1,3)
            row += 1
            grid.addWidget(self.checkbox, row, 0, 1, 2)
            grid.addWidget(self.note, row, 2, 1, 4)
            self.overlay = Overlay(self)
            self.checkbox.toggled.connect(self.refresh)
            self.selector.currentIndexChanged.connect(self.change_count)
            view.installEventFilter(self)
            # Reuse the manual display refresh timer; no new per-frame processing.
            timer = getattr(window.manual_display, 'timer', None)
            if timer is not None:
                timer.timeout.connect(self.refresh)
            self.refresh()

        def change_count(self, *args):
            count=int(self.selector.currentData())
            flow=getattr(window,'stability_workflow',None)
            if flow is not None and flow.active:
                previous=self.selector.blockSignals(True)
                self.selector.setCurrentIndex(self.selector.findData(self.count))
                self.selector.blockSignals(previous)
                self.note.setText('测试中使用已冻结分区；结束后再调整。')
                return
            self.count=count
            self.checkbox.setText(f'显示读出区域（R0–R{count-1}）')
            self.refresh()
            if flow is not None:
                flow.refresh_actual()
                if flow.result:
                    flow.widgets['summary'].setText(
                        f'当前主界面为{count}区；下方为已保存的'
                        f'{flow.result.get("region_count",8)}区结果，需重新测试才会更新。')
            if getattr(window,'hardware',False) and hasattr(window,'record'):
                window.record(dict(state='readout_partition_selected',region_count=count,
                    partition_policy='native8-longest-side-midpoint-nested-v1',
                    hardware_settings_changed=False))

        def current_layout(self, roi=None):
            roi = _roi(roi if roi is not None else getattr(view, 'active_roi', None))
            if getattr(self, '_roi_key', None) != (roi,self.count):
                self._current = subdivide_layout(layout_for_roi(frozen, roi),self.count)
                self._roi_key = (roi,self.count)
            return copy.deepcopy(self._current)

        def plan(self):
            if not self.checkbox.isChecked():
                raise ValueError('读出划分已关闭')
            if getattr(window, 'fault', None):
                raise ValueError('CCD故障／旧帧：暂不显示读出划分')
            image = getattr(view, 'image', None)
            if image is None or image.isNull():
                raise ValueError('等待当前CCD图像')
            # Full-sensor display may be downsampled, but SensorView retains the
            # native ROI extent and image rectangle. Do not reject a valid preview.
            return project_regions(self.current_layout(), getattr(view, 'active_roi', None), getattr(view, 'image_rect', None))

        def refresh(self, *args):
            self.overlay.setGeometry(view.rect())
            if not self.checkbox.isChecked():
                self.last_error = None
                self.overlay.hide()
                self.note.setText(f'{self.count}区，R0–R{self.count-1}；原八区逐级细分，预览与积分一致。')
                return
            try:
                self.plan()
            except ValueError as error:
                self.last_error = str(error)
                self.overlay.hide()
                self.note.setText('读出划分未显示：' + str(error))
                return
            self.last_error = None
            self.note.setText(f'{self.count}区，R0–R{self.count-1}。换分区需适配对应线性头；不改变光学配置。')
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
