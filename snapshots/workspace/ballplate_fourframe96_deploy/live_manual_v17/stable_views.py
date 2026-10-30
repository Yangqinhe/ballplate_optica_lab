"""Stable, sensor-extent-aware GUI views. No camera or output ownership."""
import numpy as np
from PyQt5 import QtCore,QtGui,QtWidgets


class StatusLabel(QtWidgets.QLabel):
    """Changing message length never changes the layout's geometry."""
    def __init__(self,text='',lines=2):
        super().__init__();self.setWordWrap(True)
        self.setFixedHeight(self.fontMetrics().lineSpacing()*lines+6)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Fixed)
        self.setText(text)

    def setText(self,text):
        super().setText(text);self.setToolTip(text)


class SensorView(QtWidgets.QLabel):
    """Pixmap size is not a layout hint; all images describe a native CCD ROI."""
    def __init__(self,text='等待图像'):
        super().__init__(text);self.image=None;self.extent=None;self.shared_fit=None;self.image_rect=None
        self.setMinimumSize(220,240);self.setAlignment(QtCore.Qt.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Ignored)
        self.setStyleSheet('background:#111;color:#ddd')

    def sizeHint(self):return QtCore.QSize(320,320)
    def minimumSizeHint(self):return QtCore.QSize(220,240)

    def setText(self,text):
        self.image=None;self.image_rect=None
        super().setText(text)

    def set_image(self,values,extent=None):
        a=np.ascontiguousarray(values)
        if a.dtype!=np.uint8 or a.ndim not in (2,3) or (a.ndim==3 and a.shape[2]!=3):
            raise ValueError('display-only uint8 gray/RGB image required')
        h,w=a.shape[:2];rgb=a.ndim==3
        self.image=QtGui.QImage(a.data,w,h,w*(3 if rgb else 1),
            QtGui.QImage.Format_RGB888 if rgb else QtGui.QImage.Format_Grayscale8).copy()
        self.extent=tuple(extent or (w,h))
        if min(self.extent)<=0:raise ValueError('positive native CCD extent required')
        self.refresh_image()

    def refresh_image(self):
        if self.image is None or self.extent is None:return
        available=self.shared_fit or self.size()
        ew,eh=self.extent;scale=min(available.width()/ew,available.height()/eh)
        w,h=max(1,int(ew*scale)),max(1,int(eh*scale))
        # A 256² comparison array is sampled over a potentially NON-square ROI.
        # Render using sensor extent, not the intermediate array's aspect ratio.
        pix=QtGui.QPixmap.fromImage(self.image).scaled(w,h,QtCore.Qt.IgnoreAspectRatio,QtCore.Qt.FastTransformation)
        super().setPixmap(pix)
        self.image_rect=((self.width()-w)//2,(self.height()-h)//2,w,h)

    def resizeEvent(self,event):
        super().resizeEvent(event);self.refresh_image()


class SharedSensorScale(QtCore.QObject):
    """Use exactly the same available display area for every comparison view."""
    def __init__(self,views,parent=None):
        super().__init__(parent);self.views=views;self.pending=False
        for view in views:view.installEventFilter(self)
        self.queue_refresh()

    def queue_refresh(self):
        if not self.pending:
            self.pending=True;QtCore.QTimer.singleShot(0,self.refresh)

    def eventFilter(self,obj,event):
        if event.type()==QtCore.QEvent.Resize:self.queue_refresh()
        return False

    def refresh(self):
        self.pending=False
        size=QtCore.QSize(min(v.width() for v in self.views),min(v.height() for v in self.views))
        for view in self.views:view.shared_fit=size;view.refresh_image()
