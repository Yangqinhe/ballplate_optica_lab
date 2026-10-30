"""CCD display mapping/controls only. Import never loads Qt, SDK or old GUI."""
import numbers
import time
from types import MethodType

import numpy as np


def map_display_pair(raw, filtered=None, *, mode='fixed', gain=16., black=10., gamma=.5):
    """Two displays share ONE raw-derived scale; neither is a control feature."""
    a = np.asarray(raw)
    if a.ndim != 2 or not a.size or a.size > 1048576 or a.dtype != np.uint8:
        raise ValueError('native nonempty 2D Mono8, at most1048576 pixels required')
    if mode not in ('fixed', 'auto', 'raw'):
        raise ValueError('explicit fixed/auto/raw display mode required')
    for v, lo, hi in ((gain,1,128),(black,0,255),(gamma,.3,3)):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Real) or not np.isfinite(v) or not lo<=v<=hi:
            raise ValueError('finite bounded display controls required')
    b = a if filtered is None else np.asarray(filtered)
    if b.shape != a.shape or b.dtype.kind not in 'uif' or not np.isfinite(b).all():
        raise ValueError('same-shape finite numeric filtered display required')
    levels = None
    if mode == 'auto':
        cumulative = np.bincount(a.ravel(),minlength=256).cumsum()
        low = int(np.searchsorted(cumulative,max(1,int(np.ceil(a.size*.05)))))
        high = int(np.searchsorted(cumulative,max(1,int(np.ceil(a.size*.999)))))
        levels = dict(low_ADU=low,high_ADU=high)

    def mapped(values):
        values = np.asarray(values,np.float64)
        if mode == 'fixed':
            level = np.clip((values-black)*gain/255.,0,1)
            return np.rint(np.power(level,gamma)*255.).astype(np.uint8)
        if mode == 'auto' and high-low >= 3:
            return np.clip(np.floor((values-low)*255./(high-low)),0,255).astype(np.uint8)
        return np.rint(np.clip(values,0,255)).astype(np.uint8)

    return dict(unfiltered=mapped(a),filtered=mapped(b),auto_levels=levels)


def validate_display_parameters(values):
    """Validate a retained display setting; never grants hardware authority."""
    if not isinstance(values,dict) or set(values)!={'mode','gain','black','gamma'}:
        raise ValueError('explicit mode/gain/black/gamma display settings required')
    map_display_pair(np.zeros((1,1),np.uint8),**values)
    return dict(mode=values['mode'],gain=float(values['gain']),black=float(values['black']),gamma=float(values['gamma']))


def effective_parameters(window, controls):
    """A follower uses the manual page's COMPLETE mapping, not just its toggle."""
    manual = getattr(window,'manual_display',None)
    if controls.mode.currentData()=='auto' and manual is not None and controls is not manual.controls:
        return manual.controls.parameters(window.enhance.isChecked())
    return controls.parameters(window.enhance.isChecked())


def make_display_controls(parent, key, changed, native_clicked=None):
    """Shared controls for two actual consumers; Qt is imported only on attach."""
    from PyQt5 import QtCore, QtWidgets
    class Controls(QtWidgets.QGroupBox):
        def __init__(self):
            super().__init__('CCD显示设置 · 不进入八区积分／电域网络',parent)
            grid = QtWidgets.QGridLayout(self)
            self.mode = QtWidgets.QComboBox()
            automatic = '自动对比（原手动预览）' if key=='manual' else '跟随手动对齐（全部显示参数）'
            for title,data in ((automatic,'auto'),('手动显示增益','fixed'),('原始ADU','raw')):
                self.mode.addItem(title,data)
            self.mode.setObjectName(key+'_display_mode'); grid.addWidget(self.mode,0,0,1,3)
            raw = QtWidgets.QPushButton('原始ADU显示')
            raw.clicked.connect(lambda: native_clicked() if native_clicked else self.mode.setCurrentIndex(2))
            grid.addWidget(raw,0,3)
            weak = QtWidgets.QPushButton('弱光增强16×（也放大噪声）'); weak.clicked.connect(self.weak)
            grid.addWidget(weak,0,4,1,2)
            grid.addWidget(QtWidgets.QLabel('仅显示增益'),1,0)
            self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.gain_slider.setRange(10,1280); self.gain_slider.setValue(40)
            self.gain_slider.setObjectName(key+'_display_gain_slider'); grid.addWidget(self.gain_slider,1,1,1,2)
            self.gain = QtWidgets.QDoubleSpinBox(); self.gain.setRange(1,128); self.gain.setDecimals(1)
            self.gain.setSingleStep(.1); self.gain.setValue(4); self.gain.setSuffix(' ×')
            self.black = QtWidgets.QDoubleSpinBox(); self.black.setRange(0,255); self.black.setValue(10)
            self.black.setSingleStep(.1); self.black.setSuffix(' ADU')
            self.gamma = QtWidgets.QDoubleSpinBox(); self.gamma.setRange(.3,3); self.gamma.setValue(1)
            self.gamma.setSingleStep(.1)
            for w,name in ((self.gain,'gain'),(self.black,'black'),(self.gamma,'gamma')):
                w.setObjectName(key+'_display_'+name); w.setKeyboardTracking(False)
            grid.addWidget(self.gain,1,3)
            grid.addWidget(QtWidgets.QLabel('黑偏置／γ（仅显示）'),2,0)
            grid.addWidget(self.black,2,1); grid.addWidget(self.gamma,2,2)
            tip = QtWidgets.QLabel(('调节仅影响实测CCD预览；不消除原始噪声，不改相机Gain／曝光。'
                if key=='manual' else '跟随模式使用手动页完整参数；手动模式启用本页滑块。相机Gain／曝光不变。'))
            tip.setWordWrap(True); grid.addWidget(tip,2,3,1,3)
            if key=='manual':
                self.filter_generation=0
                self.filter_mode=QtWidgets.QComboBox()
                for title,data in (('关闭时间滤波','off'),('两帧均值（预览采样帧）','mean2'),('指数平滑（预览采样帧）','ema')):
                    self.filter_mode.addItem(title,data)
                self.filter_mode.setObjectName('manual_preview_filter_mode')
                grid.addWidget(self.filter_mode,3,0,1,2)
                grid.addWidget(QtWidgets.QLabel('当前帧权重α'),3,2)
                self.filter_alpha=QtWidgets.QDoubleSpinBox()
                self.filter_alpha.setRange(.1,1.);self.filter_alpha.setDecimals(2)
                self.filter_alpha.setSingleStep(.05);self.filter_alpha.setValue(.5)
                self.filter_alpha.setObjectName('manual_preview_filter_alpha');grid.addWidget(self.filter_alpha,3,3)
                self.filter_original=QtWidgets.QCheckBox('原单帧对照')
                self.filter_original.setObjectName('manual_preview_filter_original');grid.addWidget(self.filter_original,3,4)
                self.filter_reset=QtWidgets.QPushButton('清空滤波历史');grid.addWidget(self.filter_reset,3,5)
                note=QtWidgets.QLabel('仅实测CCD预览：浮点RAW滤波后再做显示增强；不是80Hz全帧滤波。α越小越平滑，残影／滞后越大。')
                note.setWordWrap(True);grid.addWidget(note,4,0,1,6)
                self.filter_mode.currentIndexChanged.connect(self.filter_changed)
                self.filter_alpha.valueChanged.connect(changed)
                self.filter_original.toggled.connect(changed);self.filter_reset.clicked.connect(self.reset_filter)
                self.filter_changed()
            self.mode.currentIndexChanged.connect(self.mode_changed)
            self.gain_slider.valueChanged.connect(self.slider_changed)
            self.gain.valueChanged.connect(self.gain_changed)
            self.black.valueChanged.connect(changed); self.gamma.valueChanged.connect(changed)
            self.mode_changed()

        def parameters(self, enhance=True):
            mode = self.mode.currentData()
            if mode=='auto' and not enhance: mode='raw'
            return dict(mode=mode,gain=self.gain.value(),black=self.black.value(),gamma=self.gamma.value())

        def filter_parameters(self):
            return dict(mode=self.filter_mode.currentData(),alpha=self.filter_alpha.value(),
                        original=self.filter_original.isChecked(),generation=self.filter_generation)

        def filter_changed(self,*args):
            self.filter_alpha.setEnabled(self.filter_mode.currentData()=='ema');changed()

        def reset_filter(self):
            self.filter_generation+=1;changed()

        def mode_changed(self, *args):
            fixed = self.mode.currentData()=='fixed'
            for w in (self.gain_slider,self.gain,self.black,self.gamma): w.setEnabled(fixed)
            changed()

        def slider_changed(self, value):
            self.gain.blockSignals(True); self.gain.setValue(value/10.); self.gain.blockSignals(False)
            changed()

        def gain_changed(self, value):
            self.gain_slider.blockSignals(True); self.gain_slider.setValue(round(value*10)); self.gain_slider.blockSignals(False)
            changed()

        def weak(self):
            self.gain.setValue(16); self.black.setValue(10); self.gamma.setValue(.5)
            self.mode.setCurrentIndex(1)

    return Controls()


def attach_manual_display(window, initial=None):
    """Integrate into tab0 and intercept ONE view instance before it is painted."""
    from PyQt5 import QtCore
    from stable_views import StatusLabel
    from .filtering import PreviewTemporalFilter
    page=window.tabs.widget(0); view=window.view
    groups=tuple(v.parentWidget() for v in (view,window.sim_view,window.diff_view))
    def layouts(layout):
        if layout is None:return
        yield layout
        for i in range(layout.count()):
            item=layout.itemAt(i)
            if item.layout() is not None:yield from layouts(item.layout())
            elif item.widget() is not None:yield from layouts(item.widget().layout())
    locations=[(layout,i) for layout in layouts(page.layout()) for i in range(layout.count())
               if layout.itemAt(i).layout() is not None and
               all(layout.itemAt(i).layout().indexOf(group)>=0 for group in groups)]
    if len(locations)!=1:raise RuntimeError('unique original three-view comparison layout required')
    parent_layout,index=locations[0]
    original_set_frame=view.set_frame

    class ManualDisplay(QtCore.QObject):
        def __init__(self):
            super().__init__(page)
            self.dirty=True; self.last_key=None; self.last_saved=None; self.offline=None
            self.temporal=PreviewTemporalFilter();self.last_filter_saved=None
            self.controls=make_display_controls(page,'manual',self.changed)
            self.status=StatusLabel('显示调节只压暗／增强预览，不消除原始噪声；积分与控制仍使用原始CCD。',2)
            parent_layout.insertWidget(index,self.controls);parent_layout.insertWidget(index+1,self.status)
            if initial is not None:
                values=validate_display_parameters(initial)
                self.controls.gain.setValue(values['gain']);self.controls.black.setValue(values['black'])
                self.controls.gamma.setValue(values['gamma'])
                self.controls.mode.setCurrentIndex(self.controls.mode.findData(values['mode']))
            self.timer=QtCore.QTimer(self);self.timer.setInterval(200);self.timer.timeout.connect(self.refresh)
            self.timer.start()
            window.tabs.currentChanged.connect(lambda *_: self.temporal.reset())

        def changed(self,*args):self.dirty=True

        def render(self,packet,active):
            raw=packet['image'];meta=packet.get('meta',{})
            params=effective_parameters(window,self.controls)
            filtering=self.controls.filter_parameters()
            fresh=bool(self.offline is None and window.running and packet is window.last_packet and
                       0<=time.monotonic()-packet.get('received_monotonic',-1e30)<=1.5)
            valid_ids=all(type(meta.get(k))is int and meta[k]>=0 for k in ('TimeStamp','FrameID'))
            filtered=None
            if fresh and valid_ids and not window.fault:
                context=(getattr(window,'revision',None),packet.get('acquisition_epoch'),
                         tuple(sorted((packet.get('active')or active or {}).items())),filtering['generation'])
                filtered=self.temporal.update(raw,frame_key=(meta['TimeStamp'],meta['FrameID']),context=context,
                    mode=filtering['mode'],alpha=filtering['alpha'],now=packet['received_monotonic'])
            else:self.temporal.reset()
            key=(meta.get('FrameID'),meta.get('TimeStamp'),packet.get('received_monotonic'),tuple(params.items()),
                 tuple(filtering.items()),fresh)
            if key!=self.last_key or self.dirty or view.image is None or view.active_roi is None:
                mapped=map_display_pair(raw,filtered,**params)
                shown=mapped['unfiltered' if filtering['original']else 'filtered']
                original_set_frame(shown,active,False,prepared=True)
                self.last_key=key;self.dirty=False
            live_name='实时CCD时间滤波预览' if filtering['mode']!='off'and not filtering['original']else '实时单帧'
            state='离线历史样图，非实时' if self.offline is not None else (live_name if fresh else '已停采／旧帧，非实时')
            info=self.temporal.status
            filtering_text=('原单帧对照' if filtering['original']else filtering['mode']) if fresh and valid_ids else '滤波未累积'
            self.status.setText(f"{state} · FrameID {meta.get('FrameID','—')} · 原始峰值{int(raw.max())} ADU · "
                f"{params['mode']}：×{params['gain']:g}／黑{params['black']:g}／γ{params['gamma']:g}\n"
                f"预览{filtering_text}，样本n={info['count']}；仅显示，原始保存／差值／八区积分／电域网络不变。")
            selection=(self.controls.mode.currentData(),tuple(params.items()))
            if selection!=self.last_saved and window.hardware:
                window.record(dict(state='manual_display_controls',controls=params,selected_mode=selection[0],
                                   display_only=True,raw_or_actor_modified=False,frame_id=meta.get('FrameID')))
                self.last_saved=selection
            filter_selection=tuple(filtering.items())
            if filter_selection!=self.last_filter_saved and window.hardware:
                window.record(dict(state='manual_preview_filter_controls',controls=filtering,
                    raw_or_actor_modified=False,display_only=True,preview_sampled_frames_only=True,
                    frame_id=meta.get('FrameID')))
                self.last_filter_saved=filter_selection

        def refresh(self):
            if window.tabs.currentWidget() is not page:
                self.temporal.reset();return
            packet=self.offline if self.offline is not None else window.last_packet
            if window.fault:
                self.temporal.reset()
                self.status.setText('CCD故障／停采：'+str(window.fault)+'；显示参数不屏蔽采集或显示保护。');return
            if packet is None:return
            # Do not revive a cached image after original setText() cleared
            # startup/fault/ROI-waiting views and their drag coordinates.
            if self.offline is None and (view.image is None or view.active_roi is None):return
            try:
                active=packet.get('active')
                if active is None and self.offline is not None:
                    active=dict(OffsetX=0,OffsetY=0,Width=packet['image'].shape[1],Height=packet['image'].shape[0])
                self.render(packet,active)
            except Exception as error:self.status.setText('显示调整未采用；相机／原始数据不变：'+str(error))

    controller=ManualDisplay()
    def set_frame(self,frame,active,enhance,prepared=False):
        if window.fault:return
        packet=window.last_packet
        matched=bool(prepared and packet is not None and packet.get('active')==active and
                     (frame is packet.get('preview_raw') or frame is packet.get('preview_enhanced')))
        try:
            if matched:
                if window.tabs.currentWidget() is page:controller.render(packet,active)
                else:return None  # Do not paint old auto-mapping into a hidden manual view.
            else:return original_set_frame(frame,active,enhance,prepared=prepared)
        except Exception as error:
            controller.status.setText('显示调整未采用；原数据及控制不变：'+str(error))
            try:return original_set_frame(frame,active,enhance,prepared=prepared)
            except Exception:return None  # Display errors must not abort old tick's output checks.
    view.set_frame=MethodType(set_frame,view)
    window.manual_display=controller
    return controller


def attach_adaptation_display(window):
    """Compose local display controls; replace only ONE page's display callback."""
    from PyQt5 import QtCore
    from stable_views import StatusLabel
    from live_deploy_v20.page import detector_overlay
    page = window.adaptation_page

    class AdaptationDisplay(QtCore.QObject):
        def __init__(self):
            super().__init__(page)
            self.latest = None; self.dirty = True; self.last_key = None; self.last_saved = None
            self.controls = make_display_controls(page,'adaptation',self.changed)
            self.status = StatusLabel('等待原窗口的唯一CCD；本页不启动采集、不投图。',2)
            index = page.layout().indexOf(page.ccd)
            page.layout().insertWidget(index,self.controls); page.layout().insertWidget(index+1,self.status)
            self.timer = QtCore.QTimer(self); self.timer.setInterval(200); self.timer.timeout.connect(self.refresh)
            self.timer.start()

        def changed(self, *args):
            self.dirty = True

        def receive(self, packet):
            self.latest = packet  # Latest only, no history or signal-path transform.

        def refresh(self):
            if window.tabs.currentWidget() is not page: return
            packet = window.last_packet if window.last_packet is not None else self.latest
            if packet is None:
                self.status.setText('等待唯一CCD；无适配包也可预览，不自动投图。'); return
            meta = packet.get('meta',{})
            fresh = bool(window.running and packet is window.last_packet and
                         time.monotonic()-packet.get('received_monotonic',-1e30)<=1.5)
            params = effective_parameters(window,self.controls)
            key = (meta.get('FrameID'),meta.get('TimeStamp'),packet.get('received_monotonic'),
                   tuple(params.items()),id(page.bundle))
            try:
                if key!=self.last_key or self.dirty:
                    raw = packet['image']; result = map_display_pair(raw,**params)
                    shown = result['unfiltered']
                    if page.bundle is not None and page.bundle.weights.shape[1:]==raw.shape:
                        shown = detector_overlay(shown,page.bundle.weights)
                    page.ccd.set_image(shown,raw.shape[::-1])
                    self.last_key=key; self.dirty=False
                raw = packet['image']
                self.status.setText(f"{'实时单帧' if fresh else '已停采／旧帧，非实时'} · FrameID {meta.get('FrameID','—')} · "
                    f"原始峰值{int(raw.max())} ADU · {params['mode']} ×{params['gain']:g}／黑{params['black']:g}／γ{params['gamma']:g}\n"
                    '显示控制不进入积分或线性头；没有时间平均。')
                state = (self.controls.mode.currentData(),tuple(params.items()))
                if state!=self.last_saved and window.hardware:
                    window.record(dict(state='adaptation_display_controls',controls=params,
                        selected_mode=state[0],display_only=True,raw_or_actor_modified=False,frame_id=meta.get('FrameID')))
                    self.last_saved=state
            except Exception as error:
                self.status.setText('本页显示未采用；原相机／控制保持不变：'+str(error))

    controller = AdaptationDisplay()
    def fresh_view(self, packet, *legacy_display_parameters):
        controller.receive(packet)
    # No global/class patch, SDK construction, signal rewiring or worker change.
    page.fresh_view = MethodType(fresh_view,page)
    window.adaptation_display = controller
    return controller
