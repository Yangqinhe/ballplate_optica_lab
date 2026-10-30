"""One live pipeline page; passive views, bounded drawing, no SDK ownership."""
import time
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from stable_views import SensorView, StatusLabel, SharedSensorScale
from diffraction_model import compare_shape, normalize_shape
from live_deploy_v20.page import detector_overlay
from live_deploy_v21.page import display_signal
from .core import DEFAULT_HEAD, DEFAULT_SHA, load_head


class TraceView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.rows = []; self.setMinimumHeight(95)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def append(self, data):
        if self.rows and data['step'] <= self.rows[-1]['step']: return
        self.rows.append(data); self.rows = self.rows[-500:]; self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self); p.fillRect(self.rect(), QtGui.QColor('#15202b'))
        p.setPen(QtGui.QColor('white')); p.drawText(8, 17, '位置半径 mm（青） / 速度 mm/s（橙）；横轴 = MuJoCo 仿真时间')
        if not self.rows: p.drawText(8, 43, '等待真实光路控制步…'); return
        r = np.array([np.linalg.norm(x['position_m'][:2])*1000 for x in self.rows])
        v = np.array([np.linalg.norm(x['velocity_world_m_s'])*1000 for x in self.rows])
        ts = np.array([x['simulation_time_s'] for x in self.rows]); top = max(50.,float(max(r.max(),v.max())))
        for values, color in ((r,'#25d9de'),(v,'#ffa65b')):
            points = [QtCore.QPointF(40+(self.width()-65)*t/max(ts[-1],.02),
                self.height()-12-(self.height()-42)*y/top) for t,y in zip(ts,values)]
            p.setPen(QtGui.QPen(QtGui.QColor(color),2)); p.drawPolyline(QtGui.QPolygonF(points))
        p.setPen(QtGui.QColor('white')); p.drawText(self.width()-125,17,f'{ts[-1]:.2f}s / {top:.0f}')


class LoopPage(QtWidgets.QWidget):
    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    push_requested = QtCore.pyqtSignal(int,int,float)

    def __init__(self):
        super().__init__(); self.busy=False; self.head_path=None; self.head=None; self.bundle=None
        self.last = {}; self.started=0.; self.latest_step=-1; self.pending_ids=set()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel('真实光路 → 单帧CCD8积分 → Linear(8,2) → 仅MuJoCo；四幅RAW历史180/120/60/0 ms'))
        row = QtWidgets.QHBoxLayout(); layout.addLayout(row)
        self.load = QtWidgets.QPushButton('载入已适配的18参数头'); self.load.clicked.connect(self.choose)
        row.addWidget(self.load); self.path = QtWidgets.QLineEdit(); self.path.setReadOnly(True); row.addWidget(self.path,1)
        row = QtWidgets.QHBoxLayout(); layout.addLayout(row)
        self.confirm = QtWidgets.QCheckBox('原ND／低功率、防护和器件位置未变；允许本轮有界投图')
        self.confirm.toggled.connect(self.buttons); row.addWidget(self.confirm,1)
        self.seconds = QtWidgets.QDoubleSpinBox(); self.seconds.setRange(.4,10); self.seconds.setSingleStep(.2); self.seconds.setValue(3); self.seconds.setSuffix(' s仿真')
        row.addWidget(self.seconds)
        self.x = QtWidgets.QDoubleSpinBox(); self.y = QtWidgets.QDoubleSpinBox()
        for w,label,value in ((self.x,'初始X ',25),(self.y,'初始Y ',-25)):
            w.setRange(-70,70); w.setValue(value); w.setPrefix(label); w.setSuffix(' mm'); row.addWidget(w)
        self.start = QtWidgets.QPushButton('开始／重新复位'); self.start.clicked.connect(self.start_requested); row.addWidget(self.start)
        self.stop = QtWidgets.QPushButton('停止／两路全0'); self.stop.clicked.connect(self.stop_requested); row.addWidget(self.stop)
        row = QtWidgets.QHBoxLayout(); layout.addLayout(row)
        row.addWidget(QtWidgets.QLabel('手动速度冲击（仅MuJoCo）：'))
        self.speed = QtWidgets.QDoubleSpinBox(); self.speed.setRange(.01,.15); self.speed.setSingleStep(.01); self.speed.setValue(.06); self.speed.setDecimals(2); self.speed.setSuffix(' m/s'); row.addWidget(self.speed)
        self.push_buttons = []
        for label,axis,sign in (('← −X',0,-1),('→ +X',0,1),('↓ −Y',1,-1),('↑ +Y',1,1)):
            b = QtWidgets.QPushButton(label); b.clicked.connect(lambda _,a=axis,s=sign:self.push_requested.emit(a,s,self.speed.value()))
            row.addWidget(b); self.push_buttons.append(b)
        self.push_status=StatusLabel('未施加扰动',2); row.addWidget(self.push_status,1)
        self.status=StatusLabel('启动默认全0、不采集；先启动顶部“仅CCD”，再确认本轮条件。',2)
        self.status.setWordWrap(True); layout.addWidget(self.status)
        grid=QtWidgets.QGridLayout(); layout.addLayout(grid,1); self.views={}; self.labels={}
        titles=[('rgb','MuJoCo · 当前物理状态'),('dmd','DMD · 四幅二值图（旧→新）'),
            ('phase','SLM · 规范600²有效mask／4×4宏像素'),('ideal','理想CCD · 当前冻结模型，仅显示'),
            ('actual','实际CCD · 单张原图，仅显示增益'),('overlay','叠加 · 红=实测，绿=理想；独立归一化')]
        for i,(key,title) in enumerate(titles):
            box=QtWidgets.QWidget(); cell=QtWidgets.QVBoxLayout(box); cell.setContentsMargins(2,2,2,2)
            label=StatusLabel(title,2); cell.addWidget(label); self.labels[key]=label
            view=SensorView('等待本轮数据'); view.setMinimumSize(240,200); cell.addWidget(view,1)
            self.views[key]=view; grid.addWidget(box,i//3,i%3)
        self.shared_scale=SharedSensorScale([self.views[k] for k in ('ideal','actual','overlay')],self)
        self.views['phase'].setToolTip('规范有效区权重；实际SLM送屏沿用原输出窗口的一次水平翻转和已标定位置。')
        row=QtWidgets.QHBoxLayout(); layout.addLayout(row)
        row.addWidget(QtWidgets.QLabel('CCD显示增益'))
        self.gain=QtWidgets.QDoubleSpinBox(); self.gain.setRange(1,128); self.gain.setValue(16); row.addWidget(self.gain)
        row.addWidget(QtWidgets.QLabel('黑电平'))
        self.black=QtWidgets.QDoubleSpinBox(); self.black.setRange(0,100); self.black.setValue(10); row.addWidget(self.black)
        row.addWidget(QtWidgets.QLabel('γ'))
        self.gamma=QtWidgets.QDoubleSpinBox(); self.gamma.setRange(.1,3); self.gamma.setValue(.5); row.addWidget(self.gamma)
        self.regions=QtWidgets.QCheckBox('显示固定8区'); self.regions.setChecked(True); row.addWidget(self.regions)
        row.addWidget(QtWidgets.QLabel('所有增强仅用于显示，不改变积分或策略输入'),1)
        for control in (self.gain,self.black,self.gamma):control.valueChanged.connect(self.redraw_ccd)
        self.regions.toggled.connect(self.redraw_ccd)
        self.values=StatusLabel('CCD8 → FC18 → 两轴倾角：—',2); layout.addWidget(self.values)
        self.trace=TraceView(); layout.addWidget(self.trace)
        self.note=StatusLabel('实际查询时MuJoCo暂停；软件等待≠DMD/SLM/CCD硬件同步，未通过20ms实时验收。SLM为先验−200mm二次相位，非新学习权重。',2)
        self.note.setWordWrap(True); layout.addWidget(self.note)
        self.buttons()

    def choose(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'载入真实单帧线性头',str(DEFAULT_HEAD.parent),'HEAD (*.json)')
        if path:
            try:self.load_head(path)
            except Exception as e:self.status.setText('未载入：'+str(e))

    def load_head(self,path=DEFAULT_HEAD,sha=None):
        if self.busy:raise RuntimeError('运行中不可更换头')
        head,bundle=load_head(path,sha)
        self.head_path=type(DEFAULT_HEAD)(path).resolve();self.head=head;self.bundle=bundle
        self.detector_template=detector_overlay(np.zeros(bundle.weights.shape[1:],np.uint8),bundle.weights)
        self.detector_lines=np.any(self.detector_template!=0,axis=-1)
        self.path.setText(str(self.head_path));self.views['phase'].set_image(bundle.phase)
        self.note.setText('四图600² → SLM先验−200mm二次相位（600²，4×4实际像素／有效像素） → CCD单帧 → 8区积分 → 18参数FC。\n实际查询时MuJoCo暂停；无实物执行器、无合成CCD兜底；未通过20ms实时验收。')
        self.buttons()

    def buttons(self):
        self.start.setEnabled(bool(self.head_path and self.confirm.isChecked() and not self.busy))
        self.stop.setEnabled(self.busy)
        for w in (self.load,self.confirm,self.seconds,self.x,self.y):w.setEnabled(not self.busy)
        for b in self.push_buttons:b.setEnabled(self.busy)

    def set_busy(self,busy):
        self.busy=busy
        if busy:
            self.started=time.monotonic();self.last={};self.latest_step=-1;self.pending_ids.clear()
            self.trace.rows=[];self.trace.update();self.push_status.setText('等待手动扰动')
            for key,view in self.views.items():
                if key!='phase':view.setText('等待本轮真实数据（不是历史回放）')
        self.buttons()

    def queued(self,event):
        self.pending_ids.add(event['id'])
        self.push_status.setText(f'已排队{len(self.pending_ids)}个，下一控制步施加：{event["delta_v_m_s"]} m/s')

    def on_view(self,data):
        self.last=dict(data)
        for key in ('dmd','phase'):
            if key in data:self.views[key].set_image(data[key])
        if 'rgb' in data:self.views['rgb'].set_image(data['rgb'])
        if data['kind']=='mujoco_step':self.views['rgb'].set_image(data['rgb_image'])
        for step in data.get('trace_rows',[]):
            if step['step']<=self.latest_step:continue
            self.trace.append(step);self.latest_step=step['step']
            for e in step['push_events_applied']:self.pending_ids.discard(e['id'])
            if step['push_events_applied']:
                self.push_status.setText(f'已施加于t={step["simulation_time_s"]:.2f}s；待施加{len(self.pending_ids)}个')
            values=', '.join(f'{v:.0f}' for v in step['raw_integrals8'])
            a=step['action_limited_rad'];pos=np.linalg.norm(step['position_m'][:2])*1000
            self.values.setText(f'原始单帧积分8：[ {values} ] → FC18 → [ {a[0]:+.4f}, {a[1]:+.4f} ] rad；半径{pos:.1f} mm，reward={step["reward"]:.3f}')
        sim=data.get('simulation_time_s',0);wall=time.monotonic()-self.started
        phase='等待当前输入的实际CCD' if data['kind']=='input' else '已完成真实光路控制步'
        self.status.setText(f'{phase} · 仿真{sim:.2f}/{self.seconds.value():.2f}s · 墙钟{wall:.1f}s · 最近查询{data.get("query_latency_s",0):.2f}s；不是实时回放')
        self.redraw_ccd()

    def redraw_ccd(self,*args):
        d=self.last
        if 'actual' not in d or 'ideal' not in d:return
        a=d['actual'];display=display_signal(a,self.gain.value(),self.black.value(),self.gamma.value())
        if self.regions.isChecked():
            display=np.repeat(display[:,:,None],3,axis=-1)
            display[self.detector_lines]=self.detector_template[self.detector_lines]
        self.views['actual'].set_image(display,a.shape[::-1])
        ideal,_=normalize_shape(d['ideal']);self.views['ideal'].set_image(np.round(ideal*255).astype(np.uint8),a.shape[::-1])
        comp=compare_shape(a,d['ideal'],self.bundle.contract['expected_active'])
        self.views['overlay'].set_image(comp['overlay_rgb'],a.shape[::-1])
        n=d['actual_sequence'];fid=d['frame_id'];current=d.get('sequence',n)
        note=f'步{n} / raw FrameID {fid}'
        if n!=current:note+=f'（当前步{current}等待中，保留上一实测）'
        self.labels['actual'].setText('实际CCD · '+note)
        self.labels['ideal'].setText(f'理想CCD · 与实测步{n}相同输入；只作模型对照')
        ncc=comp['metrics']['correlation'];self.labels['overlay'].setText('红实测／绿理想 · 归一化形状NCC '+('无有效值' if ncc is None else f'{ncc:.3f}'))

    def finished(self,result):
        self.set_busy(False);r=result['report']
        self.status.setText(('本轮完成' if r['completed'] and not r['error'] else '本轮停止／未完整通过')+
            f' · {r["actions"]}步 · '+str(r.get('error') or '已请求两路全0；可改初始位置再开始'))
        self.status.setToolTip(result['folder'])
        if self.pending_ids:self.push_status.setText(f'{len(self.pending_ids)}个排队扰动未施加，已废弃')
        self.pending_ids.clear()
