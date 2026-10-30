"""One paired diagnostic tab; reuses the existing display and camera owner."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from PyQt5 import QtCore,QtGui,QtWidgets
import core as c

# Populated once by this release's launcher, before any Qt window is built.
INITIAL_CONTROLS={}


class Image(QtWidgets.QWidget):
    def __init__(self,title):
        super().__init__();self.title=title;self.image=None;self.layout64=None
        self.setMinimumSize(130,130);self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
    def put(self,array,layout=None):
        a=np.ascontiguousarray(np.clip(array,0,255).astype(np.uint8))
        self.image=QtGui.QImage(a.data,a.shape[1],a.shape[0],a.strides[0],QtGui.QImage.Format_Grayscale8).copy()
        self.layout64=layout;self.update()
    def paintEvent(self,event):
        p=QtGui.QPainter(self);p.fillRect(self.rect(),QtGui.QColor('#101821'));p.setPen(QtGui.QColor('#e2e7eb'))
        p.drawText(8,18,self.title)
        if self.image is None:p.drawText(self.rect(),QtCore.Qt.AlignCenter,'等待本轮数据');return
        r=self.rect().adjusted(5,25,-5,-5);size=self.image.size();size.scale(r.size(),QtCore.Qt.KeepAspectRatio)
        rect=QtCore.QRect(r.center()-QtCore.QPoint(size.width()//2,size.height()//2),size);p.drawImage(rect,self.image)
        if self.layout64:
            p.setPen(QtGui.QColor(50,200,160,140))
            sx=rect.width()/self.image.width();sy=rect.height()/self.image.height()
            for i,q in enumerate(self.layout64['regions']):
                x,y,u,v=q['bounds'];box=QtCore.QRectF(rect.x()+x*sx,rect.y()+y*sy,(u-x)*sx,(v-y)*sy)
                p.drawRect(box)
                if rect.width()>350:p.drawText(box.adjusted(2,1,0,0),QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop,str(i))


class Board(QtWidgets.QWidget):
    def __init__(self,color):
        super().__init__();self.color=color;self.state=None;self.trace=[];self.setMinimumHeight(200)
    def put(self,state,reset=False):
        self.state=state
        if reset:self.trace=[]
        self.trace.append(state['position'][:2]);self.trace=self.trace[-1000:];self.update()
    def paintEvent(self,event):
        p=QtGui.QPainter(self);p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(),QtGui.QColor('#f3f5f7'))
        if self.state is None:p.drawText(self.rect(),QtCore.Qt.AlignCenter,'MuJoCo · 等待初始化');return
        roll,pitch=self.state['angles'];scale=min(self.width()/1.1,self.height()/.8)
        def xy(x,y,z=0):
            z=z+roll*y-pitch*x
            return QtCore.QPointF(self.width()/2+scale*(x-.35*y),self.height()*.49+scale*(-.50*y-.20*x-.6*z))
        corners=[xy(x,y) for x,y in ((-.3,-.3),(.3,-.3),(.3,.3),(-.3,.3))]
        p.setPen(QtGui.QPen(QtGui.QColor('#607484'),2));p.setBrush(QtGui.QColor('#d9e4eb'));p.drawPolygon(QtGui.QPolygonF(corners))
        p.setPen(QtGui.QPen(QtGui.QColor('#acbac4'),1))
        for v in np.linspace(-.3,.3,7):p.drawLine(xy(v,-.3),xy(v,.3));p.drawLine(xy(-.3,v),xy(.3,v))
        if len(self.trace)>1:
            p.setPen(QtGui.QPen(QtGui.QColor(self.color),1.6));p.drawPolyline(QtGui.QPolygonF([xy(*v,.025) for v in self.trace]))
        x,y,z=self.state['position'];p.setBrush(QtGui.QColor(self.color));p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(xy(x,y,max(0,z)),9,9);p.setPen(QtGui.QColor('#263848'))
        vel=np.linalg.norm(self.state['velocity_world']);a=self.state['filtered_action']
        text=f"t={self.state['time']:.2f}s   x={x*100:+.2f}, y={y*100:+.2f}cm   |v|={vel:.3f}m/s\n"
        text+=f"执行器目标 roll/pitch={a[0]:+.3f}/{a[1]:+.3f}rad"+('   已离板' if self.state['fallen'] else '')
        p.drawText(self.rect().adjusted(8,0,-8,-6),QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeft,text)


class Readout(QtWidgets.QWidget):
    def __init__(self,color):
        super().__init__();self.color=color;self.values=None;self.maximum=1;self.setMinimumHeight(100)
        self.title='64区RAW积分 · 等待首个有效帧'
    def put(self,values,maximum):
        self.values=np.asarray(values);self.maximum=max(1.,maximum)
        self.setToolTip('\n'.join(f'R{i}: {v:.6f} ADU·pixel' for i,v in enumerate(self.values)));self.update()
    def paintEvent(self,event):
        p=QtGui.QPainter(self);p.fillRect(self.rect(),QtGui.QColor('white'));p.setPen(QtGui.QColor('#263848'))
        p.drawText(7,16,self.title)
        if self.values is None:return
        width=(self.width()-12)/64;h=self.height()-40
        p.setPen(QtCore.Qt.NoPen);p.setBrush(QtGui.QColor(self.color))
        for i,v in enumerate(self.values):p.drawRect(QtCore.QRectF(6+i*width,23+h*(1-v/self.maximum),max(.5,width-1),h*v/self.maximum))
        p.setPen(QtGui.QColor('#263848'));p.drawText(7,self.height()-3,'R0');p.drawText(self.width()-32,self.height()-3,'R63')


def save_capture(path,result,spec,context,revision):
    with path.open('xb') as f:np.savez(f,raw=result['raw'],image=result['image'])
    c.write(path.with_suffix('.json'),dict(frames=result['meta'],filter=asdict(spec),camera_context=context,
        revision=revision,source='existing owner continuous native RAW',archive=c.ref(path),
        independent_filter_reset=True,display_gain_NOT_applied=True,
        saturation_pixels=int(np.count_nonzero(result['raw']==255)),brightness_policy='warning_only'))
    return path


class Controller(QtCore.QObject):
    def __init__(self,w,package_ref,output):
        super().__init__(w);self.w=w;self.package_ref=package_ref;self.package=c.read(c.checked(package_ref))
        self.preview=c.load(c.PREVIEW/'model.py','e107_page_contract');self.manifest=c.read(c.checked(self.package['model']))
        self.output=Path(output);self.root=None;self.seq=0;self.busy=None;self.inited=False
        self.go=False;self.one=False;self.stage='IDLE';self.observation=None;self.saved_input=None
        self.presented=None;self.pending_ccd=None
        self.pending_reset=None;self.pending_impulse=[0.,0.];self.stderr='';self.buffer=b''
        self.burst=c.Burst();self.writer=ThreadPoolExecutor(1,thread_name_prefix='paired-archive');self.future=None
        self.rng=np.random.default_rng(107);self.started=None;self.guard=None
        self.process=QtCore.QProcess(self);self.process.readyReadStandardOutput.connect(self.received)
        self.process.readyReadStandardError.connect(self.drain);self.process.started.connect(self.send_pending)
        self.process.finished.connect(self.finished);self.process.errorOccurred.connect(lambda _:self.fail('模型进程错误：'+self.process.errorString()))
        env=QtCore.QProcessEnvironment.systemEnvironment()
        for k,v in {'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','PYTHONPATH':str(c.LAB/'src'),'MUJOCO_GL':'egl'}.items():env.insert(k,v)
        self.process.setProcessEnvironment(env)
        self.build();self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(40)
        w.stop_button.clicked.connect(self.pause)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.close)

    def build(self):
        self.page=QtWidgets.QWidget();body=QtWidgets.QVBoxLayout(self.page)
        title=QtWidgets.QLabel('仿真／真实光路控制对比 · 固定SLM · 两个独立64→2头（各130参数） · 诊断，非部署验收')
        title.setStyleSheet('font-size:15px;font-weight:bold');body.addWidget(title)
        bar=QtWidgets.QHBoxLayout();body.addLayout(bar)
        for text,fn in [('开始／继续',lambda:self.start(False)),('单步',lambda:self.start(True)),('暂停／全0',self.pause),('中心复位',lambda:self.reset([0,0]))]:
            button=QtWidgets.QPushButton(text);bar.addWidget(button);button.clicked.connect(fn)
        self.radius=QtWidgets.QDoubleSpinBox();self.radius.setRange(0,20);self.radius.setValue(INITIAL_CONTROLS.get('radius_cm',6));self.radius.setSuffix(' cm')
        bar.addWidget(QtWidgets.QLabel('随机复位半径'));bar.addWidget(self.radius)
        random=QtWidgets.QPushButton('同位置随机复位');bar.addWidget(random)
        random.clicked.connect(lambda:self.reset(c.random_xy(self.rng,self.radius.value()).tolist()))
        self.impulse=QtWidgets.QDoubleSpinBox();self.impulse.setDecimals(4);self.impulse.setRange(0,.1);self.impulse.setValue(INITIAL_CONTROLS.get('impulse_Ns',.005));self.impulse.setSingleStep(.001);self.impulse.setSuffix(' N·s')
        self.angle=QtWidgets.QDoubleSpinBox();self.angle.setRange(-180,180);self.angle.setValue(INITIAL_CONTROLS.get('angle_deg',0));self.angle.setSuffix(' °')
        bar.addWidget(QtWidgets.QLabel('冲量／方向'));bar.addWidget(self.impulse);bar.addWidget(self.angle)
        push=QtWidgets.QPushButton('同时扰动');bar.addWidget(push);push.clicked.connect(self.push)
        self.info=QtWidgets.QLabel('参数继承主界面。等候有效帧时MuJoCo暂停；0°=世界+X，90°=世界+Y。未初始化、未投图。')
        self.info.setWordWrap(True);self.info.setFixedHeight(44);body.addWidget(self.info)
        columns=QtWidgets.QHBoxLayout();body.addLayout(columns,1);self.views=[]
        for title,color in [('仿真光路：物理＋误差模型','#0072B2'),('真实光路：实际CCD → MuJoCo','#D55E00')]:
            box=QtWidgets.QGroupBox(title);layout=QtWidgets.QVBoxLayout(box);columns.addWidget(box,1)
            row=QtWidgets.QHBoxLayout();layout.addLayout(row)
            dmd=Image('DMD · 本侧4幅历史');slm=Image('SLM · 相同固定灰度mask')
            row.addWidget(dmd);row.addWidget(slm)
            row2=QtWidgets.QHBoxLayout();layout.addLayout(row2,1)
            ccd=Image('CCD · 本侧有效帧，显示增强不参与积分');board=Board(color)
            row2.addWidget(ccd,1);row2.addWidget(board,1)
            bars=Readout(color);layout.addWidget(bars)
            self.views.append(dict(dmd=dmd,slm=slm,ccd=ccd,board=board,bars=bars))
        self.note=QtWidgets.QLabel('两侧各自闭环，复位和扰动一致；轨迹分开后不应逐像素比较两侧CCD。原始帧、输入、积分、动作及轨迹自动归档。')
        self.note.setWordWrap(True);self.note.setFixedHeight(44);body.addWidget(self.note)
        self.index=self.w.tabs.addTab(self.page,'光路控制对比');self.page.setObjectName('paired_control_page')

    def context(self):
        w=self.w;packet=w.last_packet or {}
        if not w.hardware or not w.running or not w.live_ready or w.fault or w.worker is None:
            raise ValueError('请先在主界面恢复CCD；当前没有可用的真实采集')
        if time.monotonic()-packet.get('received_monotonic',0)>1.5:raise ValueError('CCD参数已过期')
        if w.stability_workflow.active or any(getattr(w,k,False) for k in ('override','roi_busy','reference_busy','spatial_busy')):
            raise ValueError('请结束主界面的测试／标定后再运行')
        context=dict(camera=dict(packet['active']),roi=dict(w.actual_roi),settings=asdict(w.setting()),
            optics=asdict(w.optical_config()),geometry=deepcopy(w.registration),orientation=w.orientation.currentData(),dmd_side=w._dmd_side)
        self.preview.validate_context(context,self.manifest)
        mask=np.asarray(w.slm_file)
        if not np.array_equal(mask,np.load(c.checked(self.package['mask']))):raise ValueError('电子头绑定原BASE mask；不会自动替换当前SLM')
        layout=w.region_overlay.current_layout(w.actual_roi)
        if [r['bounds'] for r in layout['regions']]!=[r['bounds'] for r in self.package['layout']['regions']]:
            raise ValueError('请选择与模型一致的64区读出')
        origin=getattr(w,'ball_input_origin',None)
        if origin is not None and origin['geometry'].get('radius_raw_px')!=self.package['radius']:
            raise ValueError('当前已应用球大小与训练R2不一致；不会偷偷覆盖')
        return context

    def start(self,one=False):
        if self.go or self.busy is not None:return
        try:
            from optical_lab.effective_frame import from_controls
            self.ctx=self.context();self.spec=from_controls(self.w.manual_display.controls.filter_parameters())
            self.go=True;self.one=one;self.started=time.monotonic();self.lock(True)
            if self.saved_input is None:self.saved_input=(self.w.dmd_file.copy(),self.w.dmd_mode.currentData())
            if not self.inited:
                if self.process.state()!=QtCore.QProcess.NotRunning:raise ValueError('上一模型进程正在释放，请稍后再开始')
                self.root=self.output/('RUN_'+str(time.time_ns()));self.output.mkdir(parents=True,exist_ok=True)
                self.send('init')
                self.process.start('/home/user/miniconda3/bin/python',['-B',str(c.ROOT/'worker.py'),
                    '--package',self.package_ref['path'],'--sha256',self.package_ref['sha256'],'--output',str(self.root)])
            else:self.prepare()
        except Exception as e:self.fail(str(e))

    def lock(self,on):
        for i in (0,1):self.w.tabs.setTabEnabled(i,not on)

    def send(self,op,**kwargs):
        if self.busy is not None:raise ValueError('another worker command in flight')
        self.seq+=1;self.busy=dict(id=self.seq,op=op,**kwargs);self.command_at=time.monotonic()
        self.stage='WORKER'
        if self.process.state()==QtCore.QProcess.Running:self.send_pending()

    def send_pending(self):
        if self.busy is not None:self.process.write((json.dumps(self.busy,allow_nan=False)+'\n').encode())

    def prepare(self):
        if self.go:
            self.hold_view('等待下一帧')
            self.send('prepare',context=self.ctx)

    def received(self):
        self.buffer+=bytes(self.process.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            if not line.startswith(b'E107 '):continue
            try:self.accept(json.loads(line[5:]))
            except Exception as e:self.fail(str(e))

    def accept(self,message):
        if self.busy is None or message.get('id')!=self.busy['id']:return
        request=self.busy;self.busy=None
        if not message['ok']:
            if request['op']=='init':self.process.closeWriteChannel()
            self.fail(message['error']);return
        data=message['result'];op=request['op']
        if op=='init':self.inited=True;self.clear_view();self.show_states(data['states'],True)
        elif op=='reset':
            self.observation=None;self.pending_impulse=[0.,0.];self.clear_view();self.show_states(data['states'],True)
        elif op=='prepare':
            self.observation=dict(**data,id=request['id'])
            with np.load(data['archive'],allow_pickle=False) as a:self.images={k:a[k].copy() for k in a.files}
            # A complete displayed observation is latched independently of the
            # next input/RAW window. Never blank it or mix two acquisition times.
            self.pending_ccd=None
            self.hold_view('换图稳定中')
            self.show_states(data['states'])
            if self.go:
                w=self.w;w.dmd_file=self.images['real_DMD'].copy()
                w.dmd_mode.setCurrentIndex(w.dmd_mode.findData('file'));w.changed();w.armed.setChecked(True);w.render_patterns()
                self.revision=w.revision;self.settle_at=None;self.loaded_at=time.monotonic();self.stage='SETTLE';return
        elif op=='advance':
            self.show_states(data['states_after']);self.observation=None
            self.commit_view(data)
            if any(s['fallen'] for s in data['states_after']):self.pause();self.info.setText('球已离板；数据保留，请复位。');return
            if self.one:self.pause()
        if self.pending_reset is not None:
            xy=self.pending_reset;self.pending_reset=None;self.send('reset',xy=xy)
        elif not self.go:
            if op=='prepare':self.send('cancel')
            else:self.stage='IDLE';self.lock(False)
        else:self.prepare()

    def poll(self):
        if not self.go:return
        try:
            if self.w.fault or not self.w.running:raise ValueError('相机/显示故障；不推进MuJoCo')
            if time.monotonic()-self.started>300:
                self.pause();self.info.setText('本批已到5分钟，已暂停并全0；可手动继续同一仿真状态。');return
            if self.busy and time.monotonic()-self.command_at>90:raise ValueError('计算进程超时')
            if self.stage in ('SETTLE','CAPTURE','SAVING'):
                if not self.w.armed.isChecked():raise ValueError('投图已被停止')
                if self.w.revision!=self.revision:raise ValueError('本次采样期间投图发生改变')
            if self.stage=='SETTLE':
                if time.monotonic()-self.loaded_at>20:raise ValueError('等待当前图案核验超时')
                if self.w.last_verified_revision!=self.revision:return
                if self.settle_at is None:self.settle_at=time.monotonic()+1.5
                if time.monotonic()<self.settle_at:return
                self.ctx=self.context();worker=self.w.worker
                self.burst.arm(id(worker),worker.epoch,self.ctx['camera'],self.spec,self.package['layout']['shape'])
                self.stage='CAPTURE';self.capture_at=time.monotonic();self.hold_view('采集中')
            elif self.stage=='CAPTURE':
                result,count,error=self.burst.snapshot()
                if error:raise ValueError(error)
                if time.monotonic()-self.capture_at>30:raise ValueError('RAW独立窗超时')
                if result is not None:
                    path=Path(self.observation['archive']).parent/('real_RAW_'+str(time.time_ns())+'.npz')
                    self.future=self.writer.submit(save_capture,path,result,self.spec,self.ctx,self.revision)
                    self.pending_ccd=result['image'].copy()
                    self.stage='SAVING';self.hold_view('归档与积分中')
                else:self.info.setText(f'真实CCD独立滤波 {count}/{self.spec.frames}；MuJoCo时间冻结。')
            elif self.stage=='SAVING' and self.future.done():
                path=self.future.result();self.future=None
                impulse=self.pending_impulse;self.pending_impulse=[0.,0.]
                self.send('advance',observation_id=self.observation['id'],archive=str(path),impulse=impulse)
        except Exception as e:self.fail(str(e))

    def display(self,image):
        from optical_lab.display import effective_parameters,map_display_pair
        params=effective_parameters(self.w,self.w.manual_display.controls)
        a=np.asarray(image)
        if a.ndim!=2 or not a.size or not np.isfinite(a).all():raise ValueError('finite 2D CCD required for display')
        # Only the auto-level anchor needs Mono8. Keep the actual float mean
        # in the filtered argument; no display quantization enters readout.
        anchor=np.rint(np.clip(a,0,255)).astype(np.uint8)
        return map_display_pair(anchor,a,**params)['filtered']

    def hold_view(self,phase):
        """Labels only: displayed history must never enter the control path."""
        last=self.presented
        stamp=f"t={last['time']:.2f}s" if last is not None else None
        for v in self.views:
            v['ccd'].title=f'CCD · 上一有效帧 {stamp} · {phase}' if stamp else f'CCD · 等待首个有效帧 · {phase}'
            v['ccd'].update()
            v['bars'].title=f'64区RAW积分 · {stamp} · 上一有效帧' if stamp else '64区RAW积分 · 等待首个有效帧'
            v['bars'].update()
        self.note.setText((f'保留上一组完整DMD/SLM/CCD/积分：{stamp}；{phase}，不是正在采集的新图。'
            if stamp else f'{phase}，等待首组完整观测；尚无可保留的有效帧。')
            +' 仅预览保留旧结果；控制仍等待新RAW独立窗，不重复使用旧帧。')

    def clear_view(self):
        self.presented=None;self.pending_ccd=None
        for v in self.views:
            for key in ('dmd','slm','ccd'):
                v[key].image=None;v[key].layout64=None;v[key].update()
            v['bars'].values=None;v['bars'].setToolTip('');v['bars'].update()
        self.hold_view('复位后等待')

    def commit_view(self,data):
        if self.pending_ccd is None:raise ValueError('新有效帧尚未完成；不以预览旧帧替代')
        # Prepare both display maps before modifying any visible widget. Qt
        # paints after this slot returns, so the two images/bars update together.
        displayed=[self.display(self.images['simulation_CCD']),self.display(self.pending_ccd)]
        maximum=max(max(data['simulation_sums']),max(data['real_sums']))
        for i,v in enumerate(self.views):
            v['dmd'].put(self.images[('simulation_DMD','real_DMD')[i]])
            v['slm'].put(self.images['SLM']);v['ccd'].put(displayed[i],self.package['layout'])
            v['bars'].put(data[('simulation_sums','real_sums')[i]],maximum)
        self.presented=dict(time=data['states_before'][0]['time'],episode=data['episode'],step=data['step'])
        self.pending_ccd=None;self.hold_view('本步已完成')

    def show_states(self,states,reset=False):
        for v,s in zip(self.views,states):v['board'].put(s,reset)
        elapsed=time.monotonic()-self.started if self.started else 0
        spec=getattr(self,'spec',None)
        filtering=(f'当前{spec.mode}, N={spec.frames}, α={spec.alpha:g}；拟合数据为N20均值。' if spec else '')
        self.info.setText(f"仿真时间 {states[0]['time']:.2f}s ｜本批现实耗时 {elapsed:.1f}s ｜等待采样不计入动力学时间。"
            +filtering+'预测均值不代表真实噪声；控制尚未验收。')

    def zero(self):
        if self.w.hardware:self.w.armed.setChecked(False);self.w.black_outputs()

    def pause(self,*_):
        owned=self.go or self.saved_input is not None
        self.go=False;self.burst.cancel()
        if owned:
            self.zero()
            if self.saved_input is not None:
                image,mode=self.saved_input;self.saved_input=None
                self.w.dmd_file=image;self.w.dmd_mode.setCurrentIndex(self.w.dmd_mode.findData(mode));self.w.changed()
        if self.busy is None:
            if self.inited and self.observation is not None:self.observation=None;self.send('cancel')
            else:self.stage='IDLE';self.lock(False)
        if self.busy is None or self.busy['op']!='advance':self.pending_ccd=None
        self.hold_view('已暂停')
        self.info.setText('已暂停，MuJoCo保持当前状态；已发送的单个动力学步完成后不再推进。')

    def reset(self,xy):
        self.pause();self.pending_reset=xy
        if self.inited and self.busy is None:
            self.pending_reset=None;self.send('reset',xy=xy)
        elif not self.inited:self.info.setText('已设置下次初始化的同步复位位置；点击开始／单步。')

    def push(self):
        self.pending_impulse=[self.impulse.value(),self.angle.value()]
        self.info.setText(f'已排队：两侧下一次动力学步前施加 {self.pending_impulse[0]:.4f} N·s，{self.pending_impulse[1]:.1f}°；暂停时不消耗。')

    def fail(self,error):
        self.pause();self.info.setText('未继续：'+str(error)+'。现有数据保留，不使用旧CCD帧替代。')

    def drain(self):self.stderr=(self.stderr+bytes(self.process.readAllStandardError()).decode(errors='replace'))[-8000:]
    def finished(self,*_):
        pending=self.busy;self.busy=None;self.inited=False;self.observation=None
        if pending is not None or self.go:self.fail('计算进程退出：'+self.stderr[-800:])
    def close(self):
        self.timer.stop();self.pause();self.writer.shutdown(wait=False)
        if self.process.state()!=QtCore.QProcess.NotRunning:
            self.process.closeWriteChannel()
            if not self.process.waitForFinished(1000):self.process.terminate();self.process.waitForFinished(1000)


def attach(w,package_ref,output):
    if w.worker is not None:raise ValueError('new page must be installed before camera start')
    controller=Controller(w,package_ref,output)
    import live_loop_v23.direct_alignment_gui as gui
    original=gui.LoopWorker.capture_one
    def capture(worker,*args,**kwargs):
        ticket=controller.burst.ticket()
        result=original(worker,*args,**kwargs)
        if not kwargs.get('dark',args[0] if args else False) and ticket is not None:
            raw,meta=result
            controller.burst.feed(ticket,raw,meta,id(worker),getattr(worker,'epoch',None),dict(worker.camera.active))
        return result
    gui.LoopWorker.capture_one=capture
    w.paired_control=controller
    return controller
