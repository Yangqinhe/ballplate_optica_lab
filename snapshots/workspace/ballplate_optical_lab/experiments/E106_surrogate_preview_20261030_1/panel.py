"""One extra mean-prediction view. Never sends device/camera/control commands."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
import numpy as np


def attach(window,manifest_path,core):
    from PyQt5 import QtCore,QtWidgets
    from stable_views import SensorView,StatusLabel
    from optical_lab.display import map_display_pair,effective_parameters
    manifest=core.read(manifest_path)
    page=window.tabs.widget(0)
    groups=[v.parentWidget() for v in (window.view,window.sim_view,window.diff_view)]
    def layouts(layout):
        if layout is None:return
        yield layout
        for i in range(layout.count()):
            item=layout.itemAt(i)
            if item.layout() is not None:yield from layouts(item.layout())
            if item.widget() is not None:yield from layouts(item.widget().layout())
    rows=[layout for layout in layouts(page.layout()) if all(layout.indexOf(g)>=0 for g in groups)]
    if len(rows)!=1:raise ValueError('unique three-image layout required')
    row=rows[0]
    box=QtWidgets.QGroupBox('平台误差模型CCD · 预测均值',page)
    body=QtWidgets.QVBoxLayout(box)
    enabled=QtWidgets.QCheckBox('显示预测（不参与控制）');enabled.setChecked(True)
    enabled.setObjectName('surrogate_preview_enabled');body.addWidget(enabled)
    choice=QtWidgets.QComboBox();choice.setObjectName('surrogate_preview_source')
    choice.addItem('物理＋平台误差模型','corrected');choice.addItem('校正物理基线（不加残差）','physical')
    # Controls/status stay outside the image row: all four sensor views retain
    # identical available height, even on a smaller ordinary monitor.
    body.removeWidget(enabled)
    toolbar=QtWidgets.QHBoxLayout();toolbar.addWidget(enabled);toolbar.addWidget(choice)
    status=StatusLabel('E105冻结模型；预测均值，不添加随机散斑，不是新SLM训练结果。',3)
    toolbar.addWidget(status,1)
    parents=[(layout,i) for layout in layouts(page.layout()) for i in range(layout.count())
             if layout.itemAt(i).layout() is row]
    if len(parents)!=1:raise ValueError('comparison row parent missing')
    parent,index=parents[0];parent.insertLayout(index,toolbar)
    view=SensorView('等待当前输入与相机配置');view.setObjectName('surrogate_ccd_view');body.addWidget(view,1)
    row.insertWidget(row.indexOf(groups[2]),box,1)
    groups[1].setTitle('对齐参考ASM · 原显示模型')
    groups[2].setTitle('原对齐参考的差值／叠加')
    # Same native extent and fit, including the new view; no layout jitter.
    shared=window.shared_sensor_scale;shared.views=tuple(shared.views)+(view,)
    view.installEventFilter(shared);shared.queue_refresh()

    class Preview(QtCore.QObject):
        def __init__(self):
            super().__init__(page)
            self.view=view;self.status=status;self.enabled=enabled;self.choice=choice
            self.manifest=manifest;self.result=None;self.metadata=None;self.current_key=None
            self.busy=None;self.serial=0;self.request=None;self.error=None;self.buffer=b''
            self.folder=tempfile.TemporaryDirectory(prefix='ballplate-mean-preview-')
            self.process=QtCore.QProcess(self)
            env=QtCore.QProcessEnvironment.systemEnvironment()
            env.insert('OPENBLAS_NUM_THREADS','2');env.insert('OMP_NUM_THREADS','2')
            env.insert('PYTHONPATH',str(core.LAB/'src'));self.process.setProcessEnvironment(env)
            self.process.started.connect(self.send)
            self.process.readyReadStandardOutput.connect(self.received)
            self.process.readyReadStandardError.connect(self.drain_error)
            self.process.errorOccurred.connect(self.process_error)
            self.process.finished.connect(self.finished)
            self.stderr='';self.stopping=False;self.send_after_start=False
            self.timer=QtCore.QTimer(self);self.timer.setInterval(500);self.timer.timeout.connect(self.refresh)
            self.timer.start();enabled.toggled.connect(self.toggle);choice.currentIndexChanged.connect(self.paint)
            QtWidgets.QApplication.instance().aboutToQuit.connect(self.close)

        def snapshot(self):
            if window.fault:raise ValueError('相机/显示故障；模型预览暂停')
            if getattr(window,'dirty',False):raise ValueError('图案正在更新，等待新请求')
            if any(getattr(window,k,False) for k in ('override','roi_busy','reference_busy','deployment_busy')):
                raise ValueError('主界面正在重配或标定；模型预览暂停')
            flow=getattr(window,'stability_workflow',None)
            if flow is not None and flow.active:raise ValueError('稳定性测试中；不以主界面缓存冒充测试输入')
            packet=window.last_packet
            if not window.running or packet is None or time.monotonic()-packet.get('received_monotonic',0)>1.5:
                raise ValueError('等待新鲜CCD参数；停采时不保留伪实时预测')
            context=dict(camera=dict(packet['active']),roi=dict(window.actual_roi),
                settings=asdict(window.setting()),optics=asdict(window.optical_config()),
                orientation=window.orientation.currentData(),geometry=window.registration,
                dmd_side=getattr(window,'_dmd_side',600))
            dmd=np.asarray(window.desired_dmd);mask=np.asarray(window.slm_file)
            core.validate_request(dmd,mask,context,manifest)
            return dmd,mask,context,core.request_key(dmd,mask,context)

        def invalidate(self,message):
            self.current_key=None;self.result=None;view.setText(message);status.setText(message+'；不会停止CCD或投图。')

        def refresh(self):
            if not enabled.isChecked() or window.tabs.currentWidget() is not page:return
            if self.error:self.invalidate(self.error+'；关闭再开启本复选框可重试');return
            if self.busy and time.monotonic()-self.busy['started']>40:
                self.error='预测进程超时';self.process.kill();self.invalidate(self.error);return
            try:dmd,mask,context,key=self.snapshot()
            except Exception as error:self.invalidate(str(error));return
            if self.current_key!=key:
                self.current_key=key;self.result=None;view.setText('后台计算当前请求图案…')
            if self.busy:return
            if self.result is not None:self.paint();return
            self.serial+=1
            archive=Path(self.folder.name)/'request.npz'
            # One serialized request, no camera frame sent to this process.
            np.savez(archive,dmd=dmd,mask=mask)
            self.request=dict(token=self.serial,archive=str(archive),context=context)
            self.busy=dict(token=self.serial,key=key,started=time.monotonic())
            if self.process.state()==QtCore.QProcess.NotRunning:
                self.process.start('/home/user/miniconda3/bin/python',
                    ['-B',str(core.ROOT/'model.py'),'--serve',str(manifest_path)])
            else:self.send()
            status.setText('独立进程后台预测；CCD采集/界面不等待推理。\n模型配置匹配；此图不用于控制。')

        def send(self):
            if self.request is not None:self.process.write((json.dumps(self.request,allow_nan=False)+'\n').encode())

        def received(self):
            self.buffer+=bytes(self.process.readAllStandardOutput())
            while b'\n' in self.buffer:
                line,self.buffer=self.buffer.split(b'\n',1)
                if not line.startswith(b'E106 '):continue
                try:self.accept(json.loads(line[5:]))
                except Exception as error:self.error='预测返回无效：'+str(error);self.busy=None;self.invalidate(self.error)
            if len(self.buffer)>32768:self.buffer=b'';self.error='预测输出异常';self.process.kill()

        def accept(self,message):
            pending=self.busy
            if pending is None or message.get('token')!=pending['token']:return
            self.busy=None
            try:current=self.snapshot()[3]
            except Exception:return
            if current!=pending['key'] or current!=self.current_key:return # Discard stale result.
            if not message.get('ok'):self.error=message.get('error','预测失败');self.invalidate(self.error);return
            if message['metadata']['key']!=current or message['metadata']['model_state_sha256']!=core.STATE_SHA:
                raise ValueError('预测模型/输入标识不一致')
            target=Path(message['archive']).resolve()
            if target!=Path(self.folder.name)/'result.npz':raise ValueError('unexpected response file')
            with np.load(target,allow_pickle=False) as a:images={k:a[k].copy() for k in ('corrected','physical')}
            roi=manifest['expected']['roi'];shape=(roi['Height'],roi['Width'])
            if any(a.shape!=shape or not np.isfinite(a).all() or a.min()<0 or a.max()>255 for a in images.values()):
                raise ValueError('invalid native CCD prediction')
            self.result=images;self.metadata=message['metadata'];self.paint()

        def paint(self,*args):
            if self.result is None or not enabled.isChecked():return
            try:
                if self.snapshot()[3]!=self.current_key:self.invalidate('输入已改变，等待新预测');return
                raw=window.last_packet['image']
                params=effective_parameters(window,window.manual_display.controls)
                result=map_display_pair(raw,self.result[choice.currentData()],**params)['filtered']
                view.set_image(result,(result.shape[1],result.shape[0]))
                live='当前请求图；尚未投到器件' if not window.can_project() else '当前手动请求图；非逐帧同步'
                warning='未见DMD输入，仅作外推预览' if self.metadata['unseen_input'] else '已覆盖的DMD输入'
                status.setText(f"{live} · {warning}\nE105/{self.metadata['mask']} · 后台{self.metadata['seconds']:.2f}s；同主界面显示映射。\n"
                    '仅均值预测；原差值对应旧ASM；不改积分/控制，无部署资格。')
            except Exception as error:self.invalidate('显示未采用：'+str(error))

        def drain_error(self):self.stderr=(self.stderr+bytes(self.process.readAllStandardError()).decode(errors='replace'))[-4000:]
        def process_error(self,*args):
            if not self.stopping:self.error='预测子进程启动/运行异常：'+self.process.errorString()
        def finished(self,*args):
            self.busy=None
            if not self.stopping and enabled.isChecked():self.error='预测子进程已退出；原CCD采集不受影响'
        def toggle(self,value):
            if not value:
                self.stopping=True;self.process.kill();self.process.waitForFinished(1000)
                self.busy=None;self.invalidate('误差模型预览已关闭');self.stopping=False
            else:self.error=None;self.buffer=b'';self.result=None;self.refresh()
        def close(self):
            self.timer.stop();self.stopping=True;self.process.kill();self.process.waitForFinished(1000)
            self.folder.cleanup()

    controller=Preview();window.surrogate_preview=controller
    return controller
