"""Bounded real static optical diagnostic, NOT trained-policy deployment."""
import argparse
from datetime import datetime
import hashlib
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from PIL import Image,ImageGrab
from PyQt5 import QtCore,QtGui,QtWidgets
from display_linux import read_roles,same_signals
from spin_c import Camera,LIBRARY,IncompleteFrame

ARCHIVE=Path(__file__).resolve().parents[2]/'ballplate_minframes_optical_ppo/SDON_Linux_Handoff_20261015'
MODULATOR_SOURCE=ARCHIVE/'src/sdon/hardware/modulator.py'
spec=importlib.util.spec_from_file_location('readonly_archive_modulator',MODULATOR_SOURCE)
modulator=importlib.util.module_from_spec(spec);spec.loader.exec_module(modulator)


def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def save_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def dmd_pattern(kind):
    image=np.zeros((600,600),np.uint8)
    if kind=='black':return image
    positions={'centre':(300,300),'left':(172,300),'right':(428,300)}
    if kind not in positions:raise ValueError('unknown sparse binary diagnostic')
    x,y=positions[kind];image[y-32:y+32,x-32:x+32]=255
    # Small asymmetric fiducial, total white area <1.3% of active canvas.
    image[y-52:y-36,x+36:x+52]=255
    return image


def slm_pattern(kind):
    if kind=='zero':return np.zeros((1080,1920),np.uint8)
    if kind not in ('+x','-x','+y','-y'):raise ValueError('unknown raw grating')
    y,x=np.indices((1080,1920));axis=x if kind[1]=='x' else y
    axis=(axis//4)*4
    # Raw grays, not calibrated phase. Whole-panel coverage avoids inheriting
    # an unverified historical 600px offset. Horizontal mirror applied once.
    return ((((axis if kind[0]=='+' else -axis)%128)//4)*255/31).round().astype(np.uint8)


def metrics(frames):
    stack=np.stack(frames).astype(np.float32);mean=stack.mean(0)
    noise=float(np.sqrt(np.mean((stack-mean)**2)))
    return mean,dict(mean_ADU=float(mean.mean()),max_ADU=int(stack.max()),p999_ADU=float(np.quantile(stack,.999)),
        saturated_fraction=float(np.mean(stack==255)),within_repeat_rms_ADU=noise)


def compare(a,b):
    difference=a['mean']-b['mean'];rmse=float(np.sqrt(np.mean(difference**2)))
    noise=max(.05,float(np.hypot(a['metrics']['within_repeat_rms_ADU'],b['metrics']['within_repeat_rms_ADU'])))
    return dict(rmse_ADU=rmse,repeat_noise_rms_ADU=noise,ratio_to_repeat_noise=rmse/noise,
                observable_difference=bool(rmse>max(.1,3*noise)))


class Bench(QtWidgets.QMainWindow):
    def __init__(self,output,automatic=False,low_power_confirmed=True,diagnostic_scope='raw_static'):
        super().__init__();self.output=output;output.mkdir(parents=True,exist_ok=False)
        self.output_lock=open('/run/user/1000/ballplate_optical_live_display.lock','a')
        try:fcntl.flock(self.output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another live diagnostic owns the output windows')
        self.baseline=read_roles();self.camera=None;self.busy=False;self.cancelled=False;self.closed_safely=False
        self.results={};self.windows={};self.last_arrays={};self.low_power_confirmed=low_power_confirmed;self.rejected_frames=[];self.valid_frame_count=0
        screens={screen.name():screen for screen in QtWidgets.QApplication.screens()}
        for role,row in self.baseline.items():
            screen=screens.get(row['connector'])
            if screen is None or screen.devicePixelRatio()!=1:raise RuntimeError('exact physical Qt screen and 100% scale required')
            g=screen.geometry()
            if [g.x(),g.y()]!=row['desktop_position'] or [g.width(),g.height()]!=row['desktop_size']:
                raise RuntimeError('Qt/XRandR geometry mismatch')
        primary=QtWidgets.QApplication.primaryScreen()
        if primary.name() in {r['connector'] for r in self.baseline.values()}:raise RuntimeError('optical output is primary monitor')
        p=primary.geometry();self.setGeometry(p.x()+40,p.y()+60,1200,900)
        self.setWindowTitle('真实光路静态诊断 · DMD → SLM → CCD · 非控制部署')
        root=QtWidgets.QWidget();self.setCentralWidget(root);layout=QtWidgets.QVBoxLayout(root)
        title=QtWidgets.QLabel('真实器件 · 稀疏二值 DMD → raw SLM 光栅 → 原始 CCD（不运行策略）')
        title.setStyleSheet('font-size:19px;font-weight:bold');layout.addWidget(title)
        self.warning=QtWidgets.QLabel('低功率已由用户确认；DMD控制器/硬触发未验收；历史LUT、ROI和配准不自动启用。')
        self.warning.setWordWrap(True);self.warning.setStyleSheet('color:#cc6600');layout.addWidget(self.warning)
        grid=QtWidgets.QGridLayout();layout.addLayout(grid)
        self.panels={}
        for i,(name,caption) in enumerate((('dmd','DMD600稀疏图（显示预览）'),('slm','SLM1920×1080 raw灰度（非训练mask）'),('ccd','真实CCD全幅Mono8'),('difference','与黑屏/零指令的差异（仅显示增强）'))):
            box=QtWidgets.QGroupBox(caption);body=QtWidgets.QVBoxLayout(box)
            label=QtWidgets.QLabel('等待测量');label.setAlignment(QtCore.Qt.AlignCenter);label.setMinimumSize(450,230)
            label.setStyleSheet('background:#111;color:white');body.addWidget(label);self.panels[name]=label
            grid.addWidget(box,i//2,i%2)
        self.state=QtWidgets.QLabel('准备单进程设备检查；输出为黑屏待机');self.state.setWordWrap(True);layout.addWidget(self.state)
        self.log=QtWidgets.QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(130);layout.addWidget(self.log)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.run_button=QtWidgets.QPushButton('运行一轮有界静态检查');self.run_button.clicked.connect(self.run_test);row.addWidget(self.run_button)
        stop=QtWidgets.QPushButton('停止测试并切DMD黑屏');stop.clicked.connect(self.cancel);row.addWidget(stop)
        quit_button=QtWidgets.QPushButton('关闭光源后退出');quit_button.clicked.connect(self.close);row.addWidget(quit_button)
        self.show()
        # Reuse the original output widget; diagnostic scope is explicit and
        # separate from HardwareManager's unpassed deployment/calibration gates.
        for role,row in self.baseline.items():
            size=(600,600) if role=='dmd' else tuple(row['desktop_size'])
            window=modulator.OpticalModulator(screens[row['connector']],size,role.upper(),(0,0))
            window.setCursor(QtCore.Qt.BlankCursor);self.windows[role]=window
            window.update_image(dmd_pattern('black') if role=='dmd' else slm_pattern('zero'));window.present()
        self.record(dict(state='created',video=self.baseline,diagnostic_only=True,
            low_power_confirmed_by_user=low_power_confirmed,controller_status='USB unavailable / not qualified',
            trigger_qualification_passed=False,current_calibration_verified=False,release_approved=False,
            archival_modulator_sha256=hashlib.sha256(MODULATOR_SOURCE.read_bytes()).hexdigest(),
            C_API_library=str(LIBRARY),diagnostic_scope=diagnostic_scope,
            SLM_full_panel_raw_test_NOT_network600=(diagnostic_scope=='raw_static'),old_offsets_loaded=False))
        self.request_timer=QtCore.QTimer(self);self.request_timer.timeout.connect(self.poll_request);self.request_timer.start(300)
        if automatic:QtCore.QTimer.singleShot(800,self.run_test)

    def poll_request(self):
        request=self.output/'RUN_STATIC_CONFIRMED.json'
        if self.busy or self.results or not request.exists():return
        payload=json.loads(request.read_text())
        if payload.get('command')!='run_static' or payload.get('low_power_restored_by_user') is not True:
            self.note('拒绝未确认照明的测试请求');return
        request.rename(self.output/'RUN_STATIC_CONFIRMED.consumed.json')
        self.record(dict(state='confirmed_request',request=payload));self.run_test()

    def record(self,row):
        with (self.output/'events.jsonl').open('a') as f:f.write(json.dumps(dict(unix_ns=time.time_ns(),**row),ensure_ascii=False)+'\n')

    def note(self,text):
        self.state.setText(text);self.log.appendPlainText(text);print(text,flush=True)
        QtWidgets.QApplication.processEvents()

    def pump(self,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            QtWidgets.QApplication.processEvents();time.sleep(.01)
            if self.cancelled:raise RuntimeError('user cancelled static test')

    def paint_panel(self,name,image):
        array=np.ascontiguousarray(np.clip(image,0,255),dtype=np.uint8);h,w=array.shape
        q=QtGui.QImage(array.data,w,h,w,QtGui.QImage.Format_Grayscale8).copy()
        self.panels[name].setPixmap(QtGui.QPixmap.fromImage(q).scaled(550,270,QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation))

    def blank(self):
        if 'dmd' in self.windows:
            self.windows['dmd'].update_image(dmd_pattern('black'))
            QtWidgets.QApplication.processEvents();QtGui.QGuiApplication.sync()

    def cancel(self):
        self.cancelled=True;self.blank();self.state.setText('已请求停止并将 DMD 置黑；退出前仍须关闭/遮挡光源。')

    def apply(self,dmd,slm,tag):
        same_signals(self.baseline,read_roles())
        self.windows['dmd'].update_image(dmd)
        actual_slm=np.ascontiguousarray(np.fliplr(slm))
        self.windows['slm'].update_image(actual_slm)
        QtWidgets.QApplication.processEvents();QtGui.QGuiApplication.sync();self.pump(.7)
        states={role:window.display_status() for role,window in self.windows.items()}
        for role,state in states.items():
            if not state['hash_matches'] or not self.windows[role].wait_until_applied(.2):raise RuntimeError(role+': output did not apply')
            g=self.windows[role].geometry();row=self.baseline[role]
            if [g.x(),g.y()]!=row['desktop_position'] or [g.width(),g.height()]!=row['desktop_size']:
                raise RuntimeError(role+': actual full-screen window is not on its verified physical output')
            x,y=row['desktop_position'];w,h=row['desktop_size']
            pixels=np.asarray(ImageGrab.grab(bbox=(x,y,x+w,y+h))).copy()
            expected=np.zeros((h,w),np.uint8)
            if role=='dmd':expected[:,100:700]=dmd
            else:expected=actual_slm
            error=int(np.abs(pixels[:,:,:3].astype(np.int16)-expected[:,:,None].astype(np.int16)).max())
            state['framebuffer_max_error']=error
            if error:raise RuntimeError(f'{role}: framebuffer pixels differ from requested test: {error}')
        self.paint_panel('dmd',dmd);self.paint_panel('slm',slm)
        return dict(tag=tag,display=states,SLM_raw=True,SLM_LUT_applied=False,SLM_horizontal_flip_count=1,
            DMD_sha256=digest(dmd),SLM_canonical_sha256=digest(slm),
            physical_response_or_optical_settling_not_implied_by_framebuffer=True)

    def measure(self,tag,dk,sk):
        self.note(f'测量 {tag}：DMD={dk} / SLM={sk}')
        dmd=dmd_pattern(dk);slm=slm_pattern(sk);applied=self.apply(dmd,slm,tag)
        directory=self.output/tag;directory.mkdir()
        Image.fromarray(dmd).save(directory/'DMD600_binary.png');Image.fromarray(slm).save(directory/'SLM_raw_canonical.png')
        discarded=[]
        for _ in range(2):
            _,meta=self.valid_capture();discarded.append(meta['FrameID']);QtWidgets.QApplication.processEvents()
        frames=[];records=[]
        for i in range(3):
            if self.cancelled:raise RuntimeError('user cancelled static test')
            array,meta=self.valid_capture();frames.append(array);records.append(meta)
            np.save(directory/f'CCD_raw_{meta["FrameID"]:08d}.npy',array,allow_pickle=False)
            self.paint_panel('ccd',array);QtWidgets.QApplication.processEvents()
        mean,statistics=metrics(frames);Image.fromarray(mean.round().astype(np.uint8)).save(directory/'CCD_mean_raw.png')
        row=dict(**applied,discarded_frame_ids=discarded,frames=records,metrics=statistics,
                 camera_parameters=self.camera.active,registered_CCD=False,hardware_trigger_test=False)
        save_json(directory/'measurement.json',row);self.results[tag]=dict(mean=mean,metrics=statistics)
        if 'black_start' in self.results:
            difference=np.abs(mean-self.results['black_start']['mean'])
            self.paint_panel('difference',difference*16)
        self.note(f'{tag}：max={statistics["max_ADU"]}，饱和={statistics["saturated_fraction"]:.3%}，帧ID={[r["FrameID"] for r in records]}')
        if statistics['saturated_fraction']>.001:raise RuntimeError('saturation >0.1%; stop, attenuate illumination before continuing')

    def valid_capture(self):
        # Bound retries and preserve every rejected record. No zero filling,
        # previous-frame substitution, or treating an incomplete image as data.
        for attempt in range(4):
            try:
                frame=self.camera.capture();self.valid_frame_count+=1;return frame
            except IncompleteFrame as error:
                self.rejected_frames.append(error.record);self.record(dict(state='rejected_incomplete_frame',**error.record))
                self.note('已丢弃坏帧：'+str(error.record))
        raise RuntimeError('4 consecutive incomplete frames; stop diagnostic and repair GigE transport')

    def run_test(self):
        if self.busy:return
        if self.results:
            self.note('本轮已保存；需要新会话再测，不能覆盖原结果。');return
        self.busy=True;self.cancelled=False;self.run_button.setEnabled(False)
        report=dict(role='real_static_RAW_diagnostic_NOT_policy_deployment',hardware_enabled=True,
            release_approved=False,current_calibration_verified=False,hardware_trigger_accepted=False,
            control_20ms_accepted=False,video=self.baseline,error=None)
        started=time.monotonic()
        try:
            self.note('打开唯一相机15328133，暂时固定曝光/增益；不改ROI或触发。')
            self.camera=Camera();report['camera_before']=self.camera.before
            self.camera.start();report['camera_active']=self.camera.active
            save_json(self.output/'camera_session.json',report)
            steps=[('black_start','black','zero'),('centre_zero','centre','zero'),
                   ('left_zero','left','zero'),('right_zero','right','zero'),
                   ('centre_plus_x','centre','+x'),('centre_minus_x','centre','-x'),
                   ('centre_plus_y','centre','+y'),('centre_minus_y','centre','-y'),
                   ('centre_return','centre','zero'),('black_end','black','zero')]
            for tag,dk,sk in steps:self.measure(tag,dk,sk)
            pairs={'DMD_black_vs_centre':('black_start','centre_zero'),
                   'DMD_left_vs_right':('left_zero','right_zero'),
                   'SLM_plus_vs_minus_x':('centre_plus_x','centre_minus_x'),
                   'SLM_plus_vs_minus_y':('centre_plus_y','centre_minus_y'),
                   'zero_return_drift':('centre_zero','centre_return'),
                   'black_return_drift':('black_start','black_end')}
            comparisons={name:compare(self.results[a],self.results[b]) for name,(a,b) in pairs.items()}
            report['comparisons']=comparisons
            report['DMD_response_observed']=all(comparisons[n]['observable_difference'] for n in ('DMD_black_vs_centre','DMD_left_vs_right'))
            report['SLM_response_observed']=all(comparisons[n]['observable_difference'] for n in ('SLM_plus_vs_minus_x','SLM_plus_vs_minus_y'))
            report['static_sequence_complete']=True
            report['threshold_is_diagnostic_NOT_independent_optical_validation']=True
            self.note('静态序列完成；仅记录观测差异，不等于标定或训练策略部署通过。')
        except Exception as error:
            report['error']=str(error);report['traceback']=traceback.format_exc();self.note('停止：'+str(error))
        finally:
            self.blank()
            self.windows['slm'].update_image(slm_pattern('zero'));QtWidgets.QApplication.processEvents()
            if self.camera:
                report['stream_statistics']=self.camera.stream_statistics()
                try:self.camera.close();report['camera_settings_restored']=True;report['camera_after']=self.camera.after
                except Exception as error:report['camera_restore_error']=str(error);self.note('相机恢复异常：'+str(error))
                self.camera=None
            report['wall_seconds']=time.monotonic()-started
            report['valid_frames_including_discarded_settle_frames']=self.valid_frame_count
            report['rejected_incomplete_frames']=self.rejected_frames
            report['completed_measurements']=list(self.results);report['optical_outputs_left_black_standby']=True
            save_json(self.output/'report.json',report);self.record(dict(state='finished',report=report))
            self.busy=False
            self.note(f'结果已保存：{self.output}；DMD黑屏待机，不会输出控制量。')

    def closeEvent(self,event):
        self.blank()
        if self.busy:
            self.cancelled=True;event.ignore();return
        reply=QtWidgets.QMessageBox.question(self,'退出前确认光源安全',
            '退出会移除器件上的黑色窗口并恢复桌面。请先关闭或物理遮挡光源。\n已关闭/遮挡光源，确认退出？',
            QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)
        if reply!=QtWidgets.QMessageBox.Yes:event.ignore();return
        for window in self.windows.values():window.close()
        self.output_lock.close()
        self.closed_safely=True;event.accept()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--run-static',action='store_true')
    parser.add_argument('--low-power-confirmed',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.preflight:print(json.dumps(read_roles(),indent=2));return
    if not args.low_power_confirmed:raise RuntimeError('explicit physical low-power/ND confirmation required')
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling,True)
    app=QtWidgets.QApplication(['ballplate-real-static-diagnostic'])
    if app.platformName()!='xcb':raise RuntimeError('real local X11 required; no offscreen hardware output')
    output=args.output or Path(__file__).parents[1]/'outputs/live_static'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    window=Bench(output,args.run_static);app.exec_()


if __name__=='__main__':main()
