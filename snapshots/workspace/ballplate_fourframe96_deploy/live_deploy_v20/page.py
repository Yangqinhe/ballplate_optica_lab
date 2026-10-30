"""Static deployment tab; passive views and signals, never device ownership."""
from pathlib import Path
import time
import cv2
import numpy as np
from PyQt5 import QtCore,QtGui,QtWidgets
from stable_views import SensorView,StatusLabel
from .protocol import PreparedBundle


def detector_overlay(image,weights):
    """Display-only annotations; source raw is never modified."""
    out=cv2.cvtColor(np.asarray(image,np.uint8),cv2.COLOR_GRAY2RGB)
    for k,weight in enumerate(weights):
        binary=(weight>=.5).astype(np.uint8)
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out,contours,-1,(0,230,255),1)
        y,x=np.nonzero(binary)
        if len(x):cv2.putText(out,str(k),(int(x.mean()),int(y.mean())),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,210,0),1,cv2.LINE_AA)
    return out


def signal_display(image,ceiling):
    return np.clip(np.asarray(image)*255/max(float(ceiling),1e-6),0,255).astype(np.uint8)


class DeploymentPage(QtWidgets.QWidget):
    freeze_requested=QtCore.pyqtSignal()
    run_requested=QtCore.pyqtSignal(object)
    cancel_requested=QtCore.pyqtSignal()

    def __init__(self,parent=None):
        super().__init__(parent);self.bundle=None;self.frozen=False;self.busy=False
        self.last_view=0;self.current_case=None;self.last_folder=None
        layout=QtWidgets.QVBoxLayout(self)
        title=QtWidgets.QLabel('离线部署测试 · 静态回放 → 固定八区 → 模型对照（不闭环）')
        title.setStyleSheet('font-size:19px;font-weight:bold');layout.addWidget(title)
        self.status=StatusLabel('先加载已准备的研究测试包，再冻结当前ROI/倍率/曝光/SLM位置。不会自动采集或投图。',2)
        layout.addWidget(self.status)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        load=QtWidgets.QPushButton('加载测试包');load.clicked.connect(self.choose_bundle);row.addWidget(load);self.load_button=load
        self.catalog_label=QtWidgets.QLineEdit();self.catalog_label.setReadOnly(True);row.addWidget(self.catalog_label,1)
        self.freeze_button=QtWidgets.QPushButton('① 冻结当前配置');self.freeze_button.clicked.connect(self.freeze_requested);row.addWidget(self.freeze_button)
        self.cases=QtWidgets.QComboBox();self.cases.currentIndexChanged.connect(self.selection_changed);row.addWidget(self.cases)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.confirm=QtWidgets.QCheckBox('确认ND／低功率与光束防护未变，允许本次有界静态投图')
        self.confirm.toggled.connect(self.update_buttons);row.addWidget(self.confirm,1)
        self.run_one=QtWidgets.QPushButton('② 运行所选样本');self.run_one.clicked.connect(lambda:self.run_requested.emit([self.cases.currentData()]));row.addWidget(self.run_one)
        self.run_all=QtWidgets.QPushButton('运行全部6样本');self.run_all.clicked.connect(lambda:self.run_requested.emit(list(self.bundle.cases)));row.addWidget(self.run_all)
        self.stop=QtWidgets.QPushButton('取消测试／两路全0');self.stop.setStyleSheet('color:#b02020;font-weight:bold')
        self.stop.clicked.connect(self.cancel_requested);row.addWidget(self.stop)
        self.progress=QtWidgets.QProgressBar();self.progress.setRange(0,1);self.progress.setValue(0);layout.addWidget(self.progress)
        images=QtWidgets.QHBoxLayout();layout.addLayout(images,1)
        self.dmd=SensorView('四幅RAW96 → 2×2 → 600²DMD')
        self.slm=SensorView('原始灰度mask · 不另套LUT')
        self.ccd=SensorView('共用当前CCD · 原始ROI方向 · 固定八区')
        self.prediction=SensorView('当前配置的理想4f研究预测，尚未标定幅值')
        for caption,view in (('DMD请求预览（非自动加载）',self.dmd),('SLM规范patch（非硬件镜像）',self.slm),
            ('实际CCD＋固定八区（标线仅显示）',self.ccd),('研究预测（非已放行模型）',self.prediction)):
            box=QtWidgets.QGroupBox(caption);body=QtWidgets.QVBoxLayout(box);body.addWidget(view);images.addWidget(box)
        self.readouts=QtWidgets.QTableWidget(8,6)
        self.readouts.setHorizontalHeaderLabels(['区域','净积分ADU','单帧不确定度','SNR','模型积分ADU','差值ADU'])
        self.readouts.verticalHeader().hide();self.readouts.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.readouts.verticalHeader().setDefaultSectionSize(24)
        self.readouts.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.readouts.setFixedHeight(245);layout.addWidget(self.readouts)
        self.result_text=StatusLabel('尚未运行。八区来自固定参考映射，仅可见ROI；不补ROI外能量，不直接送旧线性头。',3)
        layout.addWidget(self.result_text)
        foot=QtWidgets.QHBoxLayout();layout.addLayout(foot)
        self.path_button=QtWidgets.QPushButton('打开本轮结果目录');self.path_button.clicked.connect(self.open_results);foot.addWidget(self.path_button)
        note=QtWidgets.QLabel('下一阶段：适配18参数线性头 → 有界MuJoCo闭环；当前均未开放。CCD不累积为策略历史。')
        note.setWordWrap(True);foot.addWidget(note,1)
        self.update_buttons()

    def choose_bundle(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'加载已冻结的静态研究测试包','','catalog.json (catalog.json)')
        if path:
            try:self.load_bundle(path)
            except Exception as error:self.status.setText('测试包未采用：'+str(error))

    def load_bundle(self,path):
        if self.busy:raise RuntimeError('测试中不能更换资产')
        bundle=PreparedBundle(path);self.bundle=bundle;self.frozen=False
        self.catalog_label.setText(str(bundle.path));self.cases.clear()
        for i in (8,0,16,9,1,17):
            row=bundle.cases[i];f=row['fixture']
            self.cases.addItem(f"{i:02d} · {f['kind']} · {f['xy']}",i)
        self.status.setText('研究包已校验；旧样本回放，不是新未见策略验收。先冻结本次配置。')
        self.selection_changed();self.update_buttons()

    def selection_changed(self):
        if not self.bundle or self.cases.currentData() is None:return
        self.show_case(self.cases.currentData())

    def show_case(self,index):
        row=self.bundle.cases[index];self.current_case=index
        self.dmd.set_image(row['dmd_array']);self.slm.set_image(self.bundle.phase)
        pred=row['predictions']['learned'];ceiling=max(float(pred.max()),1.)
        self.prediction.set_image(signal_display(pred,ceiling),pred.shape[::-1])

    def update_buttons(self,*args):
        exists=self.bundle is not None
        self.load_button.setEnabled(not self.busy);self.freeze_button.setEnabled(exists and not self.busy)
        self.cases.setEnabled(exists and not self.busy);self.confirm.setEnabled(not self.busy)
        enabled=exists and self.frozen and self.confirm.isChecked() and not self.busy
        self.run_one.setEnabled(enabled);self.run_all.setEnabled(enabled);self.stop.setEnabled(self.busy)
        self.path_button.setEnabled(self.last_folder is not None)

    def set_busy(self,value,total_cases=1):
        self.busy=value
        if value:
            self.progress.setRange(0,total_cases*7*8);self.progress.setValue(0)
        else:self.confirm.setChecked(False)
        self.update_buttons()

    def fresh_view(self,packet):
        if not self.bundle or time.monotonic()-self.last_view<.2:return
        self.last_view=time.monotonic()
        if packet['image'].shape!=self.bundle.weights.shape[1:]:return
        self.ccd.set_image(detector_overlay(packet['image'],self.bundle.weights),packet['image'].shape[::-1])

    def on_progress(self,data):
        index=data['case_id'];kind=data['kind']
        if self.current_case!=index:self.show_case(index)
        if kind=='stage_start':self.status.setText(f"输入{index:02d} · {data['stage']} · 原连接有界采集；不是硬件图案同步。")
        if kind=='sample_saved':self.progress.setValue(self.progress.value()+1)
        if kind=='stage_complete':self.show_stats(data['stats'],index,data.get('gain'),data['stage'].startswith('learned'))
        if kind=='case_complete':
            a=data['analysis'];m=a['measured_and_prediction']
            self.show_stats(a['readouts'],index,data.get('gain'))
            ncc='无有效值' if m['image_NCC'] is None else f"{m['image_NCC']:.3f}"
            self.result_text.setText(f"输入{index:02d}：NCC={ncc}，八区L1={m['native8_relative_L1']*100:.2f}%；"
                f"稳定性{'通过' if a['stability_passed'] else '未通过'}，静态门槛{'通过' if a['static_criteria_passed'] else '未通过'}。不放行控制。")
            ceiling=max(float(data['predicted_signal'].max()),float(data['observed_signal'].max()),1.)
            self.prediction.set_image(signal_display(data['predicted_signal'],ceiling),data['predicted_signal'].shape[::-1])

    def show_stats(self,stats,index,gain,learned=True):
        expected=None
        if gain is not None:
            from .protocol import sums
            expected=sums(self.bundle.cases[index]['predictions']['learned' if learned else 'zero']*gain,self.bundle.weights)
        for k in range(8):
            mean=stats['mean_signed_integrals_ADU'][k]
            values=[str(k),f'{mean:.2f}',f"{stats['single_frame_SD_plus_background_uncertainty_ADU'][k]:.2f}",
                f"{stats['SNR'][k]:.2f}",'-' if expected is None else f'{expected[k]:.2f}',
                '-' if expected is None else f'{mean-expected[k]:.2f}']
            for column,value in enumerate(values):self.readouts.setItem(k,column,QtWidgets.QTableWidgetItem(value))

    def finished(self,report):
        self.set_busy(False);self.last_folder=report['folder'];self.update_buttons()
        if report['error']:self.status.setText('测试未通过／已停止：'+report['error']+'；原图保留，两路全0。')
        else:
            self.status.setText(('静态门槛通过' if report.get('static_criteria_passed') else '静态门槛未通过')+
                '；两路全0、CCD连接保持。不是策略/闭环放行。')
        self.catalog_label.setToolTip('结果：'+report['folder'])

    def open_results(self):
        if self.last_folder:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.last_folder))
