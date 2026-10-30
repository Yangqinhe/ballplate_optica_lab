"""Fresh offscreen whole-window QA, reusing archived data (no hardware)."""
from dataclasses import asdict
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
import run
from run import c,ROOT,PRIOR,COMPUTE
from PyQt5 import QtCore,QtWidgets
import page


def main():
    if os.environ.get('QT_QPA_PLATFORM')!='offscreen':raise ValueError('offscreen only')
    out=ROOT/'WINDOW_QA_01';out.mkdir()
    page.PREFERENCES_PATH=out/'offline_ui_preferences.ini'
    cfg=c.read(PRIOR/'CONFIG.json')
    for path,sha in cfg['source_sha256'].items():
        if c.digest(path)!=sha:raise ValueError('prior source changed: '+path)
    folder=ROOT.parent/'E108_paired_display_fix_20261030_1/PAIRED_RUNS/RUN_1793353034436016112/E001_T000000_Q000003'
    record=c.read(folder/'STEP.json')
    with np.load(folder/'observation.npz') as a:images={k:a[k].copy() for k in a.files}
    with np.load(c.checked(record['real_capture'])) as a:measured=a['image'].copy()
    def hook(w,p,preview):
        preview.timer.stop();w.timer.stop();p.timer.stop()
        def check():
            try:
                assert w.tabs.count()==3 and not w.windows and w.worker is None and not w.armed.isChecked()
                assert asdict(w.setting())==cfg['retained_manual_controls']['settings']
                assert w.actual_roi==c.read(c.checked(cfg['restart']))['roi']
                assert w.pending_registration==c.read(c.checked(cfg['restart']))['registration']
                for name,item in c.read(c.checked(cfg['caches']))['files'].items():
                    np.testing.assert_array_equal(getattr(w,name+'_file'),np.load(c.checked(item)))
                filtering=w.manual_display.controls.filter_parameters()
                for key,value in cfg['retained_manual_controls']['filter'].items():assert filtering[key]==value
                assert w.region_overlay.count==64 and w._dmd_side==round(cfg['retained_dmd_scale']*600)
                assert p.radius.value()==cfg['paired_ui']['radius_cm']
                assert p.impulse.value()==cfg['paired_ui']['impulse_Ns']
                assert p.angle.value()==cfg['paired_ui']['angle_deg']
                assert (p.settle_ms.minimum(),p.settle_ms.maximum(),p.settle_ms.value())==(10,1000,1000)
                for value in (10,200,1000):
                    p.settle_ms.setValue(value)
                    assert p.settle_ms.value()==value and not p.go and p.process.state()==0
                p.images=images;p.pending_ccd=measured;p.commit_view(record)
                p.show_states(record['states_after']);p.hold_view('离线演示：等待下一帧')
                w.resize(1820,1080);w.tabs.setCurrentIndex(p.index)
                p.info.setText('离线检查：10–1000ms可调，默认1000ms，下一输入生效。图像是E108历史回放，不是当前实测。')
                QtWidgets.QApplication.processEvents();w.grab().save(str(out/'paired_page.png'))
                assert p.settle_ms.isVisible() and p.page.rect().contains(p.settle_ms.mapTo(p.page,QtCore.QPoint(0,0)))
                c.write(out/'QA.json',dict(passed=True,hardware_actions=0,old_configuration_retained=True,
                    range_ms=[10,1000],default_ms=1000,shared_filter=filtering,
                    pages=[w.tabs.tabText(i) for i in range(3)],fixture=c.ref(folder/'STEP.json'),
                    page=c.ref(ROOT/'page.py'),new_source=c.ref(ROOT/'run.py')))
            except Exception as e:
                c.write(out/'QA_FAILED.json',dict(error=repr(e)));raise
        QtCore.QTimer.singleShot(350,check)
    run.run_gui(cfg,out,False,qa_hook=hook)
    assert c.read(out/'QA.json')['passed']
    r=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(COMPUTE),'-p','test_pair.py'],
        capture_output=True,text=True,timeout=40)
    c.write(ROOT/'CORE_RESULT.json',dict(passed=r.returncode==0,stdout=r.stdout,stderr=r.stderr,hardware_actions=0))
    if r.returncode:raise RuntimeError(r.stderr)
    print('Whole window and frozen core regression PASS; hardware actions=0')


if __name__=='__main__':main()
