"""Offline numerical, process and whole-window checks with explicit fixtures."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import model
from test_preview import fixture


def inference(output):
    output.mkdir();m,c,_,_=fixture();engine=model.Engine(m);checks=[];sample=None
    for folder,ids in (('FINAL_TIME_FORECAST_01',(0,4)),('FROZEN_TEST_FORECAST_01',(0,32))):
        f=model.read(model.EXPERIMENT/folder/'FORECAST.json')
        with np.load(model.checked(f['predictions']),allow_pickle=False) as frozen:
            for i in ids:
                row=f['rows'][i];d=np.load(model.checked(row['dmd_reference']));s=np.load(model.checked(row['mask_reference']))
                images,meta=engine.predict(d,s,c)
                errors={k:float(np.max(np.abs(images[k]-frozen[fk][i])))
                        for k,fk in (('corrected','predicted'),('physical','physics'))}
                rms={k:float(np.sqrt(np.mean((images[k]-frozen[fk][i])**2)))
                     for k,fk in (('corrected','predicted'),('physical','physics'))}
                # Native forecasts are float32, not bitwise reproducible GPU
                # archives. Require <1/100 sensor ADU peak and <1/10000 RMS;
                # this is numerical equivalence, NOT a model-accuracy gate.
                assert max(errors.values())<.01 and max(rms.values())<1e-4,(errors,rms)
                checks.append(dict(forecast=model.ref(model.EXPERIMENT/folder/'FORECAST.json'),index=i,
                    max_error_ADU=errors,rms_error_ADU=rms,metadata=meta))
                if sample is None:sample=(d,s,images,meta)
    idx=model.read(model.EXPERIMENT/'FINAL_TIME_AUDIT_01/INDEX.json')
    rows=[i for i,r in enumerate(idx['rows']) if r['input_id']=='CENTER' and r['mask_name']=='BASE']
    with np.load(model.checked(idx['means']),allow_pickle=False) as a:measured=a['images'][rows].mean(0)
    with (output/'FIXTURE.npz').open('xb') as f:
        np.savez(f,dmd=sample[0],mask=sample[1],measured=measured,**sample[2])
    model.write(output/'RESULT.json',dict(passed=True,checks=checks,fixture_metadata=sample[3],
        fixture=model.ref(output/'FIXTURE.npz'),source=model.ref(__file__),hardware_actions=0,
        comparisons_are_reuse_NOT_new_holdout=True))
    print('Frozen forecast equivalence PASS',checks,flush=True)


def window_check(config,source,output):
    import run
    from PyQt5 import QtCore,QtWidgets
    m,c,_,_=fixture();report=model.read(source/'RESULT.json')
    with np.load(model.checked(report['fixture'])) as a:images={k:a[k].copy() for k in a.files}
    output.mkdir();checks=[]
    def hook(w,controller):
        controller.timer.stop()
        def test():
            try:
                assert not w.hardware and not w.windows and w.worker is None
                # The fixture is an archived mean, not a complete live-worker
                # packet. Stop the unrelated camera timer in this offline QA.
                w.timer.stop()
                d,s=images['dmd'],images['mask'];key=model.request_key(d,s,c)
                controller.snapshot=lambda:(d,s,c,key)
                w.fault=None;w.actual_roi=deepcopy(m['expected']['roi'])
                w.last_packet=dict(image=np.rint(images['measured']).astype(np.uint8),active=c['camera'],
                    meta=dict(FrameID=1,TimeStamp=1),received_monotonic=time.monotonic())
                controller.current_key=key;controller.result={k:images[k] for k in ('corrected','physical')}
                controller.metadata=report['fixture_metadata']
                w.manual_display.controls.mode.setCurrentIndex(w.manual_display.controls.mode.findData('fixed'))
                controls=w.manual_display.controls
                controls.gain.setValue(38.8);controls.black.setValue(10);controls.gamma.setValue(1)
                w.enhance.setChecked(True)
                from optical_lab.display import map_display_pair
                mapped=map_display_pair(w.last_packet['image'],images['measured'],gain=38.8,black=10,gamma=1)
                w.view.set_image(mapped['filtered'],(552,552));w.view.parentWidget().setTitle('历史CCD · 离线界面检查')
                w.sim_view.setText('保留原对齐ASM入口\n离线检查未运行此模型')
                w.diff_view.setText('原对齐差值入口未改变')
                w.tabs.setCurrentIndex(0);w.resize(1840,1220);controller.paint()
                assert controller.view.image is not None and controller.view.extent==(552,552)
                first=controller.view.image.copy();controls.gain.setValue(8);controller.paint()
                assert controller.view.image!=first;checks.append('main display gain changes prediction')
                controls.gain.setValue(38.8);controller.choice.setCurrentIndex(1);physical=controller.view.image.copy()
                controller.choice.setCurrentIndex(0);assert controller.view.image!=physical
                checks.append('calibrated physics / learned mean selector')
                controller.busy=dict(token=1,key='obsolete',started=time.monotonic())
                controller.accept(dict(token=1,ok=True))
                assert controller.busy is None;checks.append('stale input response discarded before loading file')
                before=images['corrected'].copy()
                controller.enabled.setChecked(False);assert controller.view.image is None
                # Do not automatically start a subprocess for this view-only QA.
                controller.refresh=lambda:None;controller.enabled.setChecked(True)
                controller.current_key=key;controller.result={k:images[k] for k in ('corrected','physical')}
                controller.paint();np.testing.assert_array_equal(before,images['corrected'])
                checks.append('disable/re-enable changes only preview')
                QtWidgets.QApplication.processEvents();w.shared_sensor_scale.refresh()
                rects=[v.image_rect for v in (w.view,controller.view)]
                assert rects[0][2:]==rects[1][2:],rects
                checks.append('native CCD extents share display size')
                w.grab().save(str(output/'preview_with_historical_fixture.png'))
                assert not w.windows and w.worker is None and not w.armed.isChecked()
                model.write(output/'PREVIEW_QA.json',dict(passed=True,checks=checks,hardware_actions=0,
                    historical_fixture=model.ref(source/'FIXTURE.npz'),not_a_live_CCD_frame=True,
                    source=model.ref(__file__)))
            except Exception as error:
                model.write(output/'PREVIEW_QA_FAILED.json',dict(error=repr(error)));raise
        QtCore.QTimer.singleShot(200,test)
    run.run_gui(model.read(config),output,False,qa_hook=hook)
    assert model.read(output/'PREVIEW_QA.json')['passed']


def process_check(output):
    """Exercise the real subprocess transport without Qt or hardware."""
    import json,subprocess,sys
    output.mkdir();m,c,d,s=fixture()
    archive=output/'request.npz'
    with archive.open('xb') as f:np.savez(f,dmd=d,mask=s)
    command=[sys.executable,'-B',str(model.ROOT/'model.py'),'--serve',str(model.ROOT/'MODEL_01/MODEL.json')]
    rows=[]
    for token,context in ((1,c),(2,dict(c,dmd_side=1080))):
        rows.append(json.dumps(dict(token=token,archive=str(archive),context=context)))
    start=time.monotonic()
    result=subprocess.run(command,input='\n'.join(rows)+'\n',text=True,capture_output=True,timeout=40)
    messages=[json.loads(x[5:]) for x in result.stdout.splitlines() if x.startswith('E106 ')]
    assert result.returncode==0 and len(messages)==2,(result.returncode,result.stderr,messages)
    assert messages[0]['ok'] and messages[0]['token']==1
    assert not messages[1]['ok'] and messages[1]['token']==2
    model.write(output/'RESULT.json',dict(passed=True,seconds=time.monotonic()-start,messages=messages,
        source=model.ref(__file__),hardware_actions=0,stderr=result.stderr))
    print('Subprocess result / mismatch rejection PASS',flush=True)


def asynchronous_check(output):
    """Real Qt subprocess + heartbeat, in a fake offscreen window; no SDK."""
    import sys
    from types import SimpleNamespace as NS
    from PyQt5 import QtCore,QtWidgets
    import run,panel
    base=run.base_module()
    sys.path.insert(0,str(base.LEGACY/'live_manual_v17'))
    from stable_views import SensorView,SharedSensorScale
    import optical_lab
    base.load('optical_lab.display',base.FILTER/'staged/optical_lab/display.py')
    output.mkdir();m,context,dmd,mask=fixture()
    app=QtWidgets.QApplication([]);w=QtWidgets.QWidget();root=QtWidgets.QVBoxLayout(w)
    w.tabs=QtWidgets.QTabWidget();root.addWidget(w.tabs);page=QtWidgets.QWidget();w.tabs.addTab(page,'offline')
    layout=QtWidgets.QVBoxLayout(page);row=QtWidgets.QHBoxLayout();layout.addLayout(row)
    for name in ('view','sim_view','diff_view'):
        box=QtWidgets.QGroupBox(name);body=QtWidgets.QVBoxLayout(box);v=SensorView();body.addWidget(v)
        row.addWidget(box);setattr(w,name,v)
    w.shared_sensor_scale=SharedSensorScale((w.view,w.sim_view,w.diff_view),w)
    w.last_packet=dict(image=np.zeros((552,552),np.uint8));w.can_project=lambda:False
    w.enhance=QtWidgets.QCheckBox();w.enhance.setChecked(True)
    w.manual_display=NS(controls=NS(mode=NS(currentData=lambda:'fixed'),
        parameters=lambda enabled:dict(mode='fixed',gain=10.,black=10.,gamma=1.)))
    c=panel.attach(w,model.ROOT/'MODEL_01/MODEL.json',model);c.timer.stop()
    c.snapshot=lambda:(dmd,mask,context,model.request_key(dmd,mask,context))
    w.show();ticks=[];start=time.monotonic();timer=QtCore.QTimer()
    def tick():
        ticks.append(time.monotonic())
        if c.result is not None or c.error or ticks[-1]-start>45:app.quit()
    timer.timeout.connect(tick);timer.start(10);QtCore.QTimer.singleShot(0,c.refresh)
    app.exec_();gaps=np.diff([start,*ticks])*1000
    result=dict(passed=c.result is not None and c.error is None and len(ticks)>20 and float(gaps.max())<250,
        max_GUI_timer_gap_ms=float(gaps.max()),ticks=len(ticks),seconds=time.monotonic()-start,
        prediction_received=c.result is not None,preview_error=c.error,hardware_actions=0,
        note='offline widget timing only; not a live-camera latency guarantee')
    c.close();result['process_stopped']=c.process.state()==QtCore.QProcess.NotRunning
    model.write(output/'RESULT.json',result);assert result['passed'] and result['process_stopped'],result
    print(result,flush=True)


def retained_check(output):
    import run
    from dataclasses import asdict
    from PyQt5 import QtCore
    cfg=model.read(run.ROOT/'CONFIG.json');output.mkdir()
    r=model.read(run.ROOT/'RETENTION_01/CURRENT_UI.json')
    def hook(w,c):
        def check():
            assert not w.hardware and not w.windows and w.worker is None and not w.armed.isChecked()
            assert asdict(w.setting())==r['heartbeat']['manual_settings']
            assert w.actual_roi==r['heartbeat']['actual_roi']
            assert w.pending_registration==r['snapshot']['geometry']
            controls=w.manual_display.controls;f=controls.filter_parameters()
            assert {k:f[k] for k in ('mode','frames','alpha','original')}==cfg['retained_manual_controls']['filter']
            assert controls.parameters(w.enhance.isChecked())==model.read(model.checked(cfg['display']))['manual']
            assert w.region_overlay.count==64 and w._dmd_side==600
            caches=model.read(model.checked(cfg['caches']))['files']
            for key in ('dmd','slm'):
                np.testing.assert_array_equal(getattr(w,key+'_file'),np.load(model.checked(caches[key])))
            assert w.ball_size_controls['radius'].value()==4
            assert w.stability_workflow.widgets['result_domain'].currentData()=='processed'
            assert not w.stability_workflow.widgets['head_enabled'].isChecked()
            assert not w.diff_mode.currentData()
            model.write(output/'RETENTION_QA.json',dict(passed=True,settings=asdict(w.setting()),roi=w.actual_roi,filter=f,
                shared_display=controls.parameters(w.enhance.isChecked()),retained64=True,caches_match=True,
                pending_R4_NOT_applied=True,geometry_exact=True,hardware_actions=0))
        QtCore.QTimer.singleShot(50,check)
    run.run_gui(cfg,output,False,qa_hook=hook)
    assert model.read(output/'RETENTION_QA.json')['passed']
    print('Retained entire window and current configuration PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('inference','window','process','async','retained'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--fixture',type=Path)
    a=p.parse_args()
    if a.mode=='inference':inference(a.output.resolve())
    elif a.mode=='window':window_check(a.config,a.fixture,a.output.resolve())
    elif a.mode=='async':asynchronous_check(a.output.resolve())
    elif a.mode=='retained':retained_check(a.output.resolve())
    else:process_check(a.output.resolve())
