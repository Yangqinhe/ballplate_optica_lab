"""Offline numerical and full-GUI checks; fake CCD is explicitly labelled."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import core as c


def worker_check(out):
    import worker
    pair=worker.Pair(c.read(c.ROOT/'HEADS_02/PACKAGE.json'),out)
    checks=[];reports=[]
    try:
        s=pair.reset([.03,-.02]);assert s[0]==s[1];checks.append('identical reset')
        r=pair.prepare(1,pair.package['context']);assert pair.states()==s
        time.sleep(.1);assert pair.states()==s;checks.append('wall-clock wait does not advance simulation')
        with np.load(r['archive']) as a:im=a['simulation_CCD'];dmd=a['simulation_DMD'];slm=a['SLM']
        path=Path(r['archive']).parent/'SYNTHETIC_QA_NOT_CCD.npz'
        with path.open('xb') as f:np.savez(f,raw=np.rint(im)[None].astype(np.uint8),image=im)
        result=pair.advance(1,path,[.005,90.],model_only=True);reports.append(result)
        assert not result['hardware_CCD_used']
        np.testing.assert_allclose([x['time'] for x in result['states_after']],.02,atol=1e-12)
        np.testing.assert_allclose(result['delta_v_world'],[0,.1],atol=1e-12)
        checks+=['both advance exactly 20ms only after observation','impulse N s converted using 0.05kg mass']
        before=pair.states()
        try:pair.advance(1,path,[0,0],model_only=True);raise AssertionError('duplicate accepted')
        except ValueError:pass
        assert pair.states()==before;checks.append('duplicate observation cannot advance twice')
        for i in range(2,12):
            r=pair.prepare(i,pair.package['context'])
            with np.load(r['archive']) as a:im=a['simulation_CCD']
            path=Path(r['archive']).parent/'SYNTHETIC_QA_NOT_CCD.npz'
            with path.open('xb') as f:np.savez(f,raw=np.rint(im)[None].astype(np.uint8),image=im)
            result=pair.advance(i,path,[0,0],model_only=True);reports.append(result)
        checks.append('eleven independently encoded history/optics/action/physics iterations')
        pair.reset([0,0]);assert all(x['time']==0 for x in pair.states())
        assert not any(w['history'].frames for w in pair.worlds);checks.append('reset clears optical history')
        c.write(out/'QA.json',dict(passed=True,checks=checks,steps=len(reports),hardware_actions=0,
            NOT_a_control_performance_test=True,real_branch_used_synthetic_frame=True))
        with (out/'UI_FIXTURE.npz').open('xb') as f:np.savez(f,dmd=dmd,slm=slm,ccd=im)
        print(checks,flush=True)
    finally:pair.close()


def window_check(config,out,fixture):
    import run
    from dataclasses import asdict
    from PyQt5 import QtCore,QtWidgets
    out.mkdir();checks=[];cfg=c.read(config)
    with np.load(fixture) as a:images={k:a[k].copy() for k in a.files}
    def hook(w,p,preview):
        preview.timer.stop();w.timer.stop()
        def check():
            try:
                assert w.tabs.count()==3 and not w.windows and w.worker is None and not w.armed.isChecked()
                checks.append('three tabs, original two preserved; no hardware/auto projection')
                assert asdict(w.setting())==cfg['retained_manual_controls']['settings']
                assert w.actual_roi==c.read(c.checked(cfg['restart']))['roi']
                assert w.pending_registration==c.read(c.checked(cfg['restart']))['registration']
                for name,item in c.read(c.checked(cfg['caches']))['files'].items():
                    np.testing.assert_array_equal(getattr(w,name+'_file'),np.load(c.checked(item)))
                filtering=w.manual_display.controls.filter_parameters()
                for key,value in cfg['retained_manual_controls']['filter'].items():assert filtering[key]==value
                actual=w.manual_display.controls.parameters(True);wanted=c.read(c.checked(cfg['display']))['manual']
                assert actual['mode']==wanted['mode']
                np.testing.assert_allclose([actual[k] for k in ('gain','black','gamma')],[wanted[k] for k in ('gain','black','gamma')],atol=1e-12)
                assert w.region_overlay.count==cfg['retained_region_count'] and w._dmd_side==round(cfg['retained_dmd_scale']*600)
                assert w.ball_size_controls['radius'].value()==cfg['retained_pending_ball_radius']
                checks.append('retained ROI, pending geometry, pattern caches, shared filter/display, DMD scale, ball size and regions')
                w.resize(1820,1080);w.tabs.setCurrentIndex(p.index)
                state=dict(time=.12,position=[.03,-.02,.036],velocity_world=[.02,-.01],angles=[.01,-.02],
                    filtered_action=[.02,.03],fallen=False)
                p.show_states([state,state],True)
                for v in p.views:
                    v['dmd'].put(images['dmd']);v['slm'].put(images['slm']);v['ccd'].put(p.display(images['ccd']),p.package['layout'])
                    z=c.integrate(images['ccd'],p.package['layout']);v['bars'].put(z,float(max(z)))
                p.radius.setValue(12);p.impulse.setValue(.004);p.angle.setValue(90);p.push()
                assert p.pending_impulse==[.004,90.]
                p.reset([0,0]);assert p.pending_reset==[0,0]
                checks.append('reset and impulse are queued, never advance physics on UI clicks')
                p.start();assert not p.go and '真实采集' in p.info.text();assert p.process.state()==0
                checks.append('offline Start does not start hardware or fake actual CCD')
                p.info.setText('离线界面检查 · 图片为合成模型数据，不是实际CCD采集；两个MuJoCo示例状态相同。')
                QtWidgets.QApplication.processEvents();w.grab().save(str(out/'paired_page.png'))
                assert all(v['dmd'].image is not None and v['slm'].image is not None and v['ccd'].image is not None for v in p.views)
                assert p.root is None and not w.windows and w.worker is None
                c.write(out/'QA.json',dict(passed=True,checks=checks,hardware_actions=0,pages=[w.tabs.tabText(i) for i in range(3)]))
            except Exception as e:
                c.write(out/'QA_FAILED.json',dict(error=repr(e)));raise
        QtCore.QTimer.singleShot(350,check)
    run.run_gui(cfg,out,False,qa_hook=hook)
    assert c.read(out/'QA.json')['passed']


def protocol_check(out):
    out.mkdir();package=c.ref(c.ROOT/'HEADS_02/PACKAGE.json');cfg=c.read(package['path'])
    commands=[dict(op='init',id=1),dict(op='prepare',id=2,context=cfg['context']),
        dict(op='cancel',id=3),dict(op='reset',id=4,xy=[.03,-.02]),dict(op='close',id=5)]
    start=time.monotonic()
    result=subprocess.run([sys.executable,'-B',str(c.ROOT/'worker.py'),'--package',package['path'],
        '--sha256',package['sha256'],'--output',str(out/'WORKER')],
        input='\n'.join(json.dumps(q) for q in commands)+'\n',capture_output=True,text=True,timeout=60)
    rows=[json.loads(s[5:]) for s in result.stdout.splitlines() if s.startswith('E107 ')]
    report=dict(returncode=result.returncode,messages=rows,stderr=result.stderr,seconds=time.monotonic()-start,
        hardware_actions=0,source=c.ref(__file__))
    report['passed']=result.returncode==0 and len(rows)==4 and all(r['ok'] for r in rows)
    c.write(out/'QA.json',report)
    assert report['passed'],report
    assert all(s['time']==0 for r in rows for s in r['result'].get('states',[]))
    print('Real separate-process transport PASS; no CCD/no physics advancement',flush=True)


def regression_check(out):
    import os,re
    out.mkdir();python=str(c.LAB.parent/'ballplate_fourframe96_deploy/.calibration-venv/bin/python')
    old=c.EXP/'E081_readout_regions_20261028_1'
    # Do not import old test_regression.py: it writes to its frozen experiment.
    script=f'''import sys,unittest
sys.path[:0]=[{str(old)!r},{str(c.LAB/'tests')!r}]
import run
run.load('alignment',run.REPAIRS/'alignment.py')
run.load('catalog',run.CATALOG)
a=run.load('e107_alignment_tests',run.REPAIRS/'test_alignment.py')
b=run.load('e107_size_tests',run.LAB/'experiments/E072_ball_size_20261026_1/test_ball_size.py')
s=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(a),unittest.defaultTestLoader.loadTestsFromTestCase(b.SizeTests)])
for name in ('test_display','test_manual_exposure','test_camera_gain','test_manual_geometry_preview'):
    s.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
r=unittest.TextTestRunner().run(s)
raise SystemExit(not r.wasSuccessful())'''
    entries=[('pair',[python,'-B','-m','unittest','discover','-s',str(c.ROOT),'-p','test_pair.py']),
        ('flow',[python,'-B','-m','unittest','discover','-s',str(c.ROOT),'-p','test_flow.py']),
        ('preview',[python,'-B','-m','unittest','discover','-s',str(c.PREVIEW),'-p','test_preview.py']),
        ('sampling',[python,'-B','-c',f"import sys,unittest;sys.path.insert(0,{str(old)!r});s=unittest.defaultTestLoader.loadTestsFromNames(['test_effective','test_regions','test_runtime']);r=unittest.TextTestRunner().run(s);raise SystemExit(not r.wasSuccessful())"]),
        ('original',[python,'-B','-c',script])]
    results=[];env=dict(os.environ,QT_QPA_PLATFORM='offscreen',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONPATH=str(c.LAB/'src'))
    for name,cmd in entries:
        r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=60,cwd=c.LAB)
        match=re.search(r'Ran (\d+) tests',r.stderr)
        item=dict(name=name,returncode=r.returncode,tests=int(match[1]) if match else None,stdout=r.stdout,stderr=r.stderr)
        results.append(item);c.write(out/(name+'.json'),item);print(name,r.returncode,item['tests'],flush=True)
    report=dict(passed=all(r['returncode']==0 for r in results),tests=sum(r['tests'] or 0 for r in results),
        results=results,hardware_actions=0)
    c.write(out/'QA.json',report);assert report['passed'],results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('worker','window','protocol','regression'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--fixture',type=Path)
    a=p.parse_args()
    if a.mode=='worker':worker_check(a.output)
    elif a.mode=='protocol':protocol_check(a.output.resolve())
    elif a.mode=='regression':regression_check(a.output.resolve())
    else:window_check(a.config,a.output,a.fixture)
