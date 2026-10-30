"""Fresh two-page composition. Default no IO; offline preparation touches files only.

Hardware requires newly sealed normal-exit/standby evidence AND current shielding.
Never changes the existing owner's source closure or injects code into it.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent;LAB=ROOT.parents[1]
BASE=LAB/'experiments/E052_n60_crash_recovery_20261026_1'
PRIOR=LAB/'experiments/E069_actual_display_binding_20261026_1'
ADAPTER=LAB/'experiments/E068_sampling_order_20261026_1/adapter.py'
CATALOG=ROOT/'catalog.py'
SOURCE=LAB/'experiments/E071_stability_reference_ui_20261026_1'
HEAD=LAB/'experiments/E067_three_slm_20261026_1/ADAPT_L_PHYS20_01/HEAD.json'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def prepare_offline(destination):
    if destination.parent!=ROOT or destination.exists():raise ValueError('fresh child directory required')
    cfg=read(SOURCE/'CONFIG.json')
    for p,d in cfg['source_sha256'].items():
        if sha(p)!=d:raise ValueError('old pinned source changed: '+p)
    destination.mkdir()
    sources=dict(cfg['source_sha256'])
    for p in (Path(__file__).resolve(),ROOT/'staged/optical_lab/stability.py',ROOT/'staged/optical_lab/stability_gui.py',
              ADAPTER,CATALOG,PRIOR/'display_binding.py',HEAD):sources[str(p)]=sha(p)
    bank_path=LAB/'experiments/E067_three_slm_20261026_1/HARDWARE_BANK_01/CATALOG.json'
    sparse_path=LAB/'experiments/E070_reference_budget_20261026_1/ASSETS_01/CATALOG.json'
    # Nine explicit position/velocity diagnostics plus the sparse reference.
    bank=read(bank_path);sparse=read(sparse_path);bank['inputs']=[v for v in bank['inputs'] if v.get('split')=='diagnostic']+[sparse['inputs'][0]]
    bank['provenance']=[ref(bank_path),ref(sparse_path)]
    write(destination/'CATALOG.json',bank);cfg['catalog_preset']=ref(destination/'CATALOG.json')
    contract=read(cfg['contract']['path']);contract['source_sha256']=sources
    write(destination/'CONTRACT.json',contract);cfg['contract']=ref(destination/'CONTRACT.json')
    cfg['source_sha256']=sources;cfg['head_preset']=ref(HEAD);cfg['offline_preparation_only']=True
    cfg.pop('crash_recovery',None)
    write(destination/'CONFIG.json',cfg)
    return dict(prepared=ref(destination/'CONFIG.json'),hardware_actions=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group();g.add_argument('--prepare-offline',type=Path);g.add_argument('--offline-smoke',action='store_true');g.add_argument('--hardware',action='store_true')
    p.add_argument('--config',type=Path);p.add_argument('--config-sha256');p.add_argument('--output',type=Path)
    p.add_argument('--light-blocked-confirmed',action='store_true');p.add_argument('--sample-result',type=Path)
    a=p.parse_args()
    if a.prepare_offline:print(json.dumps(prepare_offline(a.prepare_offline),ensure_ascii=False));return
    if not (a.offline_smoke or a.hardware):print('{"default_IO":0}');return
    if not a.config or sha(a.config)!=a.config_sha256:raise ValueError('explicit config path and SHA required')
    cfg=read(a.config)
    for path,digest in cfg['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('source changed: '+path)
    if a.hardware:
        if cfg.get('offline_preparation_only') is not False or not a.light_blocked_confirmed or a.output is not None or a.sample_result:
            raise ValueError('fresh retained configuration, shielding, no replay and hardware destination required')
        if os.environ.get('QT_QPA_PLATFORM')!='xcb':raise ValueError('hardware xcb required')
        lifecycle=cfg['lifecycle_release']
        if sha(lifecycle['path'])!=lifecycle['sha256']:raise ValueError('lifecycle hash differs')
        evidence=read(lifecycle['path'])
        if not all(evidence.get(k) is True for k in ('completed','old_normal_exit_verified','cold_baseline_verified','physical_shielding_confirmed')):
            raise ValueError('normal close and cold rollback must be verified')
        if Path('/proc/'+str(cfg['replaced_owner_pid'])).exists():raise ValueError('prior owner still exists')
        destination=ROOT
        if a.config!=ROOT/'CONFIG.json':raise ValueError('hardware config must be sealed at experiment root')
    else:
        if a.light_blocked_confirmed or os.environ.get('QT_QPA_PLATFORM')!='offscreen' or a.output is None or a.output.parent!=ROOT or a.output.exists():
            raise ValueError('fresh offscreen destination required')
        destination=a.output;destination.mkdir();write(destination/'CONFIG.json',cfg)
    import optical_lab
    base=load('e072_frozen_composition',BASE/'run.py');base.ROOT=destination
    original_load=base.load
    def staged_load(name,path):
        if name in ('optical_lab.stability','optical_lab.stability_gui'):
            path=ROOT/'staged/optical_lab'/(name.rsplit('.',1)[1]+'.py')
        return original_load(name,path)
    base.load=staged_load
    base.load('optical_lab.state_sampling_gui',ADAPTER)
    from optical_lab import display,queued_display
    compact=load('e072_compact',LAB/'experiments/E053_raw_workbench_20261026_1/workbench.py')
    catalog=load('e072_catalog',CATALOG);catalog.validate(cfg['catalog_preset'])
    binding_mod=load('e072_display_binding',PRIOR/'display_binding.py')
    original_context=queued_display.queued_modulator;original_attach=display.attach_adaptation_display
    binding={};result={}
    def context(module):
        return binding_mod.bound_modulators(module,original_context,base.MODULATOR,cfg['source_sha256'][str(base.MODULATOR)],binding)
    def finalize(window):
        attached=original_attach(window);compact.attach(window);catalog.attach(window,cfg['catalog_preset'])
        # Startup-only file catalogue; never bypass a live owner's parked check.
        flow=window.stability_workflow;q=flow.widgets
        assert window.worker is None and not window.running and not window.armed.isChecked()
        data,assets=catalog.validate(cfg['catalog_preset'])
        flow.assets=assets
        flow.template['input_catalog']=dict(reference=deepcopy(cfg['catalog_preset']),metadata=data,
            loaded_at_fresh_startup_without_projection=True,control_qualified=False)
        for record in [cfg['catalog_preset']]+[v[k] for v in data['inputs'] for k in ('dmd','encoded_raw96')]:
            flow.template['source_sha256'][record['path']]=record['sha256']
        for row in data['inputs']:
            if 'SOURCE_RAW96' in row:
                record=row['SOURCE_RAW96'];flow.template['source_sha256'][record['path']]=record['sha256']
        for key in ('input_a','input_b'):
            q[key].clear()
            for item in assets:q[key].addItem(item['input_id'],item['input_id'])
        retained=cfg.get('retained_test_controls',{})
        q['reference_mode'].setChecked(retained.get('reference_mode',False))
        for key in ('input_a','input_b'):
            index=q[key].findData(retained.get('selected_ids',{}).get(key,'CENTER' if key=='input_a' else 'VY_POS'))
            if index>=0:q[key].setCurrentIndex(index)
        flow.refresh_reference_assets()
        for key in ('reference_frames','baseline_frames'):
            if key in retained:q[key].setValue(retained[key])
        selected_ref=retained.get('selected_ids',{}).get('input_ref')
        if selected_ref is not None:
            index=q['input_ref'].findData(selected_ref)
            if index<0:raise ValueError('retained reference not available')
            q['input_ref'].setCurrentIndex(index)
        catalog.refresh_size_view(window)
        head=cfg.get('head_preset')
        if head:
            if sha(head['path'])!=head['sha256']:raise ValueError('head preset SHA differs')
            from optical_lab import stability
            flow.head_record=stability.load_head(head['path']);q['head_enabled'].setChecked(retained.get('head_enabled',True))
            q['head_path'].setText(head['path']+' · SHA '+head['sha256'][:12])
        window.tabs.setTabText(1,'八区稳定性测试')
        window.raw_workbench['note'].setText('RAW为主基线；支持参考/目标预算与固定18参数头静态诊断。显示参数只影响对照分支；本页不推进MuJoCo。')
        window.setWindowTitle('光路工作台 · 八区稳定性／参考补偿／球尺寸'+(' · 真实器件' if a.hardware else ' · 离线检查'))
        q['status'].setText('未自动投图或采集。先载入验证清单，选择目标/参考，再显式确认并开始；参数变化会记录为控制头条件差异。')
        assert window.tabs.count()==2 and window.worker is None and not window.armed.isChecked() and not q['permission'].isChecked()
        result.update(pages=[window.tabs.tabText(i) for i in range(2)],automatic_projection=False,automatic_capture=False,
            all_eight_regions=True,reference_budget_controls=True,control_metrics=6,ball_size_controls=True,hardware_qualified=False,
            actual_display_binding=binding_mod.verify_instances(window,binding))
        if a.hardware:write(ROOT/'GUI/comparison_extension.json',result)
        else:
            from PyQt5 import QtCore,QtWidgets
            def screenshot():
                window.resize(1500,1000);window.tabs.setCurrentIndex(1)
                window.grab().save(str(destination/'controls.png'))
                scroll=q['page'].findChild(QtWidgets.QScrollArea)
                QtWidgets.QApplication.processEvents()
                scroll.ensureWidgetVisible(window.ball_size_controls['A_preview'])
                QtWidgets.QApplication.processEvents()
                window.grab().save(str(destination/'ball_size_R2.png'))
                before={role:None if getattr(window,role+'_file',None) is None else getattr(window,role+'_file').copy() for role in ('dmd','slm')}
                window.ball_size_controls['radius'].setValue(0)
                # The retained offline startup carries a deliberate recovery fault.
                # Simulate healthy parked state ONLY in this offscreen, no-device test.
                assert not window.hardware and not window.windows and window.worker is None
                saved_fault=window.fault
                try:
                    window.fault=None
                    catalog.apply_size(window)
                finally:window.fault=saved_fault
                QtWidgets.QApplication.processEvents()
                scroll.ensureWidgetVisible(window.ball_size_controls['A_preview'])
                window.grab().save(str(destination/'ball_size_R0.png'))
                import numpy as np
                assert all(value is None or np.array_equal(value,getattr(window,role+'_file')) for role,value in before.items())
                result['size_smoke']=dict(R0_preview=window.ball_size_controls['actual'].text(),
                    physical_output_caches_unchanged=True,only_offline_new_asset_files=True,
                    healthy_parked_state_simulated_offline=True,real_camera_recovery_not_tested=True)
                if a.sample_result:
                    flow.folder=a.sample_result.parent.parent;flow.result=read(a.sample_result);flow.show_result()
                    QtWidgets.QApplication.processEvents()
                    scroll=q['page'].findChild(QtWidgets.QScrollArea)
                    scroll.ensureWidgetVisible(q['table']);window.grab().save(str(destination/'eight_regions.png'))
                    scroll.ensureWidgetVisible(q['control_table']);window.grab().save(str(destination/'control_metrics.png'))
                    scroll.ensureWidgetVisible(q['analysis_plot']);window.grab().save(str(destination/'analysis.png'))
                    result['replay_only']=ref(a.sample_result)
            QtCore.QTimer.singleShot(100,screenshot)
        return attached
    queued_display.queued_modulator=context;display.attach_adaptation_display=finalize
    try:
        base.execute(SimpleNamespace(config_sha256=sha(destination/'CONFIG.json'),hardware=a.hardware,light_blocked_confirmed=a.light_blocked_confirmed))
    finally:
        queued_display.queued_modulator=original_context;display.attach_adaptation_display=original_attach
    if not result:raise RuntimeError('composition not exercised')
    if not a.hardware and 'size_smoke' not in result:raise RuntimeError('offline ball-size UI check did not complete')
    if not a.hardware:write(destination/'RESULT.json',result)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
