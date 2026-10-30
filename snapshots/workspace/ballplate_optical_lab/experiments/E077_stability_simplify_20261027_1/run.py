"""Two-page workbench, one explicit composition; no chained experiment mains.

Default and preparation are offline. Hardware needs freshly retained state,
normal-close evidence and current shielding; no automatic capture/projection.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
LEGACY = LAB.parent/'ballplate_fourframe96_deploy'
PRIOR = LAB/'experiments/E076_dmd_scale_20261027_1'
REPAIRS = LAB/'experiments/E075_spatial_repair_20261027_1'
DISPLAY = LAB/'experiments/E074_dmd1920_rollback_20261026_1'
CATALOG = LAB/'experiments/E072_ball_size_20261026_1/catalog.py'
MODULATOR = LAB.parent/'ballplate_minframes_optical_ppo/SDON_Linux_Handoff_20261015/src/sdon/hardware/modulator.py'
MASK = LAB/'experiments/E035_mapping_night_20261024_1/REFIT_L00_BALANCED_01/BALANCED_CCD_AXES_NATIVE8.npy'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    if '.' in name:
        parent, child = name.rsplit('.', 1); setattr(sys.modules[parent], child, m)
    spec.loader.exec_module(m)
    return m


def install_modules():
    import optical_lab
    for name in ('region_overlay', 'state_sampling', 'state_sampling_gui', 'stability', 'stability_gui'):
        load('optical_lab.'+name, ROOT/'staged/optical_lab'/(name+'.py'))


def prepare(folder):
    if folder.parent != ROOT or folder.exists(): raise ValueError('fresh experiment child required')
    cfg = read(PRIOR/'CONFIG.json')
    for path, digest in cfg['source_sha256'].items():
        if sha(path) != digest: raise ValueError('preserved source changed: '+path)
    folder.mkdir()
    files = [ROOT/'run.py', ROOT/'scaling.py', *sorted((ROOT/'staged/optical_lab').glob('*.py'))]
    cfg['source_sha256'].update({str(p):sha(p) for p in files})
    cfg.update(offline_preparation_only=True, retained_dmd_scale=1.)
    cfg.pop('lifecycle_release', None)
    contract = read(cfg['contract']['path'])
    contract['source_sha256'] = cfg['source_sha256']
    contract['dmd_scale_feature'] = dict(default=1., maximum=1.8, manual_only=False, control_qualified=False)
    write(folder/'CONTRACT.json', contract); cfg['contract'] = ref(folder/'CONTRACT.json')
    write(folder/'CONFIG.json', cfg)
    return dict(config=ref(folder/'CONFIG.json'), hardware_actions=0)


def run_gui(cfg, destination, hardware):
    install_modules()
    repairs = load('e077_alignment_composition', REPAIRS/'run.py')
    repairs.install(cfg)
    display_backend = load('e077_display_identity', DISPLAY/'run.py')
    display_backend.install_display(read(cfg['display_profile']['path']))
    scale = load('e077_scaling', ROOT/'scaling.py')
    compact = load('e077_compact', LAB/'experiments/E053_raw_workbench_20261026_1/workbench.py')
    catalog = load('e077_catalog', CATALOG)
    binding = load('e077_display_binding', LAB/'experiments/E069_actual_display_binding_20261026_1/display_binding.py')
    startup = load('e077_sampler_composition', LAB/'experiments/E045_dense_state_sampling_20261025_1/run.py')
    from optical_lab import preview, exposure, display, region_overlay, state_sampling_gui, stability_gui
    from optical_lab.queued_display import queued_modulator
    layout = region_overlay.load_layout(MASK, sha(MASK), read(cfg['contract']['path'])['fixed_context']['roi'])
    output = destination/'GUI'; spool = destination/'GUI_SPOOL'
    stack = ExitStack(); binding_evidence = {}; window_box = []; errors = []
    original_load = preview._load_legacy
    def legacy(path):
        cls = original_load(path)
        import realtime_gui, diffraction_worker, sdon.hardware.modulator as mod
        stack.enter_context(binding.bound_modulators(mod, queued_modulator, MODULATOR,
            cfg['source_sha256'][str(MODULATOR)], binding_evidence))
        scale.install_window(realtime_gui.RealtimeWindow)
        scale.install_reference(diffraction_worker.DiffractionWorker)
        scale.tag_reference_results(diffraction_worker.DiffractionWorker)
        realtime_gui.RealtimeWindow.show_outputs = scale.show_outputs
        return cls
    preview._load_legacy = legacy
    original_manual = display.attach_manual_display
    def manual(w, initial=None):
        panel = original_manual(w, initial); overlay=region_overlay.attach(w, layout)
        overlay.checkbox.setChecked(cfg.get('retained_region_overlay',False))
        return panel
    display.attach_manual_display = manual
    def sampler_attached(w, sampler):
        flow = stability_gui.attach(w, cfg['contract'], destination/'TESTS', ref(MASK), layout['roi'])
        fields = dict(frames='frames', repeats='repeats', switch='switch_inputs', settle='settle_seconds',
            interval='interval_seconds', software_gain='software_gain', black='black_ADU', gamma='gamma')
        for key, field in fields.items():
            widget=flow.widgets[key]; value=cfg['settings'][field]
            widget.setChecked(value) if key=='switch' else widget.setValue(value)
        flow.widgets['exposure'].setValue(cfg['camera_requested']['exposure_us'])
        flow.widgets['gain'].setValue(cfg['camera_requested']['gain_db'])
    original_adaptation = display.attach_adaptation_display
    def finalize(w):
        result = original_adaptation(w)
        compact.attach(w); catalog.attach(w, cfg['catalog_preset']); scale.attach(w)
        flow=w.stability_workflow; q=flow.widgets
        data, assets=catalog.validate(cfg['catalog_preset']); flow.assets=assets
        flow.template['input_catalog']=dict(reference=deepcopy(cfg['catalog_preset']),metadata=data,
            loaded_at_fresh_startup_without_projection=True,control_qualified=False)
        for item in [cfg['catalog_preset']]+[v[k] for v in data['inputs'] for k in ('dmd','encoded_raw96')]:
            flow.template['source_sha256'][item['path']]=item['sha256']
        for row in data['inputs']:
            if 'SOURCE_RAW96' in row:
                item=row['SOURCE_RAW96'];flow.template['source_sha256'][item['path']]=item['sha256']
        retained=cfg.get('retained_test_controls',{})
        for key in ('input_a','input_b'):
            q[key].clear()
            for item in assets:q[key].addItem(item['input_id'],item['input_id'])
            desired=retained.get('selected_ids',{}).get(key,'CENTER' if key=='input_a' else 'VY_POS')
            q[key].setCurrentIndex(max(0,q[key].findData(desired)))
        flow.refresh_reference_assets()
        for key in ('reference_frames','baseline_frames'):
            if key in retained:q[key].setValue(retained[key])
        q['reference_mode'].setChecked(retained.get('reference_mode',False))
        wanted=retained.get('selected_ids',{}).get('input_ref')
        if wanted is not None:q['input_ref'].setCurrentIndex(max(0,q['input_ref'].findData(wanted)))
        catalog.refresh_size_view(w)
        if cfg.get('retained_pending_ball_radius') is not None:
            w.ball_size_controls['radius'].setValue(cfg['retained_pending_ball_radius'])
        head=cfg.get('head_preset')
        if head:
            from optical_lab import stability
            if sha(head['path']) != head['sha256']:raise ValueError('head source changed')
            flow.head_record=stability.load_head(head['path'])
            q['head_enabled'].setChecked(retained.get('head_enabled',False))
            q['head_path'].setText(head['path'])
        w.dmd_scale.setValue(cfg.get('retained_dmd_scale',1.))
        w.tabs.setTabText(1,'八区稳定性测试')
        w.raw_workbench['note'].setText('沿用当前SLM、ROI和DMD缩放；每次独立N帧RAW均值，R0–R7统一统计。亮度仅提示；不运行控制。')
        w.setWindowTitle('光路工作台 · 八区稳定性／手动对齐'+(' · 真实器件' if hardware else ' · 离线检查'))
        q['status'].setText('未自动投图或采集。选择输入与N，确认光路条件后点击一键测试。')
        assert w.tabs.count()==2 and w.worker is None and not w.armed.isChecked()
        evidence=dict(pages=[w.tabs.tabText(i) for i in range(2)],
            display_binding=binding.verify_instances(w,binding_evidence),automatic_projection=False,automatic_capture=False)
        write((output if hardware else destination)/'COMPOSITION.json',evidence); window_box.append(w)
        if not hardware:
            from PyQt5 import QtCore,QtWidgets,QtGui
            import numpy as np
            def screenshots():
                assert not w.hardware and not w.windows and w.worker is None
                w.resize(1500,1000); w.tabs.setCurrentIndex(0)
                # Synthetic downsampled preview exercises proportional eight-region overlay.
                image=np.zeros((416,416),np.uint8)
                w.view.active_roi=dict(OffsetX=744,OffsetY=276,Width=832,Height=832)
                w.view.set_image(image,extent=(832,832))
                w.fault=None;w.region_overlay.checkbox.setChecked(True);w.region_overlay.refresh()
                assert len(w.region_overlay.plan())==8 and w.region_overlay.last_error is None
                QtWidgets.QApplication.processEvents();w.grab().save(str(destination/'manual_eight_regions.png'))
                w.tabs.setCurrentIndex(1);QtWidgets.QApplication.processEvents()
                w.grab().save(str(destination/'stability_controls.png'))
                write(destination/'SMOKE.json',dict(passed=True,hardware_actions=0,
                    pages=2,all_eight_overlay=True,downsampled_preview_supported=True,
                    no_scale_start_wrapper=flow.start.__func__ is stability_gui.Workflow.start))
            QtCore.QTimer.singleShot(100,screenshots)
        return result
    display.attach_adaptation_display=finalize
    args=['--legacy-root',str(LEGACY),'--output',str(output),
        '--restart-state',cfg['restart']['path'],'--recovery-record',cfg['recovery']['path'],
        '--display-state',cfg['display']['path'],'--display-state-sha256',cfg['display']['sha256'],
        '--preview-caches',cfg['caches']['path'],'--preview-caches-sha256',cfg['caches']['sha256']]
    if cfg.get('geometry'):
        args+=['--manual-geometry-state',cfg['geometry']['path'],'--manual-geometry-sha256',cfg['geometry']['sha256']]
    else:
        # Matching current reference uses the original pending-registration path.
        # Restore pattern modes after manual controls exist, with output disarmed.
        original_finalize=display.attach_adaptation_display
        def with_retained_modes(w):
            value=original_finalize(w)
            state=cfg.get('retained_manual_controls',{})
            for widget,key in ((w.dmd_mode,'dmd'),(w.slm_mode,'slm')):
                if state.get('settings'):widget.setCurrentIndex(widget.findData(state['settings'][key]))
            if 'orientation' in state:w.orientation.setCurrentIndex(state['orientation'])
            registration=read(cfg['restart']['path']).get('registration')
            if registration is not None:
                if registration.get('dmd_side',600)!=w._dmd_side:raise ValueError('retained matrix DMD scale mismatch')
                # Startup scale/direction widgets clear mapping; restore ONLY
                # pending evidence after these widgets, not a validated result.
                assert w.worker is None and not w.armed.isChecked()
                w.pending_registration=deepcopy(registration)
                w.mapping_status.setText('保留当前倍率；待新CCD参数签名复核后恢复，不重新拟合。')
            w.changed()
            return value
        display.attach_adaptation_display=with_retained_modes
    args+=['--hardware','--light-blocked-confirmed'] if hardware else ['--smoke-test']
    old_hook=sys.excepthook
    def exception(typ,error,tb):errors.append(str(error));old_hook(typ,error,tb)
    if not hardware:sys.excepthook=exception
    try:
        with stack,startup.startup_composition(exposure,display,state_sampling_gui.attach,cfg['contract'],spool,read(cfg['display']['path']),sampler_attached):
            preview.main(args)
    finally:
        preview._load_legacy=original_load;display.attach_manual_display=original_manual
        display.attach_adaptation_display=original_adaptation;sys.excepthook=old_hook
    if not window_box or errors:raise RuntimeError('composition failed: '+str(errors))
    if not hardware and not read(destination/'SMOKE.json')['passed']:raise RuntimeError('smoke failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group();g.add_argument('--prepare-offline',type=Path)
    g.add_argument('--offline-smoke',action='store_true');g.add_argument('--hardware',action='store_true')
    p.add_argument('--config',type=Path);p.add_argument('--config-sha256');p.add_argument('--output',type=Path)
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if a.prepare_offline:print(json.dumps(prepare(a.prepare_offline.resolve()),ensure_ascii=False));return
    if not(a.offline_smoke or a.hardware):print('{"default_IO":0}');return
    if a.config:a.config=a.config.resolve()
    if a.output:a.output=a.output.resolve()
    if not a.config or sha(a.config)!=a.config_sha256:raise ValueError('explicit config SHA required')
    cfg=read(a.config)
    for path,digest in cfg['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('source changed: '+path)
    for key in ('contract','restart','recovery','display','caches','display_profile'):
        if sha(cfg[key]['path'])!=cfg[key]['sha256']:raise ValueError('config artifact changed: '+key)
    if cfg.get('geometry') and sha(cfg['geometry']['path'])!=cfg['geometry']['sha256']:raise ValueError('geometry artifact changed')
    if a.hardware:
        if not a.light_blocked_confirmed or cfg.get('offline_preparation_only') is not False or a.output is not None:
            raise ValueError('current shielding and fresh sealed hardware configuration required')
        proof=cfg['lifecycle_release'];evidence=read(proof['path'])
        if sha(proof['path'])!=proof['sha256'] or not all(evidence.get(k) is True for k in
            ('completed','old_normal_exit_verified','cold_baseline_verified','physical_shielding_confirmed')):
            raise ValueError('normal-close and standby verification required')
        backend=load('e077_owner_check',DISPLAY/'run.py');backend.no_owner()
        if a.config!=ROOT/'CONFIG.json' or os.environ.get('QT_QPA_PLATFORM')!='xcb':raise ValueError('sealed root config/xcb required')
        destination=ROOT
    else:
        if not a.output or a.output.parent!=ROOT or a.output.exists() or os.environ.get('QT_QPA_PLATFORM')!='offscreen' or a.light_blocked_confirmed:
            raise ValueError('fresh offscreen output required')
        destination=a.output;destination.mkdir()
    run_gui(cfg,destination,a.hardware)


if __name__=='__main__':main()
