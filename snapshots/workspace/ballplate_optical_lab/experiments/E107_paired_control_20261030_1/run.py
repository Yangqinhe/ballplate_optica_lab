"""E081 workbench + preserved E106 mean preview + one paired control tab.

Offline by default. The old GUI, hardware service and archived code are never
modified. Hardware start requires a newly retained, shielded lifecycle record.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import core as c


def base_module():
    old=c.EXP/'E081_readout_regions_20261028_1/run.py'
    if c.digest(old)!='80d30c45380a6faf84c53814f7b7131044651f3a99f98314bc3d751a413b4b7f':
        raise ValueError('frozen workbench changed')
    return c.load(old,'e107_base_workbench')


def prepare(output,source):
    cfg=c.read(source)
    for path,sha in cfg['source_sha256'].items():
        if c.digest(path)!=sha:raise ValueError('source changed: '+path)
    cfg['paired_control']=c.ref(c.ROOT/'HEADS_02/PACKAGE.json')
    for name in ('run.py','core.py','page.py','worker.py'):
        p=c.ROOT/name;cfg['source_sha256'][str(p)]=c.digest(p)
    for name in ('model.py','panel.py'):
        p=c.PREVIEW/name;cfg['source_sha256'][str(p)]=c.digest(p)
    cfg['offline_preparation_only']=True;cfg.pop('lifecycle_release',None)
    output.mkdir();contract=c.read(c.checked(cfg['contract']));contract['source_sha256']=cfg['source_sha256']
    c.write(output/'CONTRACT.json',contract);cfg['contract']=c.ref(output/'CONTRACT.json')
    c.write(output/'CONFIG.json',cfg);return c.ref(output/'CONFIG.json')


def run_gui(cfg,destination,hardware,qa_hook=None):
    destination=Path(destination).resolve()
    from PyQt5 import QtCore
    import page
    preview_core=c.load(c.PREVIEW/'model.py','e107_existing_preview_model')
    preview_panel=c.load(c.PREVIEW/'panel.py','e107_existing_preview_panel')
    base=base_module();original=base.load
    def composed(name,path):
        module=original(name,path)
        if name!='e079_compact':return module
        def attach(w):
            value=module.attach(w)
            saved=cfg.get('retained_view_choices',{})
            if saved:
                w.enhance.setChecked(saved['enhance'])
                w.target_mode.setCurrentIndex(w.target_mode.findData(saved['fixed_target']))
                w.diff_mode.setCurrentIndex(w.diff_mode.findData(saved['overlay']))
                q=w.stability_workflow.widgets['result_domain'];q.setCurrentIndex(q.findData(saved['result_domain']))
            preview=preview_panel.attach(w,c.checked(cfg['surrogate_preview']),preview_core)
            # Original composition validates its own two pages before returning.
            # Attach the new independent page on the next GUI turn, before any
            # user action; startup still never starts acquisition/projection.
            def finish():
                controller=page.attach(w,cfg['paired_control'],Path(destination)/'PAIRED_RUNS')
                c.write(Path(destination)/'PAIRED_COMPOSITION.json',dict(
                    pages=[w.tabs.tabText(i) for i in range(w.tabs.count())],
                    package=cfg['paired_control'],automatic_capture=False,automatic_projection=False,
                    previous_two_pages_preserved=True,same_camera_and_display_owner=True))
                if qa_hook is not None:
                    if hardware:raise ValueError('QA cannot run on hardware')
                    qa_hook(w,controller,preview)
            QtCore.QTimer.singleShot(0,finish)
            return value
        return SimpleNamespace(attach=attach)
    base.load=composed
    return base.run_gui(cfg,destination,hardware)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-offline',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--config-sha256');p.add_argument('--output',type=Path)
    g=p.add_mutually_exclusive_group();g.add_argument('--offline-smoke',action='store_true');g.add_argument('--hardware',action='store_true')
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if a.prepare_offline:
        print(prepare(a.prepare_offline,a.config or c.PREVIEW/'CONFIG.json'));return
    if not (a.offline_smoke or a.hardware):print('{"hardware_actions":0}');return
    cfg=c.read(c.checked(dict(path=str(a.config),sha256=a.config_sha256)))
    for path,sha in cfg['source_sha256'].items():
        if c.digest(path)!=sha:raise ValueError('source changed: '+path)
    for key in ('contract','restart','recovery','display','caches','display_profile','surrogate_preview','paired_control'):c.checked(cfg[key])
    if a.hardware:
        if not a.light_blocked_confirmed or cfg.get('offline_preparation_only') is not False or a.output:
            raise ValueError('fresh shielded lifecycle and retained config required')
        proof=c.read(c.checked(cfg['lifecycle_release']))
        if not all(proof.get(k) is True for k in ('completed','old_normal_exit_verified','cold_baseline_verified','physical_shielding_confirmed')):
            raise ValueError('normal close/standby evidence required')
        if a.config.resolve()!=c.ROOT/'CONFIG.json' or os.environ.get('QT_QPA_PLATFORM')!='xcb':raise ValueError('sealed config/xcb required')
        base=base_module();base.load('e107_no_owner',base.DISPLAY/'run.py').no_owner();output=c.ROOT
    else:
        if not a.output or a.output.exists() or os.environ.get('QT_QPA_PLATFORM')!='offscreen':raise ValueError('fresh offscreen output required')
        output=a.output.resolve();output.mkdir()
    run_gui(cfg,output,a.hardware)

if __name__=='__main__':main()
