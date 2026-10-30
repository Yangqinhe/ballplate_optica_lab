"""E081 workbench plus a read-only surrogate panel, by explicit composition.

Default offline; no new inheritance layer, no changes to frozen E081/E105.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import model

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parents[1]
PRIOR=LAB/'experiments/E081_readout_regions_20261028_1'
BASE_SHA='80d30c45380a6faf84c53814f7b7131044651f3a99f98314bc3d751a413b4b7f'


def base_module():
    if model.digest(PRIOR/'run.py')!=BASE_SHA:raise ValueError('original E081 composition changed')
    spec=importlib.util.spec_from_file_location('e106_base_workbench',PRIOR/'run.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    return base


def prepare(output,source):
    if output.parent!=ROOT or output.exists():raise ValueError('new preparation directory required')
    cfg=model.read(source)
    for path,sha in cfg['source_sha256'].items():
        if model.digest(path)!=sha:raise ValueError('preserved source changed: '+path)
    cfg['surrogate_preview']=model.ref(ROOT/'MODEL_01/MODEL.json')
    for p in (ROOT/'run.py',ROOT/'model.py',ROOT/'panel.py',PRIOR/'run.py'):
        cfg['source_sha256'][str(p)]=model.digest(p)
    cfg['offline_preparation_only']=True;cfg.pop('lifecycle_release',None)
    output.mkdir();contract=model.read(model.checked(cfg['contract']))
    contract['source_sha256']=cfg['source_sha256']
    model.write(output/'CONTRACT.json',contract);cfg['contract']=model.ref(output/'CONTRACT.json')
    model.write(output/'CONFIG.json',cfg)
    return model.ref(output/'CONFIG.json')


def run_gui(cfg,destination,hardware,qa_hook=None):
    from panel import attach
    base=base_module();original=base.load
    def composed(name,path):
        module=original(name,path)
        if name=='e079_compact':
            def add(window):
                value=module.attach(window)
                saved=cfg.get('retained_view_choices',{})
                if saved:
                    window.enhance.setChecked(saved['enhance'])
                    window.target_mode.setCurrentIndex(window.target_mode.findData(saved['fixed_target']))
                    window.diff_mode.setCurrentIndex(window.diff_mode.findData(saved['overlay']))
                    q=window.stability_workflow.widgets['result_domain']
                    q.setCurrentIndex(q.findData(saved['result_domain']))
                controller=attach(window,model.checked(cfg['surrogate_preview']),model)
                if qa_hook is not None:
                    if hardware:raise ValueError('offline QA cannot enter a hardware process')
                    qa_hook(window,controller)
                return value
            return SimpleNamespace(attach=add)
        return module
    base.load=composed
    return base.run_gui(cfg,destination,hardware)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-offline',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--config-sha256');p.add_argument('--output',type=Path)
    g=p.add_mutually_exclusive_group();g.add_argument('--offline-smoke',action='store_true');g.add_argument('--hardware',action='store_true')
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if a.prepare_offline:
        print(prepare(a.prepare_offline.resolve(),a.config or PRIOR/'CONFIG.json'));return
    if not(a.offline_smoke or a.hardware):print('{"hardware_actions":0}');return
    if not a.config or model.digest(a.config)!=a.config_sha256:raise ValueError('explicit config SHA required')
    cfg=model.read(a.config)
    for path,sha in cfg['source_sha256'].items():
        if model.digest(path)!=sha:raise ValueError('source changed: '+path)
    for key in ('contract','restart','recovery','display','caches','display_profile','surrogate_preview'):
        model.checked(cfg[key])
    if a.hardware:
        if not a.light_blocked_confirmed or cfg.get('offline_preparation_only') is not False or a.output:
            raise ValueError('fresh shielded lifecycle and retained config required')
        proof=model.read(model.checked(cfg['lifecycle_release']))
        if not all(proof.get(k) is True for k in ('completed','old_normal_exit_verified','cold_baseline_verified','physical_shielding_confirmed')):
            raise ValueError('normal close/standby evidence required')
        if a.config.resolve()!=ROOT/'CONFIG.json' or os.environ.get('QT_QPA_PLATFORM')!='xcb':
            raise ValueError('sealed new root config and xcb required')
        base=base_module();base.load('e106_no_owner',base.DISPLAY/'run.py').no_owner()
        output=ROOT
    else:
        if not a.output or a.output.resolve().parent!=ROOT or a.output.exists() or a.light_blocked_confirmed or os.environ.get('QT_QPA_PLATFORM')!='offscreen':
            raise ValueError('fresh offscreen output required')
        output=a.output.resolve();output.mkdir()
    run_gui(cfg,output,a.hardware)


if __name__=='__main__':main()
