"""File-only preparation and fresh-process integration of spatial repairs.

Hardware activation requires a new retained restart contract and current physical
shielding, not a copy of an earlier approval. Existing E074 sources stay unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parents[1]
PRIOR=LAB/'experiments/E074_dmd1920_rollback_20261026_1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    if '.' in name:
        parent,child=name.rsplit('.',1);setattr(sys.modules[parent],child,module)
    spec.loader.exec_module(module);return module

def prepare(folder):
    if folder.parent!=ROOT or folder.exists():raise ValueError('fresh experiment child required')
    cfg=read(PRIOR/'CONFIG.json')
    for p,d in cfg['source_sha256'].items():
        if sha(p)!=d:raise ValueError('original frozen source changed: '+p)
    folder.mkdir()
    sources=dict(cfg['source_sha256'])
    for p in (Path(__file__),ROOT/'alignment.py',ROOT/'staged/optical_lab/display.py'):
        sources[str(p)]=sha(p)
    cfg.update(source_sha256=sources,offline_preparation_only=True)
    # Do not carry an old safe-exit approval into a future hardware session.
    cfg.pop('lifecycle_release',None)
    contract=read(cfg['contract']['path']);contract['source_sha256']=sources
    contract['spatial_repair']=dict(display_only_resampling=True,raw_processing_changed=False,
        actual_display_ack=True,calibration_clipping_rejected=True,hardware_verified=False)
    write(folder/'CONTRACT.json',contract);cfg['contract']=ref(folder/'CONTRACT.json')
    write(folder/'CONFIG.json',cfg)
    return dict(config=ref(folder/'CONFIG.json'),hardware_actions=0)

def install(cfg):
    import optical_lab
    alignment=load('optical_lab.alignment',ROOT/'alignment.py')
    load('optical_lab.display',ROOT/'staged/optical_lab/display.py')
    from optical_lab import exposure,preview
    alignment.install_worker_factory(exposure)
    original=preview._load_legacy
    def legacy(path):
        cls=original(path)
        import realtime_gui,diffraction_worker
        alignment.install_display_transactions(realtime_gui.RealtimeWindow)
        diffraction_worker.fit_zero_mapping=alignment.support_aware_fit(diffraction_worker.fit_zero_mapping)
        return cls
    preview._load_legacy=legacy
    geometry=preview.attach_manual_geometry_prior
    def restore(window,prior):
        result=geometry(window,prior)
        state=cfg.get('retained_manual_controls')
        if state:
            if window.worker is not None or window.running or window.armed.isChecked():
                raise RuntimeError('retained controls only at stopped fresh startup')
            for widget,key in ((window.dmd_mode,'dmd'),(window.slm_mode,'slm')):
                index=widget.findData(state['settings'][key])
                if index<0:raise ValueError('missing retained pattern mode')
                widget.setCurrentIndex(index)
            window.orientation.setCurrentIndex(state['orientation'])
            controller=window.manual_display.controls
            # Restore preview filter from the visible controls, not old events.
            controller.filter_mode.setCurrentIndex(controller.filter_mode.findData(state['filter']['mode']))
            controller.filter_alpha.setValue(state['filter']['alpha'])
            controller.filter_original.setChecked(state['filter']['original'])
            window.changed()
        return result
    preview.attach_manual_geometry_prior=restore

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-offline',type=Path)
    p.add_argument('--offline-smoke',action='store_true');p.add_argument('--config',type=Path)
    p.add_argument('--hardware',action='store_true');p.add_argument('--light-blocked-confirmed',action='store_true')
    p.add_argument('--config-sha256');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.prepare_offline:
        print(json.dumps(prepare(a.prepare_offline.resolve()),ensure_ascii=False));return
    if not (a.offline_smoke or a.hardware):print('{"default_IO":0}');return
    if not a.config or sha(a.config)!=a.config_sha256:raise ValueError('explicit config SHA required')
    cfg=read(a.config)
    for path,digest in cfg['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('source changed: '+path)
    if a.hardware:
        proof=cfg.get('lifecycle_release',{})
        if a.offline_smoke or not a.light_blocked_confirmed or a.config!=ROOT/'CONFIG.json' or cfg.get('offline_preparation_only') is not False:
            raise ValueError('fresh sealed hardware config and shielding required')
        if sha(proof['path'])!=proof['sha256'] or not read(proof['path'])['completed']:
            raise ValueError('new lifecycle proof required')
    install(cfg)
    previous=load('e075_1080_composition',PRIOR/'run.py')
    previous.ROOT=a.config.resolve().parent
    previous.run_gui()

if __name__=='__main__':main()
