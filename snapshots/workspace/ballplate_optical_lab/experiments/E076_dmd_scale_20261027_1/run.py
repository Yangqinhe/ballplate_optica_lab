"""Offline-first DMD scale candidate. Existing E075 owner is never modified."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
PRIOR = LAB/'experiments/E075_spatial_repair_20261027_1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(folder):
    if folder.parent != ROOT or folder.exists():
        raise ValueError('fresh experiment child required')
    cfg = read(PRIOR/'CONFIG.json')
    for path, digest in cfg['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('old source changed: '+path)
    folder.mkdir()
    cfg['source_sha256'].update({str(p): sha(p) for p in
                                (ROOT/'run.py', ROOT/'scaling.py')})
    cfg['offline_preparation_only'] = True
    cfg.pop('lifecycle_release', None)
    contract = read(cfg['contract']['path'])
    contract['source_sha256'] = cfg['source_sha256']
    contract['dmd_scale_feature'] = dict(default=1., maximum=1.8,
        manual_only=True, physical_verified=False, control_qualified=False)
    write(folder/'CONTRACT.json', contract)
    cfg['contract'] = dict(path=str(folder/'CONTRACT.json'), sha256=sha(folder/'CONTRACT.json'))
    write(folder/'CONFIG.json', cfg)
    return dict(config=str(folder/'CONFIG.json'), sha256=sha(folder/'CONFIG.json'), hardware_actions=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-offline', type=Path)
    p.add_argument('--offline-smoke', action='store_true')
    p.add_argument('--hardware', action='store_true')
    p.add_argument('--light-blocked-confirmed', action='store_true')
    p.add_argument('--config', type=Path)
    p.add_argument('--config-sha256')
    p.add_argument('--output', type=Path)
    # Activation unavailable until a fresh retained lifecycle is sealed.
    a = p.parse_args()
    if a.prepare_offline:
        print(json.dumps(prepare(a.prepare_offline.resolve()), ensure_ascii=False)); return
    if not (a.offline_smoke or a.hardware):
        print('{"default_IO":0}'); return
    if not a.config or sha(a.config) != a.config_sha256:
        raise ValueError('explicit config SHA required')
    cfg = read(a.config)
    for path, digest in cfg['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('source changed: '+path)
    if a.hardware:
        if (a.offline_smoke or a.output is not None or not a.light_blocked_confirmed
                or a.config != ROOT/'CONFIG.json' or cfg.get('offline_preparation_only') is not False):
            raise ValueError('fresh sealed config and shielding required')
        proof=cfg['lifecycle_release']
        if sha(proof['path']) != proof['sha256'] or not read(proof['path'])['completed']:
            raise ValueError('fresh normal-close and standby lifecycle required')
    repairs = load('e076_preserved_repairs', PRIOR/'run.py')
    repairs.install(cfg)
    scale = load('e076_scaling', ROOT/'scaling.py')
    from optical_lab import preview
    original = preview._load_legacy
    def legacy(path):
        cls = original(path)
        import realtime_gui, diffraction_worker
        scale.install_window(realtime_gui.RealtimeWindow)
        scale.install_reference(diffraction_worker.DiffractionWorker)
        scale.tag_reference_results(diffraction_worker.DiffractionWorker)
        return cls
    preview._load_legacy = legacy
    geometry = preview.attach_manual_geometry_prior
    def attach(window, prior):
        result = geometry(window, prior)
        scale.attach(window)
        if cfg.get('retained_pending_ball_radius') is not None:
            window.ball_size_controls['radius'].setValue(cfg['retained_pending_ball_radius'])
        from PyQt5 import QtCore
        def label():
            window.setWindowTitle('光路工作台 · 八区稳定性／DMD等比缩放'+(' · 真实器件' if a.hardware else ' · 离线检查'))
            window.raw_workbench['note'].setText('DMD1920×1080；手动方图1.00–1.80倍居中。SLM、CCD不自动改变；旧八区协议仍使用1.00倍。')
        def check_manual_page():
            assert not window.hardware and not window.windows and window.worker is None
            try:
                window.tabs.setCurrentIndex(0)
                label()
                window.resize(1500,1000)
                original_settings = window.setting()
                original_roi = dict(window.actual_roi)
                seen = []
                for value in (1.,1.2,1.5,1.8,1.):
                    window.dmd_scale.setValue(value)
                    window.render_patterns()
                    assert window._dmd_side == scale.side_for_scale(value)
                    assert window.setting() == original_settings and window.actual_roi == original_roi
                    seen.append(dict(scale=value,side=window._dmd_side))
                window.dmd_scale.setValue(1.5)
                window.render_patterns()
                window.grab().save(str(a.output/'manual_dmd_scale.png'))
                window.dmd_scale.setValue(1.)
                write(a.output/'DMD_SCALE_SMOKE.json',dict(passed=True,checks=seen,
                    camera_and_slm_unchanged=True,hardware_actions=0))
            except Exception as error:
                write(a.output/'DMD_SCALE_SMOKE.json',dict(passed=False,error=str(error),hardware_actions=0))
        QtCore.QTimer.singleShot(0,label)
        if not a.hardware:
            QtCore.QTimer.singleShot(220,check_manual_page)
        return result
    preview.attach_manual_geometry_prior = attach
    previous = load('e076_1080_composition', repairs.PRIOR/'run.py')
    previous.ROOT = a.config.resolve().parent
    previous.show_outputs = scale.show_outputs
    # Python Qt-slot failures otherwise print a traceback but may leave process
    # exit code0; capture them explicitly in the offline acceptance evidence.
    errors=[]
    original_hook=sys.excepthook
    if not a.hardware:
        def failed_slot(typ,error,tb):
            errors.append(str(error));original_hook(typ,error,tb)
        sys.excepthook=failed_slot
    try:previous.run_gui()
    finally:sys.excepthook=original_hook
    if not a.hardware and (errors or not read(a.output/'DMD_SCALE_SMOKE.json')['passed']):
        raise RuntimeError('manual scaling smoke failed')


if __name__ == '__main__': main()
