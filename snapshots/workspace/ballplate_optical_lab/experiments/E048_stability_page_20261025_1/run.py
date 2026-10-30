"""Fresh-window composition for the one-click stability page; default zero IO.

No live-process patching. Hardware requires a fresh retained E047 owner and
independent shielded normal-exit/cold-camera evidence. No automatic projection.
"""
from contextlib import contextmanager
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

SELF = Path(__file__).resolve(); ROOT = SELF.parent; LAB = ROOT.parents[1]
BASE = LAB/'experiments/E047_region_overlay_20261025_1/run.py'
BASE_SHA = '1331475d0ca4f2121ff0e5c10f7b76ecbcec7b96242f3d0fbdf44da0434d8ddf'
CORE = LAB/'src/optical_lab/stability.py'
GUI = LAB/'src/optical_lab/stability_gui.py'


def pin(path, digest):
    if (not path.is_absolute() or path.is_symlink() or path.resolve() != path
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
        raise ValueError('explicit source/reference SHA differs: '+str(path))
    return path


def base_module():
    spec = importlib.util.spec_from_file_location('E048_frozen_original_window', pin(BASE, BASE_SHA))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare(a):
    base = base_module(); dense = base.dense_module()
    data = dense.validate(dict(path=str(a.contract), sha256=a.contract_sha256),
                          dict(path=str(a.retention), sha256=a.retention_sha256), self_sha256=base.DENSE_SHA)
    for path, digest in ((SELF, a.entry_sha256), (CORE, a.stability_sha256), (GUI, a.stability_gui_sha256)):
        pin(path, digest)
    if not a.output.is_absolute() or a.output.parent != ROOT or a.output.exists():
        raise ValueError('new direct E048 contract directory required')
    contract = data['contract']
    contract.pop('compatibility_validation_only_NOT_launch_contract', None)
    contract['contract_role'] = 'derived_runtime_research_contract'
    contract['lifecycle_release_approved'] = False
    contract['runtime_launch_requires_separate_verified_lifecycle'] = True
    contract['source_sha256'].update({str(SELF): a.entry_sha256, str(CORE): a.stability_sha256, str(GUI): a.stability_gui_sha256,
        str(a.contract): a.contract_sha256})
    contract['stability_page'] = dict(frames_range=[2, 120], repeated_visits=[3, 20], default_frames=30,
        per_test_actual_camera_profile_frozen=True, input_groups_never_pooled=True,
        per_test_duration_max_seconds=300, no_automatic_projection=True, control_qualified=False)
    contract['derived_from_stability_parent'] = dict(path=str(a.contract), sha256=a.contract_sha256)
    a.output.mkdir(); dense.write(a.output/'CONTRACT.json', contract)
    print(json.dumps(dict(path=str(a.output/'CONTRACT.json'), sha256=base.sha(a.output/'CONTRACT.json'), hardware_actions=0)))


def execute(a):
    base = base_module()
    for path, digest in ((SELF, a.entry_sha256), (CORE, a.stability_sha256), (GUI, a.stability_gui_sha256)):
        pin(path, digest)
    contract = json.loads(pin(a.contract, a.contract_sha256).read_text())
    for path, digest in ((SELF, a.entry_sha256), (CORE, a.stability_sha256), (GUI, a.stability_gui_sha256)):
        if contract['source_sha256'].get(str(path)) != digest: raise ValueError('new startup modules missing from source closure')
    if a.hardware:
        retained = json.loads(pin(a.retention, a.retention_sha256).read_text())
        owner = retained['original_owner']
        if (owner.get('pid') != 1856825 or owner.get('start_ticks') != '10067144'
                or owner.get('session') != str(BASE.parent/'GUI')):
            raise ValueError('must retain the actual E047 window; historical E045 lifecycle cannot authorize this switch')
        if a.qa_screenshot: raise ValueError('offline screenshot cannot run in hardware mode')
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) == os.getpid(): continue
            try:
                argv = [s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s]
                cwd = (p/'cwd').resolve()
            except (OSError, UnicodeError): continue
            if '--hardware' in argv and any(s.endswith('.py') and (cwd/s).resolve() == SELF for s in argv):
                raise ValueError('another E048 hardware owner already exists')
    dense = base.dense_module(); original = dense.startup_composition
    errors = []
    @contextmanager
    def compose(exposure, display, attach, contract_ref, spool, initial, callback):
        def attached(window, sampler):
            callback(window, sampler)
            from optical_lab.stability_gui import attach as attach_test
            flow = attach_test(window, contract_ref, ROOT/'TESTS', dict(path=str(base.MASK), sha256=base.MASK_SHA), base.ROI)
            if a.offline_smoke and a.qa_screenshot:
                from PyQt5 import QtCore
                def check():
                    try:
                        window.tabs.setCurrentWidget(flow.widgets['page'])
                        if window.worker is not None or window.windows or flow.active or flow.widgets['permission'].isChecked():
                            raise ValueError('offline page unexpectedly active')
                        flow.widgets['status'].setText('离线界面检查：未连接器件、未采集；表格为空，不是实测结果。')
                        if a.qa_screenshot.exists() or not window.grab().save(str(a.qa_screenshot)):
                            raise ValueError('fresh offline screenshot failed')
                    except Exception as error: errors.append(repr(error))
                QtCore.QTimer.singleShot(100, check)
            if a.hardware:
                dense.write(a.output/'stability_extension.json', dict(owner_pid=os.getpid(),
                    contract_ref=contract_ref, test_root=str(ROOT/'TESTS'), source_sha256=contract['source_sha256'],
                    no_second_camera_owner=True, automatic_capture=False, automatic_projection=False))
        with original(exposure, display, attach, contract_ref, spool, initial, attached): yield
    dense.startup_composition = compose
    base.dense_module = lambda: dense
    base.ROOT = ROOT  # Private startup instance only; frozen source remains unchanged.
    forwarded = argparse.Namespace(**vars(a))
    forwarded.source_sha256 = BASE_SHA
    forwarded.qa_screenshot = None
    forwarded.prepare_contract = False
    result = base.execute(forwarded)
    if errors: raise ValueError('offline stability page check failed: '+repr(errors))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    for flag in ('prepare-contract', 'offline-smoke', 'hardware'): group.add_argument('--'+flag, action='store_true')
    p.add_argument('--light-blocked-confirmed', action='store_true')
    for key in ('entry', 'stability', 'stability-gui', 'overlay'): p.add_argument('--'+key+'-sha256')
    for key in ('contract', 'retention', 'lifecycle', 'qa-frame'):
        p.add_argument('--'+key, type=Path); p.add_argument('--'+key+'-sha256')
    for key in ('output', 'spool', 'qa-screenshot'): p.add_argument('--'+key, type=Path)
    a = p.parse_args(argv)
    if not any((a.prepare_contract, a.offline_smoke, a.hardware)):
        if any(v not in (None, False) for v in vars(a).values()): raise ValueError('default description takes no run arguments')
        print(json.dumps(dict(default_IO=0, SDK_connections=0, projection=False, manual_one_click_only=True))); return 0
    for k in ('entry_sha256', 'stability_sha256', 'stability_gui_sha256', 'contract', 'contract_sha256', 'retention', 'retention_sha256', 'output'):
        if getattr(a, k) is None: raise ValueError('required explicit argument: '+k)
    if a.prepare_contract:
        if a.light_blocked_confirmed or a.lifecycle or a.spool or a.qa_frame or a.qa_screenshot: raise ValueError('file-only preparation only')
        prepare(a); return 0
    if a.spool is None or a.overlay_sha256 is None: raise ValueError('explicit spool and original overlay SHA required')
    return execute(a)


if __name__ == '__main__': raise SystemExit(main())
