"""Fresh-owner composition: unchanged dense sampler plus native-eight overlay.

Default describes only. No GUI inheritance, target injection, camera auto-start
or automatic projection. Hardware requires current shielding and independently
verified normal old-owner exit. Old frozen source files are never modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

SELF = Path(__file__).resolve()
ROOT = SELF.parent
LAB = ROOT.parents[1]
DENSE = LAB/'experiments/E045_dense_state_sampling_20261025_1/run.py'
DENSE_SHA = '45d1bac343890db2bff543e3db00c3941cb9421c410bc0e19c0a76a4becf5028'
OVERLAY = LAB/'src/optical_lab/region_overlay.py'
MASK = LAB/'experiments/E035_mapping_night_20261024_1/REFIT_L00_BALANCED_01/BALANCED_CCD_AXES_NATIVE8.npy'
MASK_SHA = '39badc6336f87025695ac603dfde7eeed50506096b04c230c9c774b5425dd9bf'
ROI = dict(OffsetX=900, OffsetY=416, Width=552, Height=554)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dense_module():
    require(sha(DENSE) == DENSE_SHA, 'original dense launcher changed')
    spec = importlib.util.spec_from_file_location('E047_unchanged_dense_startup', DENSE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_contract(a):
    """File-only extension of a validated retained contract; no hardware release."""
    dense = dense_module()
    source = dict(path=str(a.contract), sha256=a.contract_sha256)
    retention = dict(path=str(a.retention), sha256=a.retention_sha256)
    data = dense.validate(source, retention, self_sha256=DENSE_SHA)
    for path, digest in ((SELF, a.source_sha256), (OVERLAY, a.overlay_sha256)):
        dense.pin(dict(path=str(path), sha256=digest))
    dense.pin(dict(path=str(MASK), sha256=MASK_SHA))
    require(a.output.is_absolute() and a.output.parent == ROOT and not a.output.exists()
            and not a.output.is_symlink(), 'NEW E047 file-only contract directory required')
    # A compatibility-only retained input remains immutable evidence. This new
    # source-bound runtime configuration still does not itself release hardware.
    result = dict(data['contract'], source_sha256=dict(data['contract']['source_sha256']))
    result.pop('compatibility_validation_only_NOT_launch_contract', None)
    result['contract_role'] = 'derived_runtime_research_contract'
    result['lifecycle_release_approved'] = False
    result['runtime_launch_requires_separate_verified_lifecycle'] = True
    result['source_sha256'].update({str(SELF): a.source_sha256, str(OVERLAY): a.overlay_sha256,
                                   str(MASK): MASK_SHA, source['path']: source['sha256']})
    result['display_overlay'] = dict(source=dict(path=str(OVERLAY), sha256=a.overlay_sha256),
        native8=dict(path=str(MASK), sha256=MASK_SHA), roi=ROI, numbering='Region 0–7',
        changes_RAW=False, changes_readout=False, changes_actor=False, default_enabled=False)
    result['prepared_from_retained_contract'] = source
    a.output.mkdir()
    dense.write(a.output/'CONTRACT.json', result)
    dense.write(a.output/'MANIFEST.json', dict(file_sha256={'CONTRACT.json': sha(a.output/'CONTRACT.json')},
        original_sources_unchanged=True, hardware_actions=0, lifecycle_release_approved=False))
    print(json.dumps(dict(path=str(a.output/'CONTRACT.json'), sha256=sha(a.output/'CONTRACT.json'), hardware_actions=0)))
    return 0


@contextmanager
def overlay_composition(display, layout, callback=None):
    """Local startup factory composition; restored on all exit paths."""
    from optical_lab.region_overlay import attach
    original = display.attach_manual_display
    def factory(window, initial=None):
        panel = original(window, initial)
        controller = attach(window, layout)
        if callback is not None:
            callback(window, controller)
        return panel
    display.attach_manual_display = factory
    try:
        yield
    finally:
        display.attach_manual_display = original


def execute(a):
    dense = dense_module()
    contract_ref = dict(path=str(a.contract), sha256=a.contract_sha256)
    retention_ref = dict(path=str(a.retention), sha256=a.retention_sha256)
    data = dense.validate(contract_ref, retention_ref, self_sha256=DENSE_SHA)
    sources = data['contract']['source_sha256']
    for path, digest in ((SELF, a.source_sha256), (OVERLAY, a.overlay_sha256), (MASK, MASK_SHA)):
        require(digest is not None and sources.get(str(path)) == digest == sha(path),
                'new overlay source/mask must be in exact runtime closure: '+str(path))
    require(data['contract']['fixed_context']['roi'] == ROI, 'native eight ROI differs')
    for path in (a.output, a.spool):
        require(path.is_absolute() and path.parent == ROOT and path.resolve() == path
                and not path.exists() and not path.is_symlink(), 'new direct E047 output/spool required')
    require(a.output != a.spool, 'separate owner and archive directories required')
    if a.hardware:
        require(a.light_blocked_confirmed and a.lifecycle and a.lifecycle_sha256 and not a.qa_frame
                and not a.qa_screenshot, 'fresh shielded lifecycle required, no hardware fixture')
        dense.validate_lifecycle(dict(path=str(a.lifecycle), sha256=a.lifecycle_sha256), retention_ref, data['retained'])
        dense.no_other_owner()
        # The original launcher also detects original GUI entrypoints. Include
        # this new wrapper explicitly so it cannot start a second E047 owner.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                argv = [v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
                cwd = (proc/'cwd').resolve()
            except (OSError, UnicodeError):
                continue
            require(not ('--hardware' in argv and any(v.endswith('.py') and (cwd/v).resolve() == SELF for v in argv)),
                    'another native-eight owner already exists')
    else:
        require(a.offline_smoke and not a.light_blocked_confirmed and not a.lifecycle,
                'offline mode cannot claim hardware lifecycle')
        require(os.environ.get('QT_QPA_PLATFORM') == 'offscreen', 'offline test must use offscreen Qt')
    require(bool(a.qa_frame) == bool(a.qa_frame_sha256), 'offline frame and SHA must be paired')
    if a.qa_screenshot:
        require(a.offline_smoke and a.qa_frame and a.qa_frame_sha256 and not a.qa_screenshot.exists(),
                'new offline screenshot and explicit frozen frame required')
    sys.path.insert(0, str(LAB/'src'))
    from optical_lab import preview, exposure, display
    from optical_lab.region_overlay import load_layout
    from optical_lab.state_sampling_gui import attach
    layout = load_layout(MASK, MASK_SHA, ROI)
    retained = data['retained']
    command = ['--legacy-root', str(dense.LEGACY), '--output', str(a.output)]
    for flag, key in (('restart-state', 'restart_contract'), ('recovery-record', 'recovery_record'),
                      ('display-state', 'display_state'), ('manual-geometry-state', 'manual_geometry'),
                      ('preview-caches', 'preview_caches')):
        if flag == 'recovery-record' and a.offline_smoke and a.qa_frame:
            # An explicitly offline historical image has no camera to recover.
            # Hardware always receives its original recovery record unchanged.
            continue
        command += ['--'+flag, retained[key]['path']]
        if flag not in ('restart-state', 'recovery-record'):
            command += ['--'+flag.replace('-state', '')+'-sha256' if flag == 'manual-geometry-state'
                        else '--'+flag+'-sha256', retained[key]['sha256']]
    command += ['--hardware', '--light-blocked-confirmed'] if a.hardware else ['--smoke-test']
    if a.qa_frame:
        command += ['--preview-frame', str(a.qa_frame), '--preview-frame-sha256', a.qa_frame_sha256]
    def sampler_attached(window, controller):
        if a.hardware:
            dense.write(a.output/'dense_sampling_extension.json', dict(
                schema='E047-original-dense-plus-display-overlay-1', owner_pid=os.getpid(),
                owner_start_ticks=controller.owner_start, contract_ref=contract_ref,
                retention_ref=retention_ref, sampler_spool=str(a.spool), source_sha256=sources,
                native8_mask=dict(path=str(MASK), sha256=MASK_SHA), native8_roi=ROI,
                optical_or_numeric_processing_changed=False, control_release_approved=False,
                physical_shielding_confirmed=True, automatic_capture=False, automatic_projection=False))
    qa_errors = []
    def overlay_attached(window, controller):
        if a.qa_screenshot:
            from PyQt5 import QtCore
            def save():
                try:
                    # Fixture is explicitly a historical CCD image, never live.
                    window.manual_display.offline['active'] = dict(ROI)
                    window.manual_display.dirty = True
                    window.manual_display.refresh()
                    controller.checkbox.setChecked(True); controller.refresh()
                    require(controller.last_error is None and not controller.overlay.isHidden(),
                            'offline overlay not visible: '+str(controller.last_error))
                    require(not window.windows and window.worker is None and not a.output.exists(), 'offline UI created hardware state')
                    require(window.grab().save(str(a.qa_screenshot)), 'offline screenshot failed')
                except Exception as error:
                    qa_errors.append(repr(error))
            QtCore.QTimer.singleShot(100, save)
    with overlay_composition(display, layout, overlay_attached):
        with dense.startup_composition(exposure, display, attach, contract_ref, a.spool,
                                       data['display'], sampler_attached):
            result = preview.main(command)
    require(not qa_errors, 'offline UI validation failed: '+repr(qa_errors))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--hardware', action='store_true'); group.add_argument('--offline-smoke', action='store_true')
    group.add_argument('--prepare-contract', action='store_true')
    p.add_argument('--light-blocked-confirmed', action='store_true')
    p.add_argument('--source-sha256'); p.add_argument('--overlay-sha256')
    for name in ('contract', 'retention', 'lifecycle'):
        p.add_argument('--'+name, type=Path); p.add_argument('--'+name+'-sha256')
    for name in ('output', 'spool', 'qa-frame', 'qa-screenshot'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--qa-frame-sha256')
    a = p.parse_args(argv)
    if a.prepare_contract:
        require(all(getattr(a, key) is not None for key in ('source_sha256', 'overlay_sha256', 'contract',
            'contract_sha256', 'retention', 'retention_sha256', 'output')) and not a.light_blocked_confirmed
            and not a.lifecycle and not a.lifecycle_sha256 and not a.spool and not a.qa_frame
            and not a.qa_frame_sha256 and not a.qa_screenshot, 'explicit file-only contract preparation arguments required')
        return prepare_contract(a)
    if not a.hardware and not a.offline_smoke:
        require(all(v is None or v is False for v in vars(a).values()), 'default description takes no run arguments')
        print(json.dumps(dict(default_IO=0, SDK_actions=0, projection=False,
            native8_overlay_default=False, raw_and_control_unchanged=True, original_dense_sampler_preserved=True)))
        return 0
    require(all(getattr(a, key) is not None for key in ('source_sha256', 'overlay_sha256', 'contract',
        'contract_sha256', 'retention', 'retention_sha256', 'output', 'spool')), 'explicit source/contract/retention/new outputs required')
    return execute(a)


if __name__ == '__main__':
    raise SystemExit(main())
