"""Two-page composition. Real switching requires sealed normal-exit evidence.

Reuses the explicit frozen E052 camera/display composition, with no changes to
the live source closure. Hardware stays all-zero and idle on startup.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import faulthandler
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
BASE = LAB/'experiments/E052_n60_crash_recovery_20261026_1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--offline-smoke', action='store_true')
    group.add_argument('--hardware', action='store_true')
    parser.add_argument('--light-blocked-confirmed',action='store_true')
    parser.add_argument('--config-sha256')
    parser.add_argument('--retained',action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not (args.offline_smoke or args.hardware):
        print('{"default_IO":0,"hardware_requires_sealed_lifecycle":true}')
        return
    if args.hardware:
        if (not args.light_blocked_confirmed or args.output is not None or
                os.environ.get('QT_QPA_PLATFORM')!='xcb' or Path('/proc/1918336').exists()):
            raise RuntimeError('shielding, xcb and absent exact prior owner required')
        source=ROOT/'CONFIG.json'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=args.config_sha256:
            raise RuntimeError('explicit startup config SHA required')
    elif (os.environ.get('QT_QPA_PLATFORM') != 'offscreen' or args.output is None
            or args.output.parent != ROOT or args.output.exists()):
        raise RuntimeError('fresh direct experiment folder and offscreen required')
    else:source=ROOT/'CONFIG.json'
    if source==BASE/'CONFIG.json' and hashlib.sha256(source.read_bytes()).hexdigest() != 'f0146ecb6a8c0725a5d4a37a5e9ddf087492628baed43ed7d8b4d892abe6f971':
        raise RuntimeError('frozen reference config changed')
    config = json.loads(source.read_text())
    catalog_source = config['catalog_preset']
    catalog = load('state_validation_preflight', ROOT/'catalog.py')
    catalog.validate(catalog_source)
    for path, digest in config['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('frozen reference changed: '+path)
    if args.hardware or args.retained:
        release=config['lifecycle_release'];path=Path(release['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=release['sha256']:
            raise RuntimeError('lifecycle release changed')
        result=json.loads(path.read_text())
        if not all(result.get(k) is True for k in ('completed','old_normal_exit_verified','cold_baseline_verified','physical_shielding_confirmed')):
            raise RuntimeError('lifecycle not verified')
    destination=ROOT if args.hardware else args.output
    if not args.hardware:
        args.output.mkdir()
        (args.output/'CONFIG.json').write_bytes(source.read_bytes())
    base = load('raw_workbench_frozen_base', BASE/'run.py')
    base.ROOT = destination
    from optical_lab import display
    original = display.attach_adaptation_display
    compact = load('raw_workbench_compact', LAB/'experiments/E053_raw_workbench_20261026_1/workbench.py')
    evidence = {}

    def finalize(window):
        result = original(window)
        compact.attach(window)
        catalog = load('state_validation_catalog', ROOT/'catalog.py')
        catalog.attach(window, config['catalog_preset'])
        assert window.tabs.count() == 2 and window.worker is None
        if not args.hardware:assert not window.windows
        assert window.stability_workflow.widgets['result_domain'].currentData() == 'raw'
        for name in ('run_deployment', 'freeze_deployment', 'run_adaptation', 'run_loop', 'queue_push'):
            try:
                getattr(window, name)()
            except RuntimeError as error:
                assert 'legacy task entry removed' in str(error)
            else:
                raise AssertionError('old task still callable: '+name)
        assert window.state_sampling_controller.timer.isActive()
        assert window.stability_workflow.timer.isActive()
        assert not window.stability_workflow.widgets['permission'].isChecked()
        assert window.stability_workflow.widgets['table'].rowCount() == 8
        evidence.update(pages=[window.tabs.tabText(i) for i in range(2)],
                        hidden_pages_retained=len(window.raw_workbench['detached_pages']),
                        old_entries_rejected=5, coordinator_timers_retained=True,
                        eight_regions=True, permission_default=False,
                        camera_owner_created=False, projection=False, control_qualified=False)
        window.stability_workflow.widgets['status'].setText('两页RAW工作台：已保留参数；未自动采集或投图。RAW头尚未适配，闭环未放行。')
        if config['input_sources']['A']==config['input_sources']['B']:
            window.stability_workflow.widgets['status'].setText('已保留设置，但A/B当前选了同一张图；换图测试前请选择不同输入。未自动采集或投图。')
        if args.hardware:
            (ROOT/'GUI/lean_workbench.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
            return result
        from PyQt5 import QtCore
        def screenshot():
            window.grab().save(str(args.output/'manual.png'))
            window.tabs.setCurrentIndex(1)
            window.grab().save(str(args.output/'raw.png'))
            window.tabs.setCurrentIndex(0)
        QtCore.QTimer.singleShot(100, screenshot)
        return result

    display.attach_adaptation_display = finalize
    try:
        faulthandler.enable(all_threads=True)
        base.execute(SimpleNamespace(config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                     hardware=args.hardware, light_blocked_confirmed=args.light_blocked_confirmed))
    finally:
        display.attach_adaptation_display = original
    if not evidence:
        raise RuntimeError('composition was not exercised')
    if not args.hardware:(args.output/'RESULT.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    main()
