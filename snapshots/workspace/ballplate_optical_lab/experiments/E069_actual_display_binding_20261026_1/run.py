"""Two-page fresh crash recovery; no output/SDK work by default.

Composes the frozen E052 entry directly, not another GUI inheritance layer.
Keeps E068 sampling order; corrects the actual Bench display module binding.
"""
import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from recover import ROOT, LAB, OLD, BASE, no_owner, read, sha, write
from display_binding import bound_modulators, verify_instances

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group();g.add_argument('--hardware',action='store_true');g.add_argument('--offline-smoke',action='store_true')
    p.add_argument('--light-blocked-confirmed',action='store_true');p.add_argument('--config-sha256');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not(a.hardware or a.offline_smoke):print('{"default_IO":0}');return
    if a.hardware:
        no_owner()
        if not a.light_blocked_confirmed or a.output is not None or os.environ.get('QT_QPA_PLATFORM')!='xcb':
            raise RuntimeError('shielded xcb one-owner startup required')
    elif (os.environ.get('QT_QPA_PLATFORM')!='offscreen' or a.light_blocked_confirmed or
          a.output is None or a.output.parent!=ROOT or a.output.exists()):
        raise RuntimeError('fresh direct folder and offscreen required')
    config_path=ROOT/'CONFIG.json'
    if a.hardware and sha(config_path)!=a.config_sha256:raise RuntimeError('explicit config SHA required')
    cfg=read(config_path)
    for path,digest in cfg['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('source changed: '+path)
    crash=cfg['crash_recovery']
    if sha(crash['path'])!=crash['sha256'] or read(crash['path'])['old_normal_exit'] is not False:
        raise RuntimeError('preserved explicit crash evidence required')
    destination=ROOT if a.hardware else a.output
    if not a.hardware:
        destination.mkdir();(destination/'CONFIG.json').write_bytes(config_path.read_bytes())
    base=load('display_recovery_frozen_base',BASE/'run.py');base.ROOT=destination
    import optical_lab
    base.load('optical_lab.state_sampling_gui',OLD/'adapter.py')
    from optical_lab import display,queued_display
    compact=load('compact_workbench',LAB/'experiments/E053_raw_workbench_20261026_1/workbench.py')
    catalog=load('input_catalog',LAB/'experiments/E054_state_validation_20261026_1/catalog.py')
    catalog.validate(cfg['catalog_preset'])
    old_context=queued_display.queued_modulator;old_attach=display.attach_adaptation_display
    binding={};result={}
    source=base.MODULATOR
    def actual_context(module):
        return bound_modulators(module,old_context,source,cfg['source_sha256'][str(source)],binding)
    def finalize(window):
        attached=old_attach(window)
        compact.attach(window);catalog.attach(window,cfg['catalog_preset'])
        assert window.tabs.count()==2 and window.worker is None
        assert not window.armed.isChecked()
        assert not window.stability_workflow.widgets['permission'].isChecked()
        assert window.stability_workflow.widgets['result_domain'].currentData()=='raw'
        display_evidence=verify_instances(window,binding)
        result.update(pages=[window.tabs.tabText(i) for i in range(2)],
            actual_display_binding=display_evidence,automatic_capture=False,automatic_projection=False,
            old_normal_exit=False,control_qualified=False)
        window.stability_workflow.widgets['status'].setText('已修正实际器件窗口的排队重绘；保留采样时序修正。旧崩溃批次无效，未自动采集或投图。')
        if a.hardware:write(ROOT/'GUI/display_binding_verified.json',result)
        else:
            from PyQt5 import QtCore
            def screenshot():
                window.grab().save(str(destination/'manual.png'))
                window.tabs.setCurrentIndex(1);window.grab().save(str(destination/'raw.png'))
            QtCore.QTimer.singleShot(100,screenshot)
        return attached
    queued_display.queued_modulator=actual_context
    display.attach_adaptation_display=finalize
    try:
        faulthandler.enable(all_threads=True)
        base.execute(SimpleNamespace(config_sha256=sha(config_path),hardware=a.hardware,
                                     light_blocked_confirmed=a.light_blocked_confirmed))
    finally:
        queued_display.queued_modulator=old_context;display.attach_adaptation_display=old_attach
    if not result:raise RuntimeError('startup composition not exercised')
    if not a.hardware:write(destination/'RESULT.json',result)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
