"""Offline launcher guards: no real Qt, SDK, process control or projection."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('E045_run_tested',HERE/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)


class Scalar:
    def __init__(self):self.value=None
    def setValue(self,value):self.value=value
    def setChecked(self,value):self.value=value
    def setCurrentIndex(self,value):self.value=value
    def findData(self,value):return {'ema':2}.get(value,-1)


class LauncherTest(unittest.TestCase):
    def test_description_does_not_read(self):
        with patch.object(Path,'read_bytes',side_effect=AssertionError('read')),patch.object(Path,'read_text',side_effect=AssertionError('read')):
            self.assertEqual(run.description()['default_file_UI_SDK_IO'],0)
            self.assertEqual(run.main([]),0)

    def test_no_execution_fields_in_default(self):
        with self.assertRaises(ValueError):run.main(['--light-blocked-confirmed'])

    def test_hardware_requires_explicit_references(self):
        with self.assertRaises(ValueError):run.main(['--hardware'])

    def test_pin_rejects_wrong_SHA_and_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'a';path.write_bytes(b'a');link=Path(temp)/'b';link.symlink_to(path)
            self.assertEqual(run.pin(dict(path=str(path),sha256=run.sha(path))),path)
            with self.assertRaises(ValueError):run.pin(dict(path=str(path),sha256='0'*64))
            with self.assertRaises(ValueError):run.pin(dict(path=str(link),sha256=run.sha(path)))

    def test_lifecycle_rejects_planned_or_wrong_owner(self):
        value=dict(schema='E045-normal-shielded-release-1',physical_shielding_confirmed=True,
            old_normal_exit_verified=False,cold_baseline_verified=True,forced_kill=False,
            camera_readonly_connections=2,camera_readonly_snapshots=10,retention_ref={'ref':1})
        with patch.object(run,'read_ref',return_value=value):
            with self.assertRaises(ValueError):run.validate_lifecycle({}, {'ref':1},{})
        value.update(old_normal_exit_verified=True,old_owner={},parameter_writes=0,acquisition_started=False,errors=[])
        with patch.object(run,'read_ref',return_value=value):
            with self.assertRaises(ValueError):run.validate_lifecycle({}, {'ref':1},dict(original_owner=dict(pid=1,start_ticks='2')))

    def test_owner_module_absolute_and_relative_launches(self):
        for command,cwd in ((['python','-m','optical_lab.preview','--hardware'],run.LAB),
            (['python','-m','live_loop_v23.direct_alignment_gui','--hardware'],run.LEGACY),
            (['python',str(run.SELF),'--hardware'],run.LAB),
            (['python','-B','run.py','--hardware'],run.ROOT),
            (['python','./run.py','--hardware'],run.ROOT),
            (['python','experiments/'+run.ROOT.name+'/run.py','--hardware'],run.LAB),
            (['python','live_loop_v23/direct_alignment_gui.py','--hardware'],run.LEGACY)):
            self.assertTrue(run.owner_argv(command,cwd),command)
        self.assertFalse(run.owner_argv(['python','run.py','--hardware'],Path('/tmp')))
        self.assertFalse(run.owner_argv(['python',str(run.SELF),'--offline-smoke'],run.LAB))

    def test_startup_hooks_once_restore_and_reset_preview_only(self):
        calls=[];panel=SimpleNamespace(controls=SimpleNamespace(filter_mode=Scalar(),filter_alpha=Scalar(),filter_original=Scalar()),
            temporal=SimpleNamespace(reset=lambda:calls.append('reset')))
        original=lambda window:calls.append('original')
        original_display=lambda w,initial:panel
        exposure=SimpleNamespace(attach_exposure_warnings=original)
        display=SimpleNamespace(attach_manual_display=original_display)
        attach=lambda *a:calls.append('attach') or 'controller'
        with run.startup_composition(exposure,display,attach,{},Path('/unused'),
            dict(preview_filter=dict(mode='ema',alpha=.1,original=False)),lambda *a:calls.append('record')):
            exposure.attach_exposure_warnings(object())
            with self.assertRaises(ValueError):exposure.attach_exposure_warnings(object())
            self.assertIs(display.attach_manual_display(object()),panel)
            self.assertEqual(calls,['original','attach','record','reset'])
            self.assertEqual(panel.controls.filter_alpha.value,.1)
        self.assertIs(exposure.attach_exposure_warnings,original)
        self.assertIs(display.attach_manual_display,original_display)

    def test_hooks_restored_on_exception(self):
        original=lambda w:None;display_original=lambda w,i:None
        e=SimpleNamespace(attach_exposure_warnings=original);d=SimpleNamespace(attach_manual_display=display_original)
        with self.assertRaises(RuntimeError):
            with run.startup_composition(e,d,None,{},None,{},None):raise RuntimeError('fixture')
        self.assertIs(e.attach_exposure_warnings,original);self.assertIs(d.attach_manual_display,display_original)


if __name__=='__main__':unittest.main()
