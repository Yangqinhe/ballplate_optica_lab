"""Synthetic file-only retention tests; no live owner, Qt, SDK or devices."""
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('E045_retention_tested',HERE/'retention.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ORIGINAL_HELPER=m._legacy_helper


class TestRetention(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.session=self.root/'OLD/GUI';self.session.mkdir(parents=True)
        self.legacy=self.root/'LEGACY';self.legacy.mkdir();self.new=self.root/'E045';self.new.mkdir()
        self.proc=self.root/'PROC';p=self.proc/str(m.EXPECTED_PID);p.mkdir(parents=True)
        self.paths=dict(ROOT=self.new,SELF=self.new/'retention.py',OLD=self.root/'OLD',SESSION=self.session,
            PROC=self.proc,LEGACY=self.legacy,CONTRACT=self.root/'OLD/restart_contract.json',
            RECOVERY=self.root/'OLD/recovery_record.json',PROFILE=self.root/'PROFILE.json',
            PREVIEW_RESTART=self.root/'preview_restart.py',DMD=self.root/'D12.npy',SLM=self.root/'L00.npy',
            MANUAL=self.root/'E029_prior.json')
        self.patcher=patch.multiple(m,**self.paths);self.patcher.start();self.addCleanup(self.patcher.stop)
        for name,path in self.paths.items():
            if name in ('SELF','PREVIEW_RESTART','DMD','SLM'):path.write_bytes((name+' source').encode())
        (self.new/'test_retention.py').write_text('# fixture tests\n')
        basefile=self.root/'baseline.json'
        self.baseline=dict(DeviceSerialNumber='15328133',ExposureTime='67',Gain='0',TriggerMode='Off')
        self.jwrite(basefile,dict(target=self.baseline))
        self.settings=dict(dmd='file',slm='file',square=128,full_panel=False,x=-410,y=4,focal_mm=400.,period=35)
        self.active=dict(DeviceSerialNumber='15328133',ExposureTime='74',Gain='10.88',TriggerMode='On')
        self.roi=dict(OffsetX=900,OffsetY=416,Width=552,Height=554);self.optical={'f1_m':.15,'f2_m':.1}
        self.profile=dict(active=self.active,settings=self.settings,actual_roi=self.roi,optical_reference=self.optical,geometry_frozen=False)
        self.jwrite(m.PROFILE,self.profile)
        self.old=dict(schema='v20-retained-line0-v1',original_fault_before=self.baseline,registration=None,expected_active=None,
            evidence_sha256={str(basefile):m.sha(basefile.read_bytes())},display_roles={'dmd':{'id':'DMD'},'slm':{'id':'SLM'}})
        self.jwrite(m.CONTRACT,self.old)
        self.recovery=dict(camera_baseline=self.baseline,baseline_source=str(basefile),baseline_source_sha256=m.sha(basefile.read_bytes()))
        self.jwrite(m.RECOVERY,self.recovery)
        self.runtime=dict(pid=m.EXPECTED_PID,sources={str(m.PREVIEW_RESTART):m.sha(m.PREVIEW_RESTART.read_bytes())},source_configs={})
        self.jwrite(self.session/'runtime_sources.json',self.runtime)
        self.jwrite(self.session/'spatial_preview_extension.json',{'new_source_sha256':{}})
        reference=self.root/'REFERENCE';reference.mkdir();self.jwrite(reference/'capture.json',{'active':'original'})
        self.jwrite(m.MANUAL,dict(source_reference_folder=str(reference),source_reference_sha256={'capture.json':m.sha((reference/'capture.json').read_bytes())}))
        hashes=dict(CONTRACT_SHA=m.sha(m.CONTRACT.read_bytes()),RECOVERY_SHA=m.sha(m.RECOVERY.read_bytes()),
            PROFILE_SHA=m.sha(m.PROFILE.read_bytes()),PREVIEW_RESTART_SHA=m.sha(m.PREVIEW_RESTART.read_bytes()),
            RUNTIME_SHA=m.sha((self.session/'runtime_sources.json').read_bytes()),
            EXTENSION_SHA=m.sha((self.session/'spatial_preview_extension.json').read_bytes()),
            DMD_SHA=m.sha(m.DMD.read_bytes()),SLM_SHA=m.sha(m.SLM.read_bytes()),MANUAL_SHA=m.sha(m.MANUAL.read_bytes()))
        self.hashpatch=patch.multiple(m,**hashes);self.hashpatch.start();self.addCleanup(self.hashpatch.stop)
        self.argv=['/fake/python','-m','optical_lab.preview','--hardware','--legacy-root',str(m.LEGACY),
            '--output',str(m.SESSION),'--restart-state',str(m.CONTRACT),'--recovery-record',str(m.RECOVERY),
            '--manual-geometry-state',str(m.MANUAL),'--manual-geometry-sha256',m.MANUAL_SHA]
        for flag in ('display-state','preview-caches'):
            path=self.root/(flag+'.json')
            value={'startup':flag} if flag=='display-state' else dict(schema='optical-lab-preview-cache-1',
                files={'dmd':{'path':str(m.DMD),'sha256':m.DMD_SHA},'slm':{'path':str(m.SLM),'sha256':m.SLM_SHA}})
            self.jwrite(path,value)
            self.argv+=['--'+flag,str(path),'--'+flag+'-sha256',m.sha(path.read_bytes())]
        self.cmd='\0'.join(self.argv).encode()+b'\0';(p/'cmdline').write_bytes(self.cmd)
        stat=['S']+['0']*19;stat[19]=m.EXPECTED_START;(p/'stat').write_text('1390768 (owner) '+' '.join(stat))
        (p/'cwd').symlink_to(self.root,target_is_directory=True)
        self.now=100_000_000_000
        self.h=dict(pid=m.EXPECTED_PID,unix_ns=self.now-1_000_000,deployment_ui_revision=23,hardware_fault=None,
            projection_armed=False,loop_active=False,adaptation_active=False,deployment_busy=False,closed_loop_enabled=False,
            busy=True,live_camera=True,display_revision=9,last_verified_revision=9,camera_active_readback=self.active,
            manual_settings=self.settings,actual_roi=self.roi,optical_reference=self.optical,geometry_frozen=False)
        self.jwrite(self.session/'heartbeat.json',self.h)
        self.events=[dict(state='created',video=self.old['display_roles']),
            dict(state='manual_file_loaded',path=str(m.SLM),sha256=m.SLM_SHA,SLM=True),
            dict(state='manual_file_loaded',path=str(m.DMD),sha256=m.DMD_SHA,SLM=False),
            dict(state='manual_display_controls',controls={'mode':'fixed','gain':1.,'black':12.5,'gamma':.4999999999999998},
                display_only=True,raw_or_actor_modified=False),
            dict(state='manual_preview_filter_controls',controls={'mode':'ema','alpha':.1,'original':False,'generation':7},
                display_only=True,raw_or_actor_modified=False),
            dict(state='manual_display_requested',revision=9,dmd_nonzero=False,slm_nonzero=False,settings=self.settings)]
        self.write_events()
        self.validations=[]
        helper=SimpleNamespace(validate_manual_geometry_sidecar=lambda *args:None,
            _validate=lambda c,*_:self.validations.append(deepcopy(c)))
        self.helpers=patch.object(m,'_legacy_helper',return_value=helper);self.helpers.start();self.addCleanup(self.helpers.stop)
        self.clock=patch.object(m.time,'time_ns',return_value=self.now);self.clock.start();self.addCleanup(self.clock.stop)

    @staticmethod
    def jwrite(path,value):Path(path).write_text(json.dumps(value))
    def write_events(self):
        (self.session/'events.jsonl').write_bytes(b''.join(json.dumps(e).encode()+b'\n' for e in self.events))
    def prepare(self):return m.prepare(self.new/'RETENTION_01',source_sha256=m.sha(m.SELF.read_bytes()),execute_file_only=True)

    def test_default_is_stdlib_describe_and_reads_nothing(self):
        code="import runpy,sys;x=runpy.run_path(sys.argv[1],run_name='inspect');x['main']([]);assert not any(k.startswith(('numpy','PyQt','PySpin','mujoco')) for k in sys.modules)"
        completed=subprocess.run([sys.executable,'-B','-c',code,str(HERE/'retention.py')],capture_output=True,text=True,check=True)
        self.assertFalse(json.loads(completed.stdout)['lifecycle_release_approved'])
        with patch.object(m,'pin',side_effect=AssertionError('must not read')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main([]),0)
        with patch.object(m,'pin',side_effect=AssertionError('must not read')):
            with self.assertRaises(ValueError):m.prepare(self.new/'unused',source_sha256='x')

    def test_pure_legacy_helper_load_cannot_write_old_bytecode(self):
        m.PREVIEW_RESTART.write_text('import sys\nbytecode_disabled_during_import=sys.dont_write_bytecode\n')
        prior=sys.dont_write_bytecode
        helper=ORIGINAL_HELPER()
        self.assertTrue(helper.bytecode_disabled_during_import)
        self.assertEqual(sys.dont_write_bytecode,prior)
        self.assertFalse((m.PREVIEW_RESTART.parent/'__pycache__').exists())

    def test_live_camera_allowed_but_all_actions_and_faults_rejected(self):
        m.heartbeat_guard(self.h,self.runtime,self.profile,m.EXPECTED_START,self.now)
        for delta in ({'projection_armed':True},{'hardware_fault':'fault'},{'loop_active':True},
            {'deployment_busy':True},{'adaptation_active':True},{'last_verified_revision':8},
            {'pid':1},{'geometry_frozen':True},{'camera_active_readback':self.baseline},{'manual_settings':{}}):
            with self.subTest(delta=delta),self.assertRaises(ValueError):
                m.heartbeat_guard(dict(self.h,**delta),self.runtime,self.profile,m.EXPECTED_START,self.now)
        with self.assertRaises(ValueError):m.heartbeat_guard(self.h,self.runtime,self.profile,'wrong',self.now)
        with self.assertRaises(ValueError):m.heartbeat_guard(self.h,self.runtime,self.profile,m.EXPECTED_START,self.now+3_000_000_000)

    def test_fold_latest_caches_and_exact_display_event_line_SHA(self):
        old=deepcopy(self.events[2]);old.update(path='/older.npy',sha256='0'*64)
        blob=b''.join(json.dumps(e).encode()+b'\n' for e in [old,*self.events])
        fold=m.fold_events(blob);self.assertEqual(fold['dmd']['event']['path'],str(m.DMD))
        self.assertEqual(fold['display']['line_number'],5)
        self.assertEqual(fold['display']['line_sha256'],m.sha(blob.splitlines()[4]))
        state=m.display_state(fold,self.active)
        self.assertEqual(state['manual']['gamma'],.4999999999999998)
        self.assertEqual(state['preview_filter']['alpha'],.1);self.assertFalse(state['hardware_authorization'])
        with self.assertRaises(ValueError):m.fold_events(blob[:-1])

    def test_owner_argv_must_keep_explicit_original_geometry_and_session(self):
        args,_=m.owner_arguments(self.cmd,self.root);self.assertEqual(args,self.argv)
        bad=self.argv.copy();bad[bad.index('--manual-geometry-sha256')+1]='0'*64
        with self.assertRaises(ValueError):m.owner_arguments('\0'.join(bad).encode(),self.root)
        with self.assertRaises(ValueError):m.owner_arguments(self.cmd+b'--output\0/new\0',self.root)

    def test_prepared_contract_is_planned_black_zero_baseline_not_live_active(self):
        result=self.prepare();output=self.new/'RETENTION_01';c=m.read(output/'restart_contract.json')
        self.assertEqual(c['original_session'],str(self.session));self.assertEqual(c['original_pid'],m.EXPECTED_PID)
        self.assertEqual(c['runtime_path'],str(self.session/'runtime_sources.json'))
        self.assertEqual(c['original_fault_before'],self.baseline)
        self.assertNotEqual(c['original_fault_before'],self.active)
        self.assertIsNone(c['original_last_camera_report']);self.assertIsNone(c['registration']);self.assertIsNone(c['expected_active'])
        self.assertEqual(c['requested_exposure_us'],80.);self.assertEqual(c['settings'],dict(self.settings,dmd='black',slm='zero'))
        self.assertFalse(c['release_approved']);self.assertFalse(result['lifecycle_release_approved'])
        r=m.read(output/'recovery_record.json');self.assertEqual(r['camera_baseline'],self.baseline)
        self.assertFalse(r['cold_baseline_readonly_verification_passed'])
        self.assertTrue(m.read(output/'baseline_evidence.json')['derived_evidence_NOT_new_measurement'])
        self.assertEqual(m.read(output/'PREVIEW_CACHES.json')['files']['dmd']['path'],str(m.DMD))
        self.assertEqual(m.read(output/'RESULT.json')['manual_geometry']['path'],str(m.MANUAL))
        for c in self.validations:
            self.assertEqual(c['original_fault_before'],self.baseline)
            self.assertIsNone(c.get('original_last_camera_report'))
        manifest=m.read(output/'MANIFEST.json')
        self.assertTrue(all(m.sha((output/p).read_bytes())==v for p,v in manifest['file_sha256'].items()))
        self.assertTrue((self.proc/str(m.EXPECTED_PID)).exists())

    def test_wrong_cache_zero_request_source_and_existing_output_fail_closed(self):
        old=self.events[2]['sha256'];self.events[2]['sha256']='0'*64;self.write_events()
        with self.assertRaises(ValueError):self.prepare()
        self.events[2]['sha256']=old;self.events[-1]['dmd_nonzero']=True;self.write_events()
        with self.assertRaises(ValueError):self.prepare()
        self.events[-1]['dmd_nonzero']=False;self.write_events()
        with self.assertRaises(ValueError):m.prepare(self.new/'NEW',source_sha256='0'*64,execute_file_only=True)
        self.prepare()
        with self.assertRaises(ValueError):self.prepare()

    def test_final_owner_change_preserves_failed_prefix_and_no_success_claim(self):
        original=m.heartbeat_guard;calls=[]
        def guard(*args):
            calls.append(1)
            if len(calls)==2:raise ValueError('owner changed')
            return original(*args)
        with patch.object(m,'heartbeat_guard',side_effect=guard),self.assertRaises(ValueError):self.prepare()
        output=self.new/'RETENTION_01'
        self.assertTrue((output/'FAILED.json').is_file());self.assertFalse((output/'RESULT.json').exists())
        self.assertEqual(m.read(output/'FAILED.json')['hardware_actions'],0)


if __name__=='__main__':unittest.main()
