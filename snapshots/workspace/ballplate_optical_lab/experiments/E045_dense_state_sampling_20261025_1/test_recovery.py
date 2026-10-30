"""Pure mocked recovery branch tests; never access a real UI or SDK."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('E045_test_recovery',HERE/'recovery.py')
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.m=r.engine();self.session=self.root/'GUI';self.session.mkdir();self.proc=self.root/'proc';self.proc.mkdir()
        self.owner=self.proc/str(self.m.PID);self.owner.mkdir();(self.owner/'cmdline').write_bytes(b'python\0fake\0')
        self.baseline=dict(DeviceSerialNumber='15328133',TriggerMode='Off',AcquisitionFrameRate='95',Gain='0')
        self.plan=dict(standby_baseline=self.baseline,retention_ref={'path':'/retention','sha256':'a'*64},
            old_owner=dict(pid=self.m.PID,start_ticks=self.m.START,session=str(self.session),unit='old.service'),
            retention=dict(original_owner=dict(argv=['python','fake'])),source_sha256={str(r.LIFECYCLE):r.LIFECYCLE_SHA})
        self.good=dict(passed=True,target=self.baseline,connections=2,parameter_writes=0,acquisition_started=False,errors=[],
            snapshots=[dict(connection=n,values=self.baseline,differences={}) for n in [0]*5+[1]*5],
            original_failed_stage_accepted=False,release_approved=False,
            black_framebuffer_observations=[dict(dmd={'all_RGB_zero':True},slm={'all_RGB_zero':True})])
        rp=self.root/'recheck.json';self.write(rp,self.good)
        old=dict(self.plan)
        failure=dict(completed=False,forced_kill=False,actions=[dict(action='release',result=dict(click_count=1,executed=True))])
        cleanup=dict(error=None,camera_settings_restored=False,camera_before=self.baseline,
            camera_after=dict(self.baseline,AcquisitionFrameRate='90'),
            restore_differences={'AcquisitionFrameRate':dict(before='95',after='90')})
        later=dict(old_failure_reclassified=False,report=self.good,report_ref=self.m.ref(rp),
            after=dict(pid=self.m.PID,parked_verification_passed=True,no_SDK_connection=True,no_projection=True,
                heartbeat=dict(hardware_fault=None,projection_armed=False)))
        self.values=dict(old_plan=old,old_failure=failure,old_cleanup=cleanup,later_recheck=later)
        self.evidence={}
        for name,value in self.values.items():
            path=self.root/(name+'.json');self.write(path,value);self.evidence[name]=(path,self.m.sha(path))
        (self.root/'test_recovery.py').write_text('# fixture\n')
        (self.session/'events.jsonl').write_text('{}\n')
        self.patch(r,'ROOT',self.root);self.patch(r,'EVIDENCE',self.evidence);self.patch(r,'engine',Mock(return_value=self.m))
        self.patch(self.m,'SESSION',self.session);self.patch(self.m,'PROC',self.proc)

    @staticmethod
    def write(path,value):Path(path).write_text(json.dumps(value),encoding='utf-8')
    def patch(self,module,name,value):
        p=patch.object(module,name,value);p.start();self.addCleanup(p.stop)
    def reset_evidence(self,key):
        path,_=self.evidence[key];self.write(path,self.values[key]);self.evidence[key]=(path,self.m.sha(path))

    def test_default_subprocess_no_ui_sdk_or_engine_import(self):
        output=subprocess.check_output([sys.executable,'-B',str(HERE/'recovery.py')],text=True)
        self.assertEqual(json.loads(output),dict(execution=False,UI_actions=0,SDK_connections=0,forced_kill=False,old_failure_reclassified=False))

    def test_authorization_precedes_engine_IO(self):
        r.engine.side_effect=AssertionError('no authorized operation')
        for kwargs in ({},{'execute':True},{'light_blocked_confirmed':True}):
            with self.assertRaises(ValueError):r.close(None,None,None,None,None,**kwargs)
            with self.assertRaises(ValueError):r.verify(None,None,None,None,**kwargs)

    def test_full_readonly_2_connections_10_snapshots_required(self):
        self.assertTrue(r.readonly_pass(self.baseline,self.good))
        for changed in (dict(snapshots=self.good['snapshots'][:9]),dict(parameter_writes=1),dict(acquisition_started=True),
                dict(target=dict(self.baseline,AcquisitionFrameRate='90')),dict(errors=['x']),dict(connections=1)):
            self.assertFalse(r.readonly_pass(self.baseline,dict(self.good,**changed)))
        bad=deepcopy(self.good);bad['snapshots'][4]['connection']=1;self.assertFalse(r.readonly_pass(self.baseline,bad))

    def test_only_original_fr95_to90_failure_with_later95_pass(self):
        result=r.preserved(self.m,self.plan);self.assertEqual(result['old_cleanup']['camera_after']['AcquisitionFrameRate'],'90')
        self.assertFalse(result['later_recheck']['old_failure_reclassified'])
        for key,change in (('old_cleanup',{'camera_settings_restored':True}),('old_failure',{'completed':True}),
                ('later_recheck',{'old_failure_reclassified':True})):
            saved=deepcopy(self.values[key]);self.values[key].update(change);self.reset_evidence(key)
            with self.assertRaises(ValueError):r.preserved(self.m,deepcopy(self.plan))
            self.values[key]=saved;self.reset_evidence(key)

    def test_lossy_partial_or_modified_baseline_is_rejected(self):
        self.values['later_recheck']['report']['snapshots'][0]['values']=dict(self.baseline,AcquisitionFrameRate='90')
        self.reset_evidence('later_recheck')
        with self.assertRaises(ValueError):r.preserved(self.m,self.plan)

    def test_pinned_original_recheck_sha_is_not_replaced(self):
        path,_=self.evidence['later_recheck'];self.write(path,{})
        with self.assertRaises(ValueError):r.preserved(self.m,self.plan)

    def test_checked_keeps_old_engine_and_binds_new_sources_and_evidence(self):
        self.patch(self.m,'checked',Mock(side_effect=lambda *_:deepcopy(self.plan)))
        plan=r.checked(self.m,'ret','a'*64,self.m.sha(r.SELF),'old.service')
        self.assertEqual(plan['source_sha256'][str(r.LIFECYCLE)],r.LIFECYCLE_SHA)
        self.assertIn(str(r.SELF),plan['source_sha256']);self.assertIn(str(self.root/'test_recovery.py'),plan['source_sha256'])
        for path,digest in self.evidence.values():self.assertEqual(plan['source_sha256'][str(path)],digest)

    def test_close_only_normal_exit_once_no_second_release(self):
        self.patch(self.m,'checked',Mock(side_effect=lambda *_:deepcopy(self.plan)))
        self.patch(self.m,'owner_inventory',Mock(return_value=[dict(pid=self.m.PID,start_ticks=self.m.START)]))
        self.patch(self.m,'heartbeat',Mock(return_value={}))
        self.patch(self.m,'start_ticks',Mock(return_value=self.m.START))
        self.patch(self.m,'service_status',Mock(side_effect=lambda *_:dict(MainPID=str(self.m.PID)) if self.owner.exists()
            else dict(MainPID='0',Result='success',ExecMainStatus='0')))
        self.patch(self.m,'no_owner',Mock(side_effect=lambda:self.assertFalse(self.owner.exists())))
        keys=[]
        def press(symbol,down):
            keys.append((symbol,down))
            if len(keys)==4:
                self.owner.rename(self.proc/'gone')
                row=dict(state='operator_confirmed_light_blocked_exit',closed_safely=True)
                (self.session/'events.jsonl').write_text('{}\n'+json.dumps(row)+'\n')
        client=SimpleNamespace(inspect=Mock(return_value=dict(windows=dict(control={'wid':'0x1'}))),
            Input=Mock(return_value=SimpleNamespace(key=press,close=Mock())))
        native=SimpleNamespace(invoke=Mock(side_effect=AssertionError('must not release again')))
        self.patch(self.m,'_clients',Mock(return_value=(object(),client,native)))
        result=r.close('ret','a'*64,self.m.sha(r.SELF),'old.service',self.root/'CLOSE',execute=True,light_blocked_confirmed=True)
        self.assertTrue(result['completed']);self.assertFalse(result['old_cleanup_exact_passed'])
        self.assertFalse(result['old_failure_reclassified']);self.assertTrue(result['later_baseline_reverified'])
        self.assertEqual(len(keys),4);native.invoke.assert_not_called()
        self.assertEqual(self.m.read(self.root/'CLOSE/ACTIONS.json'),[dict(action='Alt_F4',count=1)])

    def test_output_collision_prevents_UI(self):
        self.patch(r,'checked',Mock(return_value=deepcopy(self.plan)))
        self.patch(self.m,'_clients',Mock(side_effect=AssertionError('must not touch UI')))
        with self.assertRaises(ValueError):r.close('ret','a'*64,'b'*64,'old.service',self.root,execute=True,light_blocked_confirmed=True)

    def cold_fixture(self):
        self.patch(self.m,'checked',Mock(side_effect=lambda *_:deepcopy(self.plan)))
        plan=r.checked(self.m,'ret','a'*64,self.m.sha(r.SELF),'old.service')
        pp=self.root/'RECOVERY_CLOSE_PLAN.json';self.write(pp,plan)
        ep=self.root/'closed_events.jsonl';ep.write_text('{}\n')
        closed=dict(schema='E045-later-readonly-recovery-closed-1',completed=True,old_normal_exit_verified=True,
            physical_shielding_confirmed=True,forced_kill=False,old_cleanup_exact_passed=False,
            old_failure_reclassified=False,later_baseline_reverified=True,
            prospective_plan=self.m.ref(pp),events_snapshot=self.m.ref(ep),
            service_status=dict(MainPID='0',Result='success',ExecMainStatus='0'),
            **{k:plan[k] for k in ('old_owner','retention_ref','standby_baseline','source_sha256')})
        cp=self.root/'CLOSED.json';self.write(cp,closed);return cp,closed

    def test_cold_verify_exact_full95_and_launcher_schema(self):
        cp,closed=self.cold_fixture();calls=[]
        def fake_verify(baseline,guard,**kwargs):
            guard();calls.append(baseline);guard();return deepcopy(self.good)
        old=SimpleNamespace(_load_clients=Mock())
        fake=SimpleNamespace(ReadOnlyCamera=object,verify_baseline=fake_verify)
        self.patch(self.m,'no_owner',Mock());self.patch(self.m,'service_status',Mock(return_value=closed['service_status']))
        with patch.dict(sys.modules,{'optical_lab':SimpleNamespace(legacy_gui=old),'camera_recheck':fake}):
            result=r.verify(cp,self.m.sha(cp),self.m.sha(r.SELF),self.root/'VERIFY',execute=True,light_blocked_confirmed=True)
        self.assertEqual(calls,[self.baseline]);self.assertTrue(result['completed'])
        self.assertEqual(result['camera_readonly_connections'],2);self.assertEqual(result['camera_readonly_snapshots'],10)
        self.assertEqual(result['parameter_writes'],0);self.assertFalse(result['acquisition_started'])
        self.assertFalse(result['old_cleanup_exact_passed']);self.assertFalse(result['old_failure_reclassified'])
        spec=importlib.util.spec_from_file_location('E045_recovery_test_launcher',HERE/'run.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        baselinepath=self.root/'baseline.json';self.write(baselinepath,dict(camera_baseline=self.baseline))
        retained=dict(original_owner=dict(pid=self.m.PID,start_ticks=self.m.START),recovery_record=self.m.ref(baselinepath))
        verified=launcher.validate_lifecycle(self.m.ref(self.root/'VERIFY/LIFECYCLE_RESULT.json'),closed['retention_ref'],retained)
        self.assertEqual(verified,result);old._load_clients.assert_called_once()

    def test_cold_fail_keeps_actual_counts_and_does_not_claim_pass(self):
        cp,closed=self.cold_fixture();bad=dict(self.good,passed=False,connections=1,snapshots=self.good['snapshots'][:5],errors=['changed frame rate'])
        self.patch(self.m,'no_owner',Mock());self.patch(self.m,'service_status',Mock(return_value=closed['service_status']))
        with patch.dict(sys.modules,{'optical_lab':SimpleNamespace(legacy_gui=SimpleNamespace(_load_clients=Mock())),
                'camera_recheck':SimpleNamespace(ReadOnlyCamera=object,verify_baseline=Mock(return_value=bad))}):
            result=r.verify(cp,self.m.sha(cp),self.m.sha(r.SELF),self.root/'FAILED_VERIFY',execute=True,light_blocked_confirmed=True)
        self.assertFalse(result['completed']);self.assertFalse(result['cold_baseline_verified'])
        self.assertEqual(result['camera_readonly_connections'],1);self.assertEqual(result['camera_readonly_snapshots'],5)
        self.assertEqual(result['errors'],['changed frame rate'])

    def test_source_pin_rejects_replaced_frozen_engine(self):
        path=self.root/'changed_engine.py';path.write_text('# not frozen')
        with patch.object(r,'LIFECYCLE',path),patch.object(r,'engine',r.engine):
            # Call the original loader, not the mocked fixture entry.
            loader=SPEC.loader;fresh=importlib.util.module_from_spec(SPEC);loader.exec_module(fresh)
            with patch.object(fresh,'LIFECYCLE',path),self.assertRaises(ValueError):fresh.engine()


if __name__=='__main__':unittest.main()
