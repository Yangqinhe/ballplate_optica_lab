"""Pure file/process/UI/SDK fakes; no real window or camera actions."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('E045_test_lifecycle',HERE/'lifecycle.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.session=self.root/'OLD/GUI';self.session.mkdir(parents=True)
        self.proc=self.root/'proc';self.proc.mkdir()
        constants=dict(ROOT=self.root,SESSION=self.session,PROC=self.proc,LAUNCH=self.root/'LAUNCH.json',
            NATIVE=self.root/'native.py',LEGACY_GUI=self.root/'legacy_gui.py',RECHECK=self.root/'recheck.py',PURE_CLIENT_SOURCES={})
        p=patch.multiple(m,**constants);p.start();self.addCleanup(p.stop)
        for name in ('test_lifecycle.py','native.py','legacy_gui.py','recheck.py'):(self.root/name).write_text('# fixture\n')
        self.unit='old-preview.service';self.write(m.LAUNCH,{'command':['systemd-run','--unit='+self.unit]})
        p=patch.multiple(m,LAUNCH_SHA=m.sha(m.LAUNCH),NATIVE_SHA=m.sha(m.NATIVE),LEGACY_GUI_SHA=m.sha(m.LEGACY_GUI),RECHECK_SHA=m.sha(m.RECHECK))
        p.start();self.addCleanup(p.stop)
        self.retroot=self.root/'RETENTION';self.retroot.mkdir()
        self.baseline=dict(DeviceSerialNumber='15328133',TriggerMode='Off',Gain='0',ExposureTime='67.17')
        self.write(self.retroot/'contract.json',dict(original_fault_before=self.baseline,original_pid=m.PID,original_owner_start_ticks=m.START))
        self.write(self.retroot/'recovery.json',{'camera_baseline':self.baseline})
        self.ret=dict(schema='E045-planned-retention-1',status='PLANNED_FILES_ONLY_NOT_LIFECYCLE_RELEASE',
            original_baseline_unchanged=True,original_owner=dict(pid=m.PID,start_ticks=m.START,session=str(self.session),argv=['python','-m','fake']),
            restart_contract=m.ref(self.retroot/'contract.json'),recovery_record=m.ref(self.retroot/'recovery.json'),
            retained_current_active=dict(Gain='10.881',ExposureTime='74.207'))
        self.retpath=self.retroot/'RESULT.json';self.write(self.retpath,self.ret)
        self.write(self.retroot/'SOURCE_ARCHIVE_MANIFEST.json',{'original_source_sha256':{}})
        artifacts={p.name:m.sha(p) for p in self.retroot.iterdir()};self.write(self.retroot/'MANIFEST.json',{'file_sha256':artifacts})
        p=patch.object(m,'RETENTION_MANIFEST_SHA',m.sha(self.retroot/'MANIFEST.json'));p.start();self.addCleanup(p.stop)
        (self.session/'events.jsonl').write_text('{}\n')
        self.selfsha=m.sha(m.SELF)

    @staticmethod
    def write(path,value):Path(path).write_text(json.dumps(value),encoding='utf-8')

    def plan(self):return m.checked(self.retpath,m.sha(self.retpath),self.selfsha,self.unit)

    def owner_proc(self):
        p=self.proc/str(m.PID);p.mkdir();(p/'stat').write_text(str(m.PID)+' (owner with spaces) '+' '.join(['S']+['0']*18+[m.START]))
        (p/'cmdline').write_bytes(b'python\0-m\0fake\0');return p

    def good_report(self):
        return dict(error=None,camera_settings_restored=True,camera_before=self.baseline,camera_after=self.baseline,restore_differences={})

    def make_closed(self,**changes):
        out=self.root/'close';out.mkdir();plan=self.plan();plan.update(exit_event_offset=1,reports_before=[])
        self.write(out/'CLOSE_PLAN.json',plan)
        report=self.session/'live_NEW/report.json';report.parent.mkdir();self.write(report,self.good_report())
        exitrow={'state':'operator_confirmed_light_blocked_exit','closed_safely':True}
        (self.session/'events.jsonl').write_text('{}\n'+json.dumps(exitrow)+'\n')
        with patch.object(m,'no_owner'),patch.object(m,'service_status',return_value={'MainPID':'0','Result':'success','ExecMainStatus':'0'}):
            value=m.prove_closed(plan,out)
        value.update(changes)
        if changes:self.write(out/'CLOSED.json',value)
        return out/'CLOSED.json',value

    def test_default_describe_reads_no_sources_UI_or_SDK(self):
        completed=subprocess.run([sys.executable,'-B',str(HERE/'lifecycle.py')],capture_output=True,text=True,check=True)
        value=json.loads(completed.stdout);self.assertFalse(value['execution']);self.assertEqual(value['UI_actions'],0)
        self.assertEqual(value['SDK_connections'],0);self.assertFalse(value['forced_kill'])

    def test_authorization_precedes_all_reads_runtime_or_SDK(self):
        with patch.object(m,'checked',side_effect=AssertionError('must not load')):
            for kwargs in ({},{'execute':True},{'light_blocked_confirmed':True}):
                with self.assertRaises(ValueError):m.close(None,None,None,None,None,**kwargs)
                with self.assertRaises(ValueError):m.verify(None,None,None,None,**kwargs)
                with self.assertRaises(ValueError):m.confirm_closed(None,None,None,**kwargs)

    def test_checked_retention_pins_baseline_and_exact_original_unit(self):
        p=self.plan();self.assertEqual(p['standby_baseline'],self.baseline)
        self.assertEqual(p['old_owner'],dict(pid=m.PID,start_ticks=m.START,session=str(self.session),unit=self.unit))
        with self.assertRaises(ValueError):m.checked(self.retpath,m.sha(self.retpath),self.selfsha,'different.service')
        with self.assertRaises(ValueError):m.checked(self.retpath,m.sha(self.retpath),'0'*64,self.unit)

    def test_original_retention_manifest_cannot_be_replaced_unanchored(self):
        self.write(self.retroot/'MANIFEST.json',{'file_sha256':{}})
        with self.assertRaises(ValueError):self.plan()

    def test_before_and_after_complete_standby_required(self):
        m.exact_report(self.good_report(),self.baseline)
        for change in ({'camera_before':{}},{'camera_after':dict(self.baseline,Gain='11')},
                {'restore_differences':{'Gain':'changed'}},{'error':'SDK cleanup'},{'camera_settings_restored':False}):
            with self.subTest(change=change),self.assertRaises(ValueError):m.exact_report(dict(self.good_report(),**change),self.baseline)

    def test_PID_lifetime_no_reuse_no_cold_second_owner(self):
        p=self.owner_proc();self.assertEqual(m.start_ticks(m.PID),m.START)
        with self.assertRaises(ValueError):m.no_owner()
        p.rename(self.proc/'gone')
        with patch.object(m,'owner_inventory',return_value=[{'pid':999}]):
            with self.assertRaises(ValueError):m.no_owner()
        with patch.object(m,'owner_inventory',return_value=[]):m.no_owner()

    def test_owner_argv_recognizes_future_E045_relative_absolute_and_B_entry(self):
        self.assertTrue(m.owner_argv(['python','-B',str(self.root/'run.py'),'--hardware'],self.root))
        self.assertTrue(m.owner_argv(['python','-B','run.py','--hardware'],self.root))
        self.assertTrue(m.owner_argv(['python','-m','optical_lab.preview','--hardware'],self.root))
        self.assertTrue(m.owner_argv(['python','-m','optical_lab.dense_startup','--hardware'],self.root))
        self.assertFalse(m.owner_argv(['python','run.py'],self.root))
        self.assertFalse(m.owner_argv(['python','/unrelated/run.py','--hardware'],self.root))

    def test_heartbeat_exact_original_live_and_released_states(self):
        self.owner_proc();p=self.plan()
        h=dict(pid=m.PID,deployment_ui_revision=23,unix_ns=m.time.time_ns(),projection_armed=False,
            hardware_fault=None,loop_active=False,adaptation_active=False,deployment_busy=False,
            live_camera=True,busy=True,camera_reserved=True,camera_active_readback=self.ret['retained_current_active'])
        self.write(self.session/'heartbeat.json',h);self.assertTrue(m.heartbeat(p)['live_camera'])
        with self.assertRaises(ValueError):m.heartbeat(p,released=True)
        h.update(live_camera=False,busy=False,camera_reserved=False);self.write(self.session/'heartbeat.json',h)
        self.assertFalse(m.heartbeat(p,released=True)['live_camera'])
        h['hardware_fault']='restore failed';self.write(self.session/'heartbeat.json',h)
        with self.assertRaises(ValueError):m.heartbeat(p,released=True)

    def test_normal_close_exact_one_release_AltF4_no_force_or_retry(self):
        proc=self.owner_proc();plan=self.plan();calls=[];main={'wid':'0x1234','rect':[1080,600,800,700]}
        observation=dict(pid=m.PID,windows={'control':main})
        def release(ui,key,execute):
            calls.append(key);path=self.session/'live_NEW/report.json';path.parent.mkdir();self.write(path,self.good_report())
            return {'executed':True,'key':key}
        test=self
        class Input:
            def __init__(self,*_):pass
            def key(self,key,pressed):
                calls.append((key,pressed))
                if key==0xffe9 and not pressed:
                    proc.rename(test.proc/'gone')
                    with (test.session/'events.jsonl').open('a') as stream:stream.write(json.dumps({'state':'operator_confirmed_light_blocked_exit','closed_safely':True})+'\n')
            def close(self):pass
        clients=SimpleNamespace(inspect=lambda *_args,**_kwargs:observation,Input=Input)
        native=SimpleNamespace(invoke=release)
        def status(_):return {'MainPID':str(m.PID)} if proc.exists() else {'MainPID':'0','Result':'success','ExecMainStatus':'0'}
        with patch.object(m,'owner_inventory',side_effect=lambda:[{'pid':m.PID,'start_ticks':m.START}] if proc.exists() else []),\
                patch.object(m,'service_status',side_effect=status),patch.object(m,'heartbeat',return_value={}),\
                patch.object(m,'_clients',return_value=(object(),clients,native)):
            result=m.close(self.retpath,m.sha(self.retpath),self.selfsha,self.unit,self.root/'NORMAL',execute=True,light_blocked_confirmed=True)
        self.assertTrue(result['old_normal_exit_verified']);self.assertFalse(result['forced_kill'])
        self.assertEqual(calls[0],'release');self.assertEqual(len(calls),5)
        self.assertEqual(calls[1:],[(0xffe9,True),(0xffc1,True),(0xffc1,False),(0xffe9,False)])

    def test_prove_closed_requires_new_unique_event_and_new_report_not_mtime(self):
        path,closed=self.make_closed();self.assertTrue(closed['old_normal_exit_verified'])
        self.assertEqual(closed['cleanup_report']['path'],str(self.session/'live_NEW/report.json'))
        (self.session/'events.jsonl').write_text('{}\n')
        with patch.object(m,'no_owner'),patch.object(m,'service_status',return_value=closed['service_status']):
            with self.assertRaises(ValueError):m.prove_closed(m.read(path.parent/'CLOSE_PLAN.json'),self.root/'OTHER')

    def test_bad_service_exit_cannot_qualify(self):
        path,closed=self.make_closed()
        with patch.object(m,'no_owner'),patch.object(m,'service_status',return_value={'MainPID':'0','Result':'exit-code','ExecMainStatus':'1'}):
            with self.assertRaises(ValueError):m.prove_closed(m.read(path.parent/'CLOSE_PLAN.json'),self.root/'OTHER')

    def test_user_assisted_normal_exit_adoption_is_readonly_and_same_source(self):
        path,closed=self.make_closed();(path.parent/'CLOSED.json').rename(path.parent/'FIRST_CLOSED.json')
        (path.parent/'events_after.jsonl').rename(path.parent/'FIRST_events_after.jsonl')
        with patch.object(m,'no_owner'),patch.object(m,'service_status',return_value=closed['service_status']):
            result=m.confirm_closed(path.parent/'CLOSE_PLAN.json',m.sha(path.parent/'CLOSE_PLAN.json'),self.selfsha,
                execute=True,light_blocked_confirmed=True)
        self.assertTrue(result['completed']);self.assertFalse(result['forced_kill'])

    def cold_fakes(self,result):
        import optical_lab
        helper=SimpleNamespace(_load_clients=lambda *_:None)
        recheck=SimpleNamespace(ReadOnlyCamera=type('ReadOnlyCamera',(),{}),verify_baseline=lambda expected,guard,**_:guard() or deepcopy(result))
        return helper,recheck,patch.object(optical_lab,'legacy_gui',helper,create=True)

    def test_cold_two_connections_ten_snapshots_no_write_acquisition_and_final_schema(self):
        path,closed=self.make_closed();result=dict(passed=True,connections=2,snapshots=[{'values':self.baseline,'differences':{}} for _ in range(10)],
            parameter_writes=0,acquisition_started=False,errors=[])
        helper,recheck,package_patch=self.cold_fakes(result)
        with package_patch,patch.dict(sys.modules,{'optical_lab.legacy_gui':helper,'camera_recheck':recheck}),\
                patch.object(m,'no_owner'),patch.object(m,'service_status',return_value=closed['service_status']):
            value=m.verify(path,m.sha(path),self.selfsha,self.root/'VERIFY',execute=True,light_blocked_confirmed=True)
        self.assertEqual(value['schema'],'E045-normal-shielded-release-1');self.assertTrue(value['cold_baseline_verified'])
        self.assertEqual((value['camera_readonly_connections'],value['camera_readonly_snapshots']),(2,10))
        self.assertFalse(value['acquisition_started']);self.assertEqual(value['parameter_writes'],0);self.assertEqual(value['errors'],[])
        self.assertEqual(value['old_owner'],{'pid':m.PID,'start_ticks':m.START})

    def test_cold_insufficient_or_modified_snapshot_never_PASS(self):
        path,closed=self.make_closed()
        base=dict(passed=True,connections=2,snapshots=[{'values':self.baseline,'differences':{}} for _ in range(10)],
            parameter_writes=0,acquisition_started=False,errors=[])
        for index,change in enumerate(({'connections':1},{'parameter_writes':1},{'acquisition_started':True},{'errors':['timeout']},
                {'snapshots':[{'values':{},'differences':{}} for _ in range(10)]})):
            helper,recheck,package_patch=self.cold_fakes(dict(base,**change))
            with package_patch,patch.dict(sys.modules,{'optical_lab.legacy_gui':helper,'camera_recheck':recheck}),\
                    patch.object(m,'no_owner'),patch.object(m,'service_status',return_value=closed['service_status']):
                value=m.verify(path,m.sha(path),self.selfsha,self.root/('VERIFY_'+str(index)),execute=True,light_blocked_confirmed=True)
            self.assertFalse(value['completed']);self.assertFalse(value['cold_baseline_verified'])

    def test_wrong_owner_tuple_in_closed_evidence_blocks_before_SDK(self):
        path,closed=self.make_closed(old_owner=dict(pid=m.PID,start_ticks='reused',session=str(self.session),unit=self.unit))
        with self.assertRaises(ValueError):m.verify(path,m.sha(path),self.selfsha,self.root/'VERIFY',execute=True,light_blocked_confirmed=True)


if __name__=='__main__':unittest.main()
