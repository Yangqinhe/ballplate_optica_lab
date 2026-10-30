"""Offline-only real MuJoCo IPC tests; all CCD/display calls are mocks."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'live_manual_v17')]
from live_loop_v23.core import DEFAULT_HEAD,DEFAULT_SHA,load_head,limited_action
from live_adapt_v22.core import digest
PYTHON=PROJECT.parent/'ballplate_sigle_cnn_project/.venv/bin/python'


class RoundTrip(unittest.TestCase):
    def run_child(self,folder,bad=False,duplicate_push=False):
        cmd=[str(PYTHON),str(PROJECT/'live_loop_v23/child.py'),'--head',str(DEFAULT_HEAD),
            '--sha',DEFAULT_SHA,'--output',str(folder),'--seconds','.08']
        env=dict(os.environ,MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
        h,b=load_head(DEFAULT_HEAD,DEFAULT_SHA);previous=np.zeros(2);steps=[];queries=[]
        with (folder.parent/'stderr.log').open('wb') as err:
            child=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,bufsize=0,cwd=PROJECT,env=env)
            try:
                until=time.monotonic()+25;final=None
                while final is None:
                    if time.monotonic()>until:raise TimeoutError('test child timed out')
                    if not select.select([child.stdout],[],[],.1)[0]:continue
                    line=child.stdout.readline()
                    if not line:raise RuntimeError('unexpected EOF: '+(folder.parent/'stderr.log').read_text())
                    row=json.loads(line)
                    if row['kind']=='optical_query':
                        seq=row['sequence'];queries.append(row)
                        dmd=np.load(row['dmd'],allow_pickle=False)
                        self.assertEqual(dmd.shape,(600,600));self.assertTrue(np.isin(dmd,(0,255)).all())
                        self.assertEqual(digest(row['dmd']),row['dmd_sha256'])
                        events=[dict(id='test-push-once',delta_v_m_s=[.06,0.])] if seq==1 or (seq==2 and duplicate_push) else []
                        reply=dict(sequence=seq+(1 if bad else 0),dmd_sha256=row['dmd_sha256'],
                            raw_integrals8=[1e5]*8,query_latency_s=.1,push_events=events)
                        child.stdin.write((json.dumps(reply)+'\n').encode());child.stdin.flush()
                    elif row['kind']=='mujoco_step':
                        raw,action=limited_action(h,np.full(8,1e5),previous);previous=action
                        np.testing.assert_allclose(row['action_limited_rad'],action,atol=1e-12)
                        steps.append(row)
                    elif row['kind']=='pilot_complete':final=row['report']
                code=child.wait(timeout=3)
                return final,code,steps,queries
            finally:
                if child.poll() is None:child.kill();child.wait(timeout=3)
                child.stdin.close();child.stdout.close()

    def test_real_mujoco_mock_CCD_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            r,c,steps,queries=self.run_child(Path(d)/'sim')
            self.assertEqual(c,0);self.assertTrue(r['completed']);self.assertEqual(len(steps),4)
            self.assertEqual(sum(len(x['push_events_applied']) for x in steps),1)
            self.assertEqual(steps[1]['push_events_applied'][0]['delta_v_m_s'],[.06,0.])
            self.assertEqual(r['RAW_input_ages_s'],[.18,.12,.06,0.])

    def test_wrong_reply_rejected_no_step(self):
        with tempfile.TemporaryDirectory() as d:
            r,c,steps,q=self.run_child(Path(d)/'sim',bad=True)
            self.assertNotEqual(c,0);self.assertFalse(r['completed']);self.assertEqual(steps,[])

    def test_duplicate_disturbance_is_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            r,c,steps,q=self.run_child(Path(d)/'sim',duplicate_push=True)
            self.assertNotEqual(c,0);self.assertIn('repeated disturbance',r['error'])


class WorkerRoundTrip(unittest.TestCase):
    def test_cancelled_worker_never_projects(self):
        from live_loop_v23.worker import LoopWorker
        h,b=load_head(DEFAULT_HEAD,DEFAULT_SHA)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);w=LoopWorker(None,root/'unused_camera',1000,roi=b.contract['roi'])
            w.camera=SimpleNamespace(active=b.contract['expected_active'],write_trace=[],revalidate=Mock())
            w.project_and_settle=Mock();w.checked_capture=Mock();w.pushes.begin();w.pushes.add(0,1,.06)
            w.static_cancel.set();results=[];w.loop_result.connect(results.append)
            w.run_loop(DEFAULT_HEAD,DEFAULT_SHA,root/'loop',.06,[.025,-.025])
            self.assertTrue(results[0]['report']['cancelled']);w.project_and_settle.assert_not_called()
            w.checked_capture.assert_not_called();self.assertEqual(len(results[0]['report']['unapplied_pushes']),1)

    def test_native_worker_mock_camera_and_display(self):
        from live_loop_v23.worker import LoopWorker
        h,b=load_head(DEFAULT_HEAD,DEFAULT_SHA)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);w=LoopWorker(None,root/'unused_camera',1000,roi=b.contract['roi'])
            w.camera=SimpleNamespace(active=b.contract['expected_active'],write_trace=[],revalidate=Mock())
            w.project_and_settle=Mock(side_effect=lambda *a:(dict(actual_framebuffer_checked=True,verified_unix_ns=0),time.monotonic()))
            w.checked_capture=Mock(return_value=(np.full((554,552),20,np.uint8),dict(FrameID=1,logical_sequence=1)))
            w.park_checked=Mock();w.pushes.begin();w.pushes.add(1,-1,.04)
            results=[];w.loop_result.connect(results.append)
            w.run_loop(DEFAULT_HEAD,DEFAULT_SHA,root/'loop',.06,[.025,-.025])
            self.assertEqual(len(results),1);r=results[0]['report'];self.assertTrue(r['completed'])
            self.assertEqual(r['actions'],3);self.assertEqual(w.checked_capture.call_count,3)
            self.assertEqual(w.project_and_settle.call_count,3);w.park_checked.assert_called_once()
            n,view=w.loop_view.get();self.assertEqual(len(view['trace_rows']),3)
            self.assertEqual(view['actual'].shape,(554,552));self.assertIsInstance(view['rgb'],np.ndarray)
            self.assertEqual(view['actual_sequence'],2)


if __name__=='__main__':unittest.main()
