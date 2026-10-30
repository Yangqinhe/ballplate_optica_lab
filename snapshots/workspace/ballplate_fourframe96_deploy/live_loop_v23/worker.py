"""Interactive jobs in the ONE original CCD owner; native display handshake."""
from dataclasses import replace
import json
import os
from pathlib import Path
import select
import subprocess
import time
import traceback

import cv2
import numpy as np
from PyQt5 import QtCore
from live_adapt_v22.worker import AdaptationWorker, SIM_PYTHON
from live_adapt_v22.core import digest, write_new
from live_deploy_v20.worker import StaticCancelled
from live_manual_v19_trigger.camera import WorkerGuard
from realtime_model import LatestFrame, LivePattern, patterns
from diffraction_model import CachedASM, OpticalReference, oriented
from .core import load_head, PushQueue
from .camera import StreamingCamera

PROJECT = Path(__file__).resolve().parents[1]


class LoopWorker(AdaptationWorker):
    loop_result = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_factory = lambda: StreamingCamera(WorkerGuard(self.check_stop))
        self.loop_request = None; self.loop_busy = False
        self.loop_view = LatestFrame(); self.pushes = PushQueue()
        self.view_state = {}
        self.view_trace = []

    def publish(self, data):
        if data['kind'] == 'mujoco_step':
            data = {k:v for k,v in data.items() if k not in ('rgb','rgb_sha256')}
        self.view_state = dict(self.view_state, **data)
        if data['kind'] == 'optical': self.view_state['actual_sequence'] = data['sequence']
        if data['kind'] == 'mujoco_step':
            self.view_trace.append({k:v for k,v in data.items() if k not in ('rgb_image','rgb','rgb_sha256')})
            self.view_state['trace_rows'] = self.view_trace[-500:]
        self.loop_view.put(dict(self.view_state))

    def request_loop(self, head_path, head_sha, folder, seconds, initial_xy):
        with self.command_lock:
            if (self.loop_request or self.loop_busy or self.adaptation_request or self.adaptation_busy
                    or self.static_request or self.static_busy or self.configuration or self.reference_requested
                    or self.spatial_requested or self.auto_requested.is_set()):
                raise RuntimeError('another task owns the camera')
            if not self.camera or not self.camera.acquiring or self.startup_dark_required:
                raise RuntimeError('start the one CCD owner first')
            self.static_cancel.clear(); self.pushes.begin()
            self.loop_request = (Path(head_path), head_sha, Path(folder), seconds, list(initial_xy))

    def request_adaptation(self, *args, **kwargs):
        if self.loop_request or self.loop_busy: raise RuntimeError('interactive loop owns the camera')
        return super().request_adaptation(*args, **kwargs)

    def request_static(self, *args, **kwargs):
        if self.loop_request or self.loop_busy: raise RuntimeError('interactive loop owns the camera')
        return super().request_static(*args, **kwargs)

    def next_frame(self):
        with self.command_lock:
            pending = self.loop_request; self.loop_request = None
        if pending:
            self.loop_busy = True
            try: self.run_loop(*pending)
            finally: self.loop_busy = False
        return super().next_frame()

    def run_loop(self, head_path, head_sha, folder, seconds, initial_xy):
        folder.mkdir(parents=True, exist_ok=False)
        started = time.monotonic(); deadline = started+min(3300., 30+6*round(seconds/.02))
        report = dict(completed=False, error=None, real_actuators_enabled=False,
            realtime_20ms_qualified=False, actual_park_verified=False, actions=0,
            same_SDK_owner=True, camera_restore_pending_normal_release=True,
            phase_description='fixed prior hand-designed quadratic -200mm, 4x4 macro-pixels; NOT newly PPO-learned')
        bundle = None; child = None; fatal = None; initial_bad = len(self.bad)
        initial_writes = len(self.camera.write_trace)
        self.view_state = {}
        self.view_trace = []
        try:
            head, bundle = load_head(head_path, head_sha, audit=True)
            self.camera.revalidate(self.exposure, self.requested_roi)
            self.check_job(bundle, deadline)
            source = PROJECT/'live_loop_v23/child.py'
            cmd = [str(SIM_PYTHON), str(source), '--head', str(head_path), '--sha', head_sha,
                '--output', str((folder/'MuJoCo').resolve()), '--seconds', str(seconds),
                '--initial-xy', *map(str, initial_xy)]
            write_new(folder/'PROSPECTIVE_PROTOCOL.json', dict(head=str(head_path), head_sha256=head_sha,
                bundle=str(bundle.path), bundle_sha256=bundle.sha, command=cmd,
                seconds=seconds, initial_xy_m=initial_xy, wall_seconds_limit=deadline-started,
                camera_active=dict(self.camera.active), settle_seconds=bundle.data['rules']['minimum_software_settle_seconds'],
                CCD_history=0, simulated_dt_s=.02, RAW_input_ages_s=[.18,.12,.06,0.],
                software_settle_NOT_hardware_synchronization=True, real_actuators_enabled=False,
                ideal_CCD_display_only_NOT_used_for_actions=True,
                sources={str(p): digest(p) for p in (PROJECT/'live_loop_v23').glob('*.py')}))
            engine = CachedASM(); cfg = OpticalReference(**bundle.contract['optical_config'])
            settings = replace(LivePattern(**bundle.contract['settings']), dmd='file', slm='file', full_panel=False)
            sim_dir = (folder/'MuJoCo').resolve(); sequence = 0

            def bound_array(data, key, shape=None):
                path = Path(data[key]).resolve()
                if not path.is_relative_to(sim_dir) or digest(path) != data[key+'_sha256']:
                    raise ValueError('unbound child image')
                value = np.load(path, allow_pickle=False)
                if value.dtype != np.uint8 or (shape and value.shape != shape):
                    raise ValueError('bad child image')
                return value

            env = dict(os.environ, MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
            with (folder/'child_stderr.log').open('xb') as err:
                child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                    bufsize=0, cwd=PROJECT, env=env)
                while True:
                    self.check_job(bundle, deadline)
                    if not select.select([child.stdout], [], [], .05)[0]:
                        if child.poll() is not None: raise RuntimeError('child exited without final report')
                        continue
                    line = child.stdout.readline()
                    if not line: raise RuntimeError('child pipe closed without final report')
                    data = json.loads(line)
                    if data['kind'] == 'optical_query':
                        if data['sequence'] != sequence or sequence >= round(seconds/.02):
                            raise ValueError('out-of-order/extra optical query')
                        if not 0 <= (time.time_ns()-data['created_unix_ns'])/1e9 <= 5:
                            raise RuntimeError('stale optical request')
                        dmd = bound_array(data, 'dmd', (600,600))
                        if not np.isin(dmd, (0,255)).all(): raise ValueError('nonbinary DMD')
                        rgb = bound_array(data, 'rgb')
                        self.camera.revalidate(self.exposure, self.requested_roi)
                        self.check_job(bundle, deadline)
                        self.publish(dict(kind='input', sequence=sequence, dmd=dmd, rgb=rgb,
                            phase=bundle.phase, simulation_time_s=data['simulation_time_s'],
                            position_m=data['position_m'], wall_seconds=time.monotonic()-started))
                        checked, began = self.project_and_settle(bundle, dmd, 'learned', deadline)
                        raw, meta = self.checked_capture(bundle, deadline, checked['verified_unix_ns'],
                            bundle.data['rules']['minimum_software_settle_seconds'])
                        values = np.einsum('khw,hw->k', bundle.weights, raw.astype(np.float64))
                        raw_path = folder/f'CCD_{sequence:04d}.npy'; np.save(raw_path, raw, allow_pickle=False)
                        write_new(raw_path.with_suffix('.json'), dict(frame=meta, camera_active=dict(self.camera.active),
                            display_verification=checked, raw_sha256=digest(raw_path), dmd_sha256=data['dmd_sha256'],
                            raw_integrals8=values.tolist(), frame_pattern_association_verified=False,
                            software_settle_NOT_hardware_sync=True))
                        # Registered ASM native pixels: preview only, never an actor fallback.
                        _, panel = patterns(settings, dmd, bundle.phase)
                        model = oriented(engine.simulate(dmd, panel, cfg), bundle.contract['registration']['orientation'])
                        matrix = np.asarray(bundle.contract['registration']['matrix_sensor'],float).copy()
                        roi = bundle.contract['roi']; matrix[:,2] -= [roi['OffsetX'], roi['OffsetY']]
                        ideal = cv2.warpAffine(model, matrix, (roi['Width'],roi['Height']), flags=cv2.INTER_LINEAR)
                        self.check_job(bundle, deadline)
                        pushes = self.pushes.take()
                        latency = time.monotonic()-began
                        write_new(folder/f'REPLY_{sequence:04d}.json', dict(sequence=sequence,
                            dmd_sha256=data['dmd_sha256'], raw_integrals8=values.tolist(),
                            query_latency_s=latency, push_events=pushes))
                        child.stdin.write((json.dumps(dict(sequence=sequence, dmd_sha256=data['dmd_sha256'],
                            raw_integrals8=values.tolist(), query_latency_s=latency, push_events=pushes))+'\n').encode())
                        child.stdin.flush()
                        np.save(folder/f'IDEAL_display_only_{sequence:04d}.npy', ideal, allow_pickle=False)
                        self.publish(dict(kind='optical', sequence=sequence, dmd=dmd, rgb=rgb,
                            phase=bundle.phase, actual=raw, ideal=ideal, raw_integrals8=values.tolist(),
                            frame_id=meta['FrameID'], logical_sequence=meta['logical_sequence'],
                            simulation_time_s=data['simulation_time_s'], query_latency_s=latency,
                            wall_seconds=time.monotonic()-started))
                        sequence += 1
                    elif data['kind'] == 'mujoco_step':
                        if data['step'] != report['actions'] or data['step'] != sequence-1:
                            raise ValueError('bad child step sequence')
                        data['rgb_image'] = bound_array(data, 'rgb')
                        self.publish(data); report['actions'] += 1
                    elif data['kind'] == 'pilot_complete':
                        if child.wait(timeout=3) != 0: raise RuntimeError('MuJoCo child failed: '+str(data['report']))
                        report.update(pilot=data['report'], completed=data['report']['completed'])
                        if data['report'].get('error'): raise RuntimeError(data['report']['error'])
                        break
                    else: raise ValueError('unknown child message')
        except (InterruptedError, StaticCancelled) as error:
            report.update(cancelled=True, error=str(error))
        except BaseException as error:
            fatal = error; report.update(error=str(error), traceback=traceback.format_exc())
        finally:
            report['unapplied_pushes'] = self.pushes.end()
            if child and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=2)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=2)
            if child:
                child.stdin.close(); child.stdout.close()
            try:
                if bundle and not self.stopped.is_set() and not self.static_cancel.is_set():
                    self.park_checked(bundle); report['actual_park_verified'] = True
                else: self.black.emit()
            except Exception as error:
                self.black.emit(); report['park_error'] = str(error); fatal = fatal or error
            report.update(wall_seconds=time.monotonic()-started,
                rejected_incomplete_frames=self.bad[initial_bad:],
                camera_node_writes_during_loop=len(self.camera.write_trace)-initial_writes,
                camera_active_at_end=dict(self.camera.active))
            write_new(folder/'loop.json', report)
            self.loop_result.emit(dict(folder=str(folder), report=report))
        if fatal: raise RuntimeError('interactive optical loop stopped: '+str(fatal)) from fatal
