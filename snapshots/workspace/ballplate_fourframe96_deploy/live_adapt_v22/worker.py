"""Bounded jobs inside the existing Line0 SDK owner; simulator is a child."""
from dataclasses import replace
import json
from pathlib import Path
import select
import subprocess
import time
import traceback

import numpy as np
from PyQt5 import QtCore
from live_deploy_v20.worker import DeploymentWorker, StaticCancelled
from live_adapt_v22.core import digest, write_new, adapt_capture, predict
from realtime_model import LivePattern, patterns

PROJECT = Path(__file__).resolve().parents[1]
SIM_PYTHON = PROJECT.parent/'ballplate_sigle_cnn_project/.venv/bin/python'


class AdaptationWorker(DeploymentWorker):
    adaptation_progress = QtCore.pyqtSignal(object)
    adaptation_result = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_request = None
        self.adaptation_busy = False

    def request_adaptation(self, bundle, folder, mode='learned', head_path=None):
        if mode not in ('learned', 'flat'):
            raise ValueError('fixed learned or flat mask only')
        with self.command_lock:
            if (self.adaptation_request or self.adaptation_busy or self.static_request or self.static_busy
                    or self.configuration or self.reference_requested or self.spatial_requested
                    or self.auto_requested.is_set()):
                raise RuntimeError('another camera job is pending')
            if not self.camera or not self.camera.acquiring or self.startup_dark_required:
                raise RuntimeError('start the ONE existing CCD owner first')
            self.static_cancel.clear()
            self.adaptation_request = (bundle, Path(folder), mode, head_path)

    def request_static(self, *args, **kwargs):
        if self.adaptation_request or self.adaptation_busy:
            raise RuntimeError('single-frame adaptation owns the camera')
        return super().request_static(*args, **kwargs)

    def next_frame(self):
        pending = None
        with self.command_lock:
            if self.adaptation_request and not self.adaptation_busy:
                pending = self.adaptation_request; self.adaptation_request = None
        if pending:
            self.adaptation_busy = True
            try:
                self.run_adaptation(*pending)
            finally:
                self.adaptation_busy = False
        return super().next_frame()

    def check_job(self, bundle, deadline):
        self.check_stop()
        if self.static_cancel.is_set():
            raise StaticCancelled('operator cancelled; no old feature/action reuse')
        if time.monotonic() > deadline:
            raise TimeoutError('predeclared optical task time limit exceeded')
        if self.camera.active != bundle.contract['expected_active']:
            raise RuntimeError('frozen active camera contract changed')

    def checked_capture(self, bundle, deadline, verified_ns, settle):
        self.check_job(bundle, deadline)
        raw, meta = self.capture_one()
        if raw.max() >= 240 or meta['host_get_started_unix_ns'] < verified_ns+int(settle*1e9):
            raise RuntimeError('raw headroom/freshness criterion failed; do not retry')
        return raw, meta

    def project_and_settle(self, bundle, dmd_canvas, mode, deadline):
        self.check_job(bundle, deadline); bundle.verify()
        cfg = replace(LivePattern(**bundle.contract['settings']), dmd='file',
                      slm='file' if mode == 'learned' else 'zero', full_panel=False)
        dmd, slm = patterns(cfg, dmd_canvas, bundle.phase)
        started = time.monotonic()
        request = self.show('deployment_frame', dmd, slm)
        checked = request.get('video_verification', {})
        if not checked.get('actual_framebuffer_checked'):
            raise RuntimeError('actual DMD/SLM output readback required')
        settle = bundle.data['rules']['minimum_software_settle_seconds']
        until = time.monotonic()+settle
        while time.monotonic() < until:
            self.check_job(bundle, deadline); self.capture_one()
        return checked, started

    def park_checked(self, bundle):
        # Verify both all-zero images BEFORE releasing the inherited GUI override.
        cfg = LivePattern(**bundle.contract['settings'])
        dmd, slm = patterns(cfg, None, None)
        request = self.show('deployment_frame', np.zeros_like(dmd), np.zeros_like(slm))
        if not request.get('video_verification', {}).get('actual_framebuffer_checked'):
            raise RuntimeError('zero output could not be verified')
        self.show('deployment_park')

    def run_adaptation(self, bundle, folder, mode, head_path):
        folder.mkdir(parents=True, exist_ok=False)
        maximum = 600 if head_path else 300
        started = time.monotonic(); deadline = started+maximum
        initial_count = self.count; initial_bad = len(self.bad); initial_writes = len(self.camera.write_trace)
        report = dict(completed=False, error=None, inputs=[], phase_mode=mode,
                      real_actuators_enabled=False, same_SDK_owner=True, old_actor_loaded=False,
                      task='MuJoCo_slow_pilot' if head_path else 'real_single_frame_dataset',
                      release_approved=False, actual_park_verified=False)
        plan = dict(bundle=str(bundle.path), bundle_sha256=bundle.sha,
                    repeats=8, settle_seconds=bundle.data['rules']['minimum_software_settle_seconds'],
                    maximum_seconds=maximum, camera_active=dict(self.camera.active), phase_mode=mode,
                    rules=bundle.data['rules'], group_splits_predeclared=True,
                    image_NCC_NOT_a_gate_for_this_distinct_task=True,
                    old_static_failure_records_unchanged=True, CCD_history=0,
                    sources={str(Path(__file__).resolve()): digest(__file__)})
        write_new(folder/'PROSPECTIVE_PROTOCOL.json', plan)
        fatal = None
        try:
            bundle.verify(); self.camera.revalidate(self.exposure, self.requested_roi)
            self.check_job(bundle, deadline)
            if head_path:
                report['pilot'] = self.run_pilot(bundle, folder, mode, Path(head_path), deadline)
                report['completed'] = report['pilot']['completed'] and not report['pilot']['error']
            else:
                last_saved_id = -1
                for index, case in enumerate(bundle.cases):
                    checked, _ = self.project_and_settle(bundle, np.load(case['dmd'], allow_pickle=False), mode, deadline)
                    row = dict(id=case['id'], trajectory=case['trajectory'], split=case['split'],
                               display_verification=checked, frames=[])
                    report['inputs'].append(row)
                    for repeat in range(plan['repeats']):
                        raw, meta = self.checked_capture(bundle, deadline, checked['verified_unix_ns'], plan['settle_seconds'])
                        if meta['FrameID'] <= last_saved_id:
                            raise RuntimeError('saved FrameIDs must strictly increase')
                        last_saved_id = meta['FrameID']
                        relative = f"input_{case['id']:03d}_frame_{repeat:02d}.npy"
                        np.save(folder/relative, raw, allow_pickle=False)
                        row['frames'].append(dict(raw_relative=relative, raw_sha256=digest(folder/relative),
                            frame=meta, camera_active=dict(self.camera.active),
                            frame_pattern_association_verified=False, software_settle_NOT_hardware_sync=True))
                    self.adaptation_progress.emit(dict(kind='input', count=index+1, total=len(bundle.cases),
                        id=case['id'], split=case['split'], image=raw.copy(),
                        raw_integrals8=np.einsum('khw,hw->k', bundle.weights, raw.astype(np.float64)).tolist()))
                    # Atomic progress, preserve already saved raw data on failure.
                    temp = folder/'progress.tmp'
                    temp.write_text(json.dumps(report, ensure_ascii=False))
                    temp.replace(folder/'progress.json')
                report['completed'] = True
        except (InterruptedError, StaticCancelled) as error:
            report.update(cancelled=True, error=str(error))
        except BaseException as error:
            report.update(error=str(error), traceback=traceback.format_exc()); fatal = error
        finally:
            try:
                if self.stopped.is_set() or self.static_cancel.is_set():
                    self.black.emit(); self.show('deployment_park') if not self.stopped.is_set() else None
                else:
                    self.park_checked(bundle); report['actual_park_verified'] = True
            except Exception as error:
                self.black.emit(); report['park_error'] = str(error); fatal = fatal or error
            report.update(valid_stream_frames=self.count-initial_count,
                          rejected_incomplete_frames=self.bad[initial_bad:],
                          camera_node_writes_during_test=len(self.camera.write_trace)-initial_writes,
                          camera_active_at_end=dict(self.camera.active),
                          wall_seconds=time.monotonic()-started,
                          final_camera_restore_pending_normal_release=True)
            write_new(folder/'capture.json', report)
        result = dict(report=report, folder=str(folder.resolve()), head=None)
        if report['completed'] and not report['error'] and not head_path and not report.get('park_error'):
            try:
                result['head'] = adapt_capture(bundle, folder)
            except Exception as error:
                result['fit_error'] = str(error)
        self.adaptation_result.emit(result)
        if fatal is not None:
            raise RuntimeError('bounded adaptation stopped: '+str(fatal)) from fatal

    def run_pilot(self, bundle, folder, mode, head_path, deadline):
        from live_adapt_v22.pilot import __file__ as pilot_source
        head = json.loads(head_path.read_text())
        for filename, sha in head['source_sha256'].items():
            if digest(filename) != sha:
                raise ValueError('head fitting source changed')
        if (head.get('real_hardware_features') is not True
                or head['bundle_sha256'] != bundle.sha or not head['short_MuJoCo_pilot_eligible']
                or head['phase_mode'] != mode):
            raise ValueError('head failed held-out criteria or wrong mask/feature contract')
        # Re-audit full real capture before allowing the first simulator action.
        from live_adapt_v22.core import audit_capture
        audit_capture(bundle, head_path.parent)
        if digest(head_path.parent/'capture.json') != head['capture_sha256']:
            raise ValueError('head training capture changed')
        if digest(head_path.parent/'PROSPECTIVE_PROTOCOL.json') != head['prospective_protocol_sha256']:
            raise ValueError('head prospective collection protocol changed')
        pilot_dir = folder/'MuJoCo'
        command = [str(SIM_PYTHON), str(Path(pilot_source)), '--head', str(head_path.resolve()),
                   '--head-sha256', digest(head_path), '--output', str(pilot_dir.resolve())]
        write_new(folder/'CHILD_COMMAND.json', dict(command=command, no_real_actuator_interface=True,
                                                   child_source_sha256=digest(pilot_source)))
        # Binary, unbuffered pipe: select + readline cannot lose prefetched requests.
        with (folder/'simulator_stderr.log').open('xb') as err:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=err, bufsize=0, cwd=PROJECT)
            sequence = 0; done = None
            try:
                while done is None:
                    self.check_job(bundle, deadline)
                    if not select.select([child.stdout], [], [], .05)[0]:
                        if child.poll() is not None:
                            raise RuntimeError('MuJoCo child exited without a final report')
                        continue
                    line = child.stdout.readline()
                    if not line:
                        raise RuntimeError('MuJoCo child pipe closed unexpectedly')
                    data = json.loads(line)
                    if data['kind'] == 'optical_query':
                        if data['sequence'] != sequence or not 0 <= sequence < 150:
                            raise ValueError('repeated/out-of-order/extra optical request')
                        path = Path(data['dmd']).resolve()
                        if not path.is_relative_to(pilot_dir.resolve()) or digest(path) != data['dmd_sha256']:
                            raise ValueError('unbound simulator DMD file')
                        canvas = np.load(path, allow_pickle=False)
                        if (canvas.dtype != np.uint8 or canvas.shape != (600, 600)
                                or not np.isin(canvas, (0, 255)).all()):
                            raise ValueError('simulator DMD must be binary 600 square')
                        if not 0 <= (time.time_ns()-data['created_unix_ns'])/1e9 <= 5:
                            raise RuntimeError('expired simulator request')
                        checked, began = self.project_and_settle(bundle, canvas, mode, deadline)
                        raw, meta = self.checked_capture(bundle, deadline, checked['verified_unix_ns'],
                                                        bundle.data['rules']['minimum_software_settle_seconds'])
                        values = np.einsum('khw,hw->k', bundle.weights, raw.astype(np.float64))
                        # Retain RAW for EVERY action, not just screenshots/means.
                        raw_path = folder/f'pilot_CCD_{sequence:04d}.npy'
                        np.save(raw_path, raw, allow_pickle=False)
                        write_new(folder/f'pilot_CCD_{sequence:04d}.json', dict(frame=meta,
                            raw_sha256=digest(raw_path), camera_active=dict(self.camera.active),
                            display_verification=checked, frame_pattern_association_verified=False,
                            software_settle_NOT_hardware_sync=True))
                        reply = dict(sequence=sequence, dmd_sha256=data['dmd_sha256'],
                                     raw_integrals8=values.tolist(), query_latency_s=time.monotonic()-began)
                        child.stdin.write((json.dumps(reply)+'\n').encode()); child.stdin.flush()
                        sequence += 1
                    elif data['kind'] == 'mujoco_step':
                        self.adaptation_progress.emit(data)
                    elif data['kind'] == 'pilot_complete':
                        done = data['report']
                    else:
                        raise ValueError('unknown simulator IPC message')
                if child.wait(timeout=3) != 0:
                    raise RuntimeError('MuJoCo child returned failure: '+str(done.get('error')))
                return done
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait(timeout=2)
