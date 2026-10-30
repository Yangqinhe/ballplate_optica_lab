"""Manual exposure-only live transaction; no new camera, restart or retries.

Initial startup, ROI changes and quantitative jobs must retain the original
reconfigure path. Import is offline. apply() belongs to the existing SDK owner
thread, after the caller's qualifies() check; no node work occurs in the GUI.
"""
import ctypes as C
import math
from pathlib import Path
import threading

import numpy as np

from .storage import sha256, write_json_new


_TASKS = ('reference_requested', 'spatial_requested', 'static_request', 'static_busy',
          'adaptation_request', 'adaptation_busy', 'loop_request', 'loop_busy')


def qualifies(worker, roi, *, initial):
    """Pure eligibility only; False means keep the old reconfigure route."""
    camera = getattr(worker, 'camera', None)
    return bool(initial is False and camera is not None and camera.acquiring
        and dict(roi) == worker.requested_roi
        and not worker.startup_dark_required
        and not getattr(worker, '_preview_strict_depth', 0)
        and not getattr(worker, '_manual_exposure_busy', False)
        and not getattr(worker, '_manual_gain_busy', False)
        and getattr(worker, '_manual_gain_pending', None) is None
        and not any(getattr(worker, key, None) for key in _TASKS)
        and not (getattr(worker, 'auto_requested', None) and worker.auto_requested.is_set()))


def _payload(worker, requested, accepted, reason, path):
    worker.configured.emit(dict(actual_roi=dict(worker.requested_roi),
        requested_roi=dict(worker.requested_roi), active=dict(worker.camera.active),
        manual_exposure_live=dict(requested_us=requested,
            retained_requested_us=worker.exposure, accepted=accepted, reason=reason,
            actual_us=float(worker.camera.active['ExposureTime']),
            transaction_path=str(path))))


def _compare(before, after, *, allow_gain_drift):
    ignored = {'ExposureTime'} | ({'Gain'} if allow_gain_drift else set())
    changed = {key for key in before.keys() | after.keys()
               if key not in ignored and before.get(key) != after.get(key)}
    if changed:
        raise RuntimeError('live曝光中其他相机参数改变：'+','.join(sorted(changed)))
    if (not math.isfinite(float(after['Gain'])) or after.get('GainAuto') != 'Off'
            or after.get('ExposureAuto') != 'Off'):
        raise RuntimeError('live曝光的Gain／自动模式读回无效')


def apply(worker, exposure_us, *, resume=True):
    """Live-write one writable ExposureTime; failures after writes are fatal.

    A non-writable node (or missing original undo) rejects this request only,
    with feedback and normal resume. Never silently switch to End/Begin.
    The three preserved frames are subsequent fresh returns in the SAME epoch,
    not certified associations with a specific physical exposure transition.
    """
    requested = float(exposure_us)
    if not math.isfinite(requested) or not 50 <= requested <= 4000:
        raise ValueError('手动live曝光请求必须为50..4000µs有限值')
    with worker._manual_gain_barrier:
        if not qualifies(worker, worker.requested_roi, initial=False):
            raise RuntimeError('手动live曝光资格改变；未写相机')
        worker._manual_exposure_busy = True
    serial = getattr(worker, '_manual_exposure_serial', 0) + 1
    worker._manual_exposure_serial = serial
    folder = Path(worker.directory)/('manual_exposure_%04d' % serial)
    report = dict(schema='optical-lab-manual-live-exposure-1', requested_us=requested,
        previous_requested_us=worker.exposure, completed=False, request_rejected=False,
        frames=[], single_SDK_owner=True, thread_id=threading.get_ident(),
        acquisition_epoch=worker.epoch, acquisition_restarted=False,
        no_FrameID_or_timestamp_reset=True, original_rollback_preserved=True,
        quantitative_valid=False, frame_exposure_association_verified=False,
        epoch_NOT_changed=True, changes=['ExposureTime'], capture_retry_count=0,
        no_automatic_free_run_or_Software_fallback=True)
    path = folder/'transaction.json'
    created = False
    fatal = None
    feedback = None
    try:
        folder.mkdir(exist_ok=False)
        created = True
        worker.check_stop()
        worker.show('hold_black')
        camera = worker.camera
        camera.revalidate(worker.exposure, worker.requested_roi)
        before = dict(camera.parameters()); report['before'] = before
        report['before_cached_active'] = dict(camera.active)
        allow_gain = bool(getattr(camera, 'manual_gain_applied', False))
        _compare(dict(camera.active), before, allow_gain_drift=allow_gain)
        undo = tuple(camera.undo)
        saved = bool(getattr(camera, '_exposure_rollback_registered', False)
            and any(name == 'ExposureTime' and value == camera.before['ExposureTime']
                    for _, name, value in undo))
        report['original_exposure_undo_present'] = saved
        writable = C.c_uint8()
        with camera.node('ExposureTime') as node:
            camera.call('spinNodeIsWritable', node, C.byref(writable))
        report['SDK_writable'] = bool(writable.value)
        if not saved or not writable.value:
            reason = '缺少原始曝光回退，拒绝写入' if not saved else '采集中ExposureTime不可写，拒绝本次请求；不重启CCD'
            report.update(request_rejected=True, rejection_reason=reason,
                          after=dict(camera.active), completed=True)
            feedback = (False, reason)
            return False
        report['write_attempted'] = True
        camera._write('ExposureTime', str(requested), camera.map)
        after = dict(camera.parameters())
        report['immediate_readback'] = after
        _compare(before, after, allow_gain_drift=allow_gain)
        actual = float(after['ExposureTime'])
        if not math.isfinite(actual) or abs(actual-requested) > 50:
            raise RuntimeError('live曝光实际读回无效／超过原50µs量化范围')
        contract = getattr(camera, 'manual_gain_contract', None)
        if contract is not None:
            report['manual_gain_contract_before'] = dict(contract)
            camera.manual_gain_contract = dict(contract, ExposureTime=after['ExposureTime'])
            report['manual_gain_contract_after'] = dict(camera.manual_gain_contract)
        camera.revalidate(requested, worker.requested_roi)
        worker.exposure = requested
        worker.next_contract_check = 0.
        for index in range(3):
            raw, meta = worker.capture_one(dark=False)
            raw_path = folder/('transition_%d.npy' % index)
            with raw_path.open('xb') as stream: np.save(stream, raw, allow_pickle=False)
            report['frames'].append(dict(raw_path=str(raw_path), sha256=sha256(raw_path),
                frame=meta, active=dict(camera.active),
                first_returned_frame_NOT_exact_exposure_transition=True))
        final = dict(camera.revalidate(requested, worker.requested_roi))
        _compare(before, final, allow_gain_drift=allow_gain)
        if tuple(camera.undo) != undo:
            raise RuntimeError('live曝光原始回退链改变')
        report.update(after=final, actual_us=float(final['ExposureTime']), completed=True,
                      undo_entries_unchanged=True)
        feedback = (True, '曝光实际 %.4f µs（请求 %.4f）；未重启CCD，3张新原图已保留。' %
                          (report['actual_us'], requested))
        return True
    except BaseException as error:
        fatal = error
        report.update(completed=False, error=str(error))
        worker.black.emit()
        raise
    finally:
        try:
            if created:
                try:
                    write_json_new(path, report)
                except Exception as error:
                    worker._manual_exposure_report_error = str(error)
                    if fatal is None:
                        worker.black.emit()
                        raise  # A successful transaction cannot resume without evidence.
                    try:
                        worker.message.emit('曝光事务记录保存失败，保留原始采集／SDK错误：'+str(error))
                    except Exception:
                        pass  # Secondary logging must never mask the original SDK error.
            if fatal is None and feedback is not None:
                accepted, reason = feedback
                _payload(worker, requested, accepted, reason, path)
                worker.message.emit(reason)
                if resume: worker.show('resume')
        finally:
            worker._manual_exposure_busy = False


def attach_feedback(window):
    """Original configured callback plus request rollback for manual rejection."""
    if window.worker is not None or getattr(window, '_manual_exposure_feedback_attached', False):
        raise RuntimeError('live曝光反馈只在新窗口启动前安装一次')
    original = window.camera_configured

    def configured(payload):
        original(payload)
        manual = payload.get('manual_exposure_live')
        if manual is not None and not manual['accepted']:
            previous = window.exposure.blockSignals(True)
            try: window.exposure.setValue(manual['retained_requested_us'])
            finally: window.exposure.blockSignals(previous)

    window.camera_configured = configured
    window._manual_exposure_feedback_attached = True
