"""Fail-closed CCD timeout evidence, installed only by an explicit GUI launch.

Import is offline. This module never opens an SDK, captures a replacement
frame, changes nodes or retries. Diagnostics run on the existing camera owner's
thread after the original GetNextImageEx timeout and before its normal cleanup.
"""
from functools import wraps
from pathlib import Path
import re
import threading
import time

from .storage import write_json_new


_SDK_TIMEOUT = re.compile(r'^spinCameraGetNextImageEx: -1011(?::|$)')
_NODES = ('AcquisitionStatusSelector', 'AcquisitionStatus', 'LineSelector',
          'LineMode', 'LineStatus', 'LineStatusAll', 'TriggerOverlap',
          'GevSCDA', 'GevSCPHostPort', 'GevHeartbeatTimeout')


def _observe(worker, error, dark):
    """Bounded set of read-only calls, retaining unsupported-node failures.

    A level snapshot is NOT an edge counter and may include GenApi-cached
    values. Zero stream counters cannot prove absence of electrical triggers.
    """
    report = dict(schema='optical-lab-capture-timeout-1',
        original_error=str(error), error_class=type(error).__name__,
        unix_ns=time.time_ns(), thread_id=threading.get_ident(),
        acquisition_epoch=getattr(worker, 'epoch', None),
        accepted_frames_in_worker=getattr(worker, 'count', None),
        startup_zero_input=bool(dark), cause='UNDETERMINED_NO_FRESH_IMAGE_DELIVERED',
        SDK_timeout_ms_UNCHANGED=1000, capture_retry_count=0,
        software_trigger_commands=0, node_writes=0,
        same_SDK_owner=True, black_output_requested=False,
        actual_black_framebuffer_NOT_verified_by_this_report=True,
        original_cleanup_and_rollback_required=True,
        telemetry_may_include_cached_node_values=True,
        line_snapshot_NOT_trigger_edge_measurement=True,
        no_automatic_free_run_fallback=True, release_approved=False,
        errors={}, nodes={})
    try:
        worker.black.emit()
        report['black_output_requested'] = True
    except Exception as failure:
        report['errors']['black_request'] = str(failure)
    camera = getattr(worker, 'camera', None)
    if camera is None:
        report['errors']['camera'] = 'existing camera owner unavailable'
        return report
    report['last_validated_active'] = dict(getattr(camera, 'active', {}) or {})
    for name, callback in (('actual_parameters', camera.parameters),
                           ('stream_statistics', camera.stream_statistics)):
        try:
            report[name] = callback()
        except Exception as failure:
            report['errors'][name] = str(failure)
    for name in _NODES:
        try:
            report['nodes'][name] = camera.read(name)
        except Exception as failure:
            report['errors'][name] = str(failure)
    return report


def _timeout_capture(original):
    """Decorate only the current explicit preview capture_one method."""
    @wraps(original)
    def capture(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except RuntimeError as error:
            if not _SDK_TIMEOUT.match(str(error)):
                raise
            # Capture was already attempted exactly once by the existing
            # implementation. Nothing below calls capture or restarts it.
            dark = kwargs.get('dark', args[0] if args else False)
            serial = getattr(self, '_capture_timeout_serial', 0) + 1
            self._capture_timeout_serial = serial
            path = Path(self.directory) / ('capture_timeout_%04d.json' % serial)
            try:
                report = _observe(self, error, dark)
            except Exception as failure:
                # A diagnostic programming/read failure must not replace the
                # original SDK error or prevent the original finally cleanup.
                report = dict(original_error=str(error),
                    diagnostics_error=str(failure), release_approved=False)
            try:
                write_json_new(path, report)
                evidence = '现场记录：' + str(path)
            except Exception as failure:
                evidence = '现场记录保存失败（不覆盖旧记录）：' + str(failure)
            message = (str(error) + '\nCCD在1秒内未收到新图像；这不是Gain差值误停。'
                '本轮停止并请求两路全0，保留硬触发，不自动重试或切自由运行。'
                '需检查外部触发与GigE传输后再用“仅CCD”复测。\n' + evidence)
            raise RuntimeError(message) from error
    capture._optical_timeout_diagnostics = True
    return capture


def attach_timeout_diagnostics(window):
    """Bind the one local WarningPreviewWorker, never frozen legacy classes."""
    import live_loop_v23.direct_alignment_gui as gui
    from live_loop_v23.worker import LoopWorker

    current = gui.LoopWorker
    if (window.worker is not None or not issubclass(current, LoopWorker)
            or current.__module__ != 'optical_lab.exposure'
            or current.__name__ != 'WarningPreviewWorker'):
        raise RuntimeError('timeout evidence requires the fresh local preview worker')
    if getattr(current.capture_one, '_optical_timeout_diagnostics', False):
        raise RuntimeError('timeout evidence already attached')
    current.capture_one = _timeout_capture(current.capture_one)
