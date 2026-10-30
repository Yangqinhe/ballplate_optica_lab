"""Read-only settling AFTER complete rollback, before final restoration check.

Requires three consecutive exact ORIGINAL-target snapshots. Does not retry
image retrieval, write a new target, normalize a changed float, or turn an
earlier failed trial into success. Timeout remains a restoration failure.
"""
import ctypes as C
import time
from live_manual_v19_trigger_r7.camera import TransactionalGainCamera


class SettledStandbyCamera(TransactionalGainCamera):
    cleanup_settle_seconds = 3.
    cleanup_sample_seconds = .15
    cleanup_exact_samples_required = 3

    def _cleanup_parameters(self):
        # Numeric AE/AGC outputs may update after their auto modes are restored.
        # Force fresh hardware reads rather than accepting write-side caches.
        fn = self.lib.spinNodeInvalidateNode
        fn.argtypes, fn.restype = [C.c_void_p], C.c_int
        for name in ('ExposureTime', 'Gain'):
            with self.node(name) as node:
                self.call('spinNodeInvalidateNode', node)
        return super().parameters()

    def parameters(self):
        if not getattr(self, 'cleaning', False) or not hasattr(self, 'before'):
            return super().parameters()
        until = time.monotonic()+self.cleanup_settle_seconds
        observations, consecutive = [], 0
        while True:
            actual = self._cleanup_parameters()
            different = {k: dict(expected=self.before.get(k), actual=actual.get(k))
                for k in self.before.keys() | actual.keys() if self.before.get(k) != actual.get(k)}
            consecutive = consecutive+1 if not different else 0
            observations.append(dict(unix_ns=time.time_ns(), differences=different,
                consecutive_exact=consecutive, values=actual))
            if consecutive >= self.cleanup_exact_samples_required or time.monotonic() >= until:
                break
            time.sleep(self.cleanup_sample_seconds)
        self.cleanup_readback_verification = dict(passed=consecutive >= self.cleanup_exact_samples_required,
            original_before_target_replaced=False, parameter_writes=0, image_retrievals=0,
            observations=observations, maximum_seconds=self.cleanup_settle_seconds,
            required_consecutive_exact=self.cleanup_exact_samples_required)
        return actual

    def close(self):
        if self.closed:
            return
        super().close()
        check = getattr(self, 'cleanup_readback_verification', None)
        if check is not None and not check['passed']:
            raise RuntimeError('bounded original-target exact standby stability failed; no relaxed acceptance')
