"""Freeze Gain against the ORIGINAL snapshot before ROI or auto-mode writes.

Matches the existing lossless ExposureTime rollback transaction. Does not
retry frames, change the Line0 arming sequence, relax restoration equality,
or change any loaded original GUI module.
"""
import math
from live_manual_v19_trigger_r2.camera import OrderedTriggerCamera


class TransactionalGainCamera(OrderedTriggerCamera):
    def remember_gain_rollback(self):
        if getattr(self, '_gain_rollback_registered', False):
            return
        if self.acquiring:
            raise RuntimeError('stop acquisition before gain snapshot freeze')
        target, auto = self.before.get('Gain'), self.before.get('GainAuto')
        if target is None or not math.isfinite(float(target)) or auto not in ('Off', 'Continuous'):
            raise RuntimeError('finite original Gain and explicit supported auto mode required')
        if self.read('GainAuto') != auto:
            raise RuntimeError('original GainAuto mode changed')
        observed = self.read('Gain')
        if observed is None or not math.isfinite(float(observed)):
            raise RuntimeError('invalid observed Gain')
        if auto == 'Off' and observed != target:
            raise RuntimeError('manual Gain changed before freeze; original baseline not replaced')
        self.gain_startup_transaction = dict(original_GainAuto=auto,
            declared_original_Gain=target, observed_before_freeze=observed,
            before_snapshot_replaced=False, original_target_reapplied=False, completed=False)
        # Register BEFORE any write, including a partial GainAuto-Off failure.
        if auto != 'Off':
            self.undo.append((self.map, 'GainAuto', auto))
        self.undo.append((self.map, 'Gain', target))
        if auto != 'Off':
            self._write('GainAuto', 'Off', self.map)
        if self.read('GainAuto') != 'Off':
            raise RuntimeError('GainAuto freeze readback mismatch')
        latched = self.read('Gain')
        self.gain_startup_transaction['observed_after_freeze'] = latched
        if latched is None or not math.isfinite(float(latched)):
            raise RuntimeError('invalid latched Gain')
        if latched != target:
            if auto == 'Off':
                raise RuntimeError('manual Gain changed during freeze')
            self._write('Gain', target, self.map)
            self.gain_startup_transaction['original_target_reapplied'] = True
        if self.read('Gain') != target:
            raise RuntimeError('original Gain exact readback mismatch')
        self._gain_rollback_registered = True
        self.gain_startup_transaction['completed'] = True

    def set_roi(self, requested):
        self.remember_gain_rollback()
        return super().set_roi(requested)

    def start_source(self, exposure_us, source):
        self.remember_gain_rollback()
        return super().start_source(exposure_us, source)
