"""Explicit temporary ROI support; the raw static diagnostic adapter is unchanged."""
import ctypes as C
import math
import time

from spin_c import Camera, B, H


class CalibrationCamera(Camera):
    def _bind(self):
        super()._bind()
        for method in ('Min', 'Max', 'Inc'):
            func = getattr(self.lib, 'spinIntegerGet'+method)
            func.argtypes = [H, C.POINTER(C.c_int64)]; func.restype = C.c_int
        self.lib.spinIntegerSetValue.argtypes = [H, C.c_int64]
        self.lib.spinIntegerSetValue.restype = C.c_int

    def limits(self, name):
        result = {}
        with self.node(name) as node:
            for method in ('Min', 'Max', 'Inc'):
                number = C.c_int64()
                self.call('spinIntegerGet'+method, node, C.byref(number))
                result[method.lower()] = number.value
        return result

    def integer(self, name, value):
        with self.node(name) as node:
            writable = B(); self.call('spinNodeIsWritable', node, C.byref(writable))
            if not writable.value:
                raise RuntimeError(name+' is not writable')
            self.call('spinIntegerSetValue', node, C.c_int64(value))
        if int(self.read(name)) != value:
            raise RuntimeError(name+' integer readback mismatch')

    def roi(self):
        return {name: int(self.read(name)) for name in ('OffsetX', 'OffsetY', 'Width', 'Height')}

    def _configure_roi(self, requested):
        if self.acquiring:
            raise RuntimeError('stop acquisition before ROI changes')
        if min(int(requested[k]) for k in ('OffsetX','OffsetY','Width','Height')) < 0:
            raise ValueError('ROI values must be nonnegative')
        for size, offset in (('Width', 'OffsetX'), ('Height', 'OffsetY')):
            self.integer(offset, 0)
            limit = self.limits(size)
            value = int(requested[size]) or limit['max']
            value = max(limit['min'], min(limit['max'], value))
            value = limit['min'] + (value-limit['min'])//limit['inc']*limit['inc']
            self.integer(size, value)
            limoff = self.limits(offset)
            off = int(requested[offset])
            off = limoff['min'] + (off-limoff['min'])//limoff['inc']*limoff['inc']
            if not limoff['min'] <= off <= limoff['max']:
                raise RuntimeError('ROI offset would clip the requested field')
            self.integer(offset, off)
        return self.roi()

    def set_roi(self, requested):
        """Pre-acquisition only; register whole-ROI rollback before first write."""
        if self.acquiring:
            raise RuntimeError('stop acquisition before ROI changes')
        if not hasattr(self, 'original_roi'):
            self.original_roi = self.roi()
        # Width/Height changes can silently clamp ExposureTime. Save the
        # original FULL-sensor value before the first ROI write, not later
        # in start_calibration(). Keep AE off until ROI AND exposure rollback.
        self.remember_exposure_rollback()
        try:
            actual = self._configure_roi(requested)
        except Exception:
            self._configure_roi(self.original_roi)
            raise
        return actual

    def remember_exposure_rollback(self):
        if getattr(self, '_exposure_rollback_registered', False):
            return
        if self.acquiring:
            raise RuntimeError('stop acquisition before freezing diagnostic exposure')
        target = self.before.get('ExposureTime')
        auto = self.before.get('ExposureAuto')
        if target is None or not math.isfinite(float(target)) or float(target) <= 0:
            raise RuntimeError('finite positive original ExposureTime required')
        if auto not in ('Off', 'Continuous') or self.read('ExposureAuto') != auto:
            raise RuntimeError('original ExposureAuto mode changed or unsupported')
        observed = self.read('ExposureTime')
        if observed is None or not math.isfinite(float(observed)) or float(observed) <= 0:
            raise RuntimeError('invalid pre-freeze ExposureTime')
        # A drift with manual exposure is not an automatic-exposure race and
        # must not be hidden by a write or an invented baseline.
        if auto == 'Off' and observed != target:
            raise RuntimeError('manual exposure changed before diagnostic ROI write; resnapshot required')
        self.exposure_startup_transaction = dict(
            original_ExposureAuto=auto, declared_original_ExposureTime=target,
            observed_before_freeze=observed, before_snapshot_replaced=False,
            original_target_reapplied=False, completed=False)
        # Register BOTH rollback entries before the first write, including a
        # partially successful AE-Off call. LIFO restores numeric exposure
        # while AE is Off, then restores the original AE mode.
        if auto != 'Off':
            self.undo.append((self.map, 'ExposureAuto', auto))
        self.undo.append((self.map, 'ExposureTime', target))
        if auto != 'Off':
            self._write('ExposureAuto', 'Off', self.map)
        if self.read('ExposureAuto') != 'Off':
            raise RuntimeError('ExposureAuto freeze readback mismatch')
        latched = self.read('ExposureTime')
        self.exposure_startup_transaction['observed_after_freeze'] = latched
        if latched is None or not math.isfinite(float(latched)) or float(latched) <= 0:
            raise RuntimeError('invalid latched ExposureTime')
        if latched != target:
            if auto == 'Off':
                raise RuntimeError('manual exposure changed during diagnostic snapshot')
            # AE may have advanced after the original SDK snapshot. Keep that
            # snapshot as the target; explicitly reapply it while AE is Off.
            # This is not a resnapshot, a tolerance relaxation, or acquisition.
            self._write('ExposureTime', target, self.map)
            self.exposure_startup_transaction['original_target_reapplied'] = True
        if self.read('ExposureTime') != target:
            raise RuntimeError('original ExposureTime exact startup readback mismatch')
        self._exposure_rollback_registered = True
        self.exposure_startup_transaction['completed'] = True

    def start_calibration(self, exposure_us):
        if not 50 <= exposure_us <= 4000:
            raise ValueError('explicit diagnostic exposure range is 50..4000 us')
        if self.before.get('TriggerMode') != 'Off':
            raise RuntimeError('this branch requires explicit free-run; it will NOT change trigger mode')
        self.remember_exposure_rollback()
        self.set_temporary('GainAuto', 'Off')
        self._write('ExposureTime', str(exposure_us), self.map)
        actual = float(self.read('ExposureTime'))
        if not math.isfinite(actual) or abs(actual-exposure_us) > 50:
            raise RuntimeError('exposure quantization exceeds 50us; no guessed setting accepted')
        self.set_temporary('Gain', '0')
        self.set_temporary('AcquisitionMode', 'Continuous')
        self.set_temporary('StreamBufferHandlingMode', 'NewestOnly', self.stream)
        if self.before.get('GevTimestampTickFrequency') != '125000000':
            raise RuntimeError('unknown packet delay clock')
        self.set_temporary('GevSCPD', str(max(1000, int(self.before['GevSCPD']))))
        self.active = self.parameters()
        self.call('spinCameraBeginAcquisition', self.camera); self.acquiring = True

    def close(self):
        errors = []
        if self.acquiring:
            try:
                self.call('spinCameraEndAcquisition', self.camera); self.acquiring = False
                # BFLY rolling/free-run stop may finish a sensor readout after
                # EndAcquisition returns. Keep the camera idle for >2 full-FOV
                # diagnostic frame intervals before restoring its parameters.
                # This is a diagnostic settle bound, NOT control-loop latency.
                time.sleep(.35)
            except Exception as error:
                errors.append('EndAcquisition: '+str(error))
        # Restore ROI BEFORE exposure: exposure quantization may depend on ROI.
        if self.initialized and hasattr(self, 'original_roi'):
            try:
                if self._configure_roi(self.original_roi) != self.original_roi:
                    raise RuntimeError('original ROI was not exactly restored')
            except Exception as error:
                errors.append('ROI rollback: '+str(error))
        try:
            super().close()
        except Exception as error:
            errors.append('SDK cleanup: '+str(error))
        if errors:
            raise RuntimeError('; '.join(errors))
