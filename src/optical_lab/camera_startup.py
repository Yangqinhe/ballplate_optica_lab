"""Candidate Line0 exposure-write ordering, not a demonstrated root-cause fix.

Pure factory: import/creation never connects a camera. Intended base is the
preserved OrderedTriggerCamera/StreamingCamera path, whose trigger snapshot,
strict validation, SDK ownership and cleanup remain authoritative. There is no
capture retry, extra Begin, sleep, timeout change or free-run/Software fallback.
"""


def make_exposure_rearm_camera(base):
    """During Line0 start only, write Off -> ExposureTime -> On -> Rolling.

    The original start_source still performs final validation and its single
    BeginAcquisition. Original close restores the trigger snapshot even if any
    part of this sequence fails. Cleanup and explicit Software-source writes
    traverse the unmodified base behavior.
    """
    class ExposureRearmCamera(base):
        def start_source(self, exposure_us, source):
            if source == 'Line0' and self.acquiring:
                raise RuntimeError('Line0曝光重武装前必须停止采集')
            previous = getattr(self, '_line0_exposure_rearm_scope', False)
            self._line0_exposure_rearm_scope = source == 'Line0'
            try:
                return super().start_source(exposure_us, source)
            finally:
                self._line0_exposure_rearm_scope = previous

        def _write(self, name, value, node_map):
            if (name != 'ExposureTime' or getattr(self, 'cleaning', False)
                    or not getattr(self, '_line0_exposure_rearm_scope', False)):
                return super()._write(name, value, node_map)
            if self.acquiring:
                raise RuntimeError('采集过程中不允许执行曝光重武装')
            # Use the unchanged guarded _write implementation for all four
            # writes; do not bypass its per-write guard or readback trace.
            super()._write('TriggerMode', 'Off', self.map)
            result = super()._write(name, value, node_map)
            super()._write('TriggerMode', 'On', self.map)
            # On can implicitly alter this model's shutter mode. Reassert
            # Rolling exactly as in the accepted original trigger adapter.
            super()._write('SensorShutterMode', 'Rolling', self.map)
            return result

    return ExposureRearmCamera
