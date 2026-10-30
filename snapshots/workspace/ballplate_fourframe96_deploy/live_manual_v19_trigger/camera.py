"""Line0-only adapter, based on the tested transactional trigger probe.

The GUI worker, not a permissive held-black monitor, owns the guard here.
Normal acquisition may project patterns; configuration is performed under
the GUI's existing hold-black handshake. No software/free-run fallback.
"""
import math
from pathlib import Path
import re
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'tools'))
from probe_line0_trigger_v18 import Line0Camera, TRIGGER_EXPECTED


def validate_contract(active, exposure_us, roi, source='Line0'):
    if source not in ('Line0', 'Software'):
        raise ValueError('只允许Line0或显式无命令负对照')
    expected = dict(TRIGGER_EXPECTED, DeviceSerialNumber='15328133', PixelFormat='Mono8',
                    ExposureMode='Timed', ExposureAuto='Off', GainAuto='Off', AcquisitionMode='Continuous',
                    TriggerSource=source)
    for key, value in expected.items():
        if active.get(key) != value:
            raise RuntimeError('硬触发参数不匹配：'+key)
    if active.get('TriggerDelayEnabled') not in ('0', 'false', 'False'):
        raise RuntimeError('触发延迟意外开启；不自动猜测或修正')
    actual = float(active['ExposureTime'])
    if not math.isfinite(actual) or abs(actual-exposure_us) > 50 or float(active['Gain']) != 0:
        raise RuntimeError('硬触发曝光／增益不匹配')
    if any(int(active[key]) != value for key, value in roi.items()):
        raise RuntimeError('硬触发ROI实际读回不匹配')


class WorkerGuard:
    """Worker cancellation/video identity guard; NOT a physical beam interlock."""
    def __init__(self, check):
        self.callback = check

    def check(self, force=False):
        self.callback()


class HardwareTriggerCamera(Line0Camera):
    def start_calibration(self, exposure_us):
        self.start_source(exposure_us, 'Line0')

    def start_source(self, exposure_us, source):
        if source not in ('Line0', 'Software'):
            raise ValueError('未声明的触发源')
        if not 50 <= exposure_us <= 4000 or not math.isfinite(exposure_us):
            raise ValueError('显式诊断曝光范围50..4000µs')
        if self.acquiring:
            raise RuntimeError('先EndAcquisition，再重新武装硬触发')
        self.remember_exposure_rollback()
        self.set_temporary('GainAuto', 'Off')
        self.set_temporary('Gain', '0')
        self.set_temporary('AcquisitionMode', 'Continuous')
        self.set_temporary('StreamBufferHandlingMode', 'NewestOnly', self.stream)
        if self.before.get('GevTimestampTickFrequency') != '125000000':
            raise RuntimeError('未知GigE间隔时钟；不猜测传输设置')
        self.set_temporary('GevSCPD', str(max(1000, int(self.before['GevSCPD']))))
        self.configure_trigger()  # Off -> FrameStart/Line0/FallingEdge -> On -> Rolling.
        if source == 'Software':
            self._write('TriggerMode', 'Off', self.map)
            self._write('TriggerSource', 'Software', self.map)
            self._write('TriggerMode', 'On', self.map)
            self._write('SensorShutterMode', 'Rolling', self.map)
        self._write('ExposureTime', str(exposure_us), self.map)
        roi = self.roi()
        validate_contract(self.parameters(), exposure_us, roi, source)
        self.guard.check(force=True)
        self.call('spinCameraBeginAcquisition', self.camera)
        self.acquiring = True
        self.active = self.parameters()
        validate_contract(self.active, exposure_us, roi, source)

    def qualify_no_command(self, exposure_us):
        """Held-black startup self-test, matching the accepted single-owner sequence."""
        self.start_source(exposure_us, 'Software')
        started = time.monotonic()
        result = dict(passed=False, TriggerSoftware_commands=0, expected_timeout_ms=1000,
                      active=dict(self.active), automatic_free_run_fallback=False)
        try:
            image, meta = self.capture(timeout_ms=1000)
            raise UnexpectedNoCommandImage(image, meta)
        except UnexpectedNoCommandImage:
            raise
        except RuntimeError as error:
            elapsed = time.monotonic()-started
            if not (re.match(r'^spinCameraGetNextImageEx: -1011:', str(error)) and .8 <= elapsed <= 3):
                raise
            result.update(passed=True, error=str(error), elapsed_seconds=elapsed)
            return result
        finally:
            self.call('spinCameraEndAcquisition', self.camera)
            self.acquiring = False
            time.sleep(.35)
            self.last_frame_id = None

    def revalidate(self, exposure_us, roi):
        active = self.parameters()
        validate_contract(active, exposure_us, roi)
        if active['BlackLevel'] != self.before['BlackLevel']:
            raise RuntimeError('黑电平在采集中改变；停止而不是重写')
        self.active = active
        return active


class UnexpectedNoCommandImage(RuntimeError):
    def __init__(self, image, meta):
        super().__init__('Software源未发命令却收到图像；触发门控未通过')
        self.image, self.meta = image, meta
