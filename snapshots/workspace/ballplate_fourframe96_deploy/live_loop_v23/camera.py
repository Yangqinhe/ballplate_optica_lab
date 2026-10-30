"""Additive capture adapter. Original SDK/trigger/rollback code stays frozen."""
import ctypes as C
import time
import numpy as np
from live_manual_v19_trigger_r2.camera import OrderedTriggerCamera
from spin_c import H, U, B, N, IncompleteFrame
from .core import FrameOrder


class StreamingCamera(OrderedTriggerCamera):
    def start_source(self, *args, **kwargs):
        super().start_source(*args, **kwargs)
        self.frame_order = FrameOrder()

    def capture(self, timeout_ms=1000):
        # Same raw C ABI/buffer validation and finally-release as frozen v17,
        # with raw counter + logical sequence instead of raw-id <= rejection.
        if not self.acquiring: raise RuntimeError('camera is not acquiring')
        image = H(); started = time.time_ns()
        self.call('spinCameraGetNextImageEx', self.camera, U(timeout_ms), C.byref(image))
        try:
            incomplete = B(); self.call('spinImageIsIncomplete', image, C.byref(incomplete))
            if incomplete.value:
                status = C.c_int(); self.call('spinImageGetStatus', image, C.byref(status))
                description = C.create_string_buffer(1024); size = N(len(description))
                self.call('spinImageGetStatusDescription', status, description, C.byref(size))
                fid = U(); self.call('spinImageGetFrameID', image, C.byref(fid))
                raise IncompleteFrame(dict(frame_id=fid.value, status=status.value,
                    description=description.value.decode(errors='replace'), host_received_unix_ns=time.time_ns()))
            name = C.create_string_buffer(128); length = N(len(name))
            self.call('spinImageGetPixelFormatName', image, name, C.byref(length))
            if name.value != b'Mono8': raise RuntimeError('returned frame is not Mono8')
            values = {}
            for key in ('Width', 'Height', 'Stride', 'BufferSize', 'FrameID', 'TimeStamp'):
                value = U() if key in ('FrameID', 'TimeStamp') else N()
                self.call('spinImageGet'+key, image, C.byref(value)); values[key] = int(value.value)
            w, h, stride, size = (values[k] for k in ('Width', 'Height', 'Stride', 'BufferSize'))
            if not (0 < w <= 10000 and 0 < h <= 10000 and stride >= w
                    and (h-1)*stride+w <= size <= 100000000):
                raise RuntimeError('unsafe SDK buffer geometry')
            pointer = H(); self.call('spinImageGetData', image, C.byref(pointer))
            if not pointer: raise RuntimeError('empty image data pointer')
            buffer = (C.c_uint8*size).from_address(pointer.value)
            array = np.ndarray((h, w), np.uint8, buffer=buffer, strides=(stride, 1)).copy()
            values.update(self.frame_order.accept(values['FrameID'], values['TimeStamp']))
            self.last_frame_id = values['FrameID']
            values.update(host_get_started_unix_ns=started, host_received_unix_ns=time.time_ns(),
                          camera_clock_relationship_verified=False, pixel_format='Mono8')
            return array, values
        finally:
            if image: self.call('spinImageRelease', image)
