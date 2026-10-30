"""Minimal original Spinnaker C ABI adapter, not a replacement named PySpin.

One process owns both Qt output and this camera. No ForceIP, EEPROM,
UserSetSave, ROI, trigger, shutter or black-level writes in diagnostics.
Exposure/gain are temporarily fixed, recorded, then restored.
"""
import ctypes as C
from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np

H=C.c_void_p; P=C.POINTER; N=C.c_size_t; U=C.c_uint64; B=C.c_uint8
LIBRARY=Path('/opt/spinnaker/lib/libSpinnaker_C.so.4.4.0.246')


class IncompleteFrame(RuntimeError):
    def __init__(self,record):
        self.record=record
        super().__init__('incomplete GigE frame: '+str(record))


class Camera:
    def __init__(self,serial='15328133'):
        self.serial=serial;self.system=H();self.cameras=H();self.camera=H()
        self.map=H();self.stream=H();self.initialized=False;self.acquiring=False
        self.undo=[];self.last_frame_id=None;self.lock=None;self.closed=False
        running=subprocess.run(['ps','-C','SpinView_QT','-o','pid='],capture_output=True,text=True)
        if running.stdout.strip():raise RuntimeError('SpinView is running: close it before this single-owner diagnostic')
        self.lock=open(f'/run/user/{os.getuid()}/ballplate_optical_live_{serial}.lock','a')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close();self.lock=None;raise RuntimeError('another optical session owns the camera lock')
        self.lib=C.CDLL(str(LIBRARY));self._bind()
        try:
            self.call('spinSystemGetInstance',C.byref(self.system))
            self.call('spinCameraListCreateEmpty',C.byref(self.cameras))
            self.call('spinSystemGetCameras',self.system,self.cameras)
            self.call('spinCameraListGetBySerial',self.cameras,serial.encode(),C.byref(self.camera))
            tl=H();self.call('spinCameraGetTLDeviceNodeMap',self.camera,C.byref(tl))
            if self.read('DeviceSerialNumber',tl)!=serial:raise RuntimeError('camera serial mismatch')
            self.call('spinCameraInit',self.camera);self.initialized=True
            self.call('spinCameraGetNodeMap',self.camera,C.byref(self.map))
            self.call('spinCameraGetTLStreamNodeMap',self.camera,C.byref(self.stream))
            self.before=self.parameters()
            if self.before.get('PixelFormat')!='Mono8':raise RuntimeError('explicit Mono8 required; will not silently change pixel format')
        except BaseException:
            self.close();raise

    def _bind(self):
        specs={
            'spinErrorGetLastMessage':[C.c_char_p,P(N)],
            'spinSystemGetInstance':[P(H)],'spinSystemReleaseInstance':[H],
            'spinCameraListCreateEmpty':[P(H)],'spinCameraListDestroy':[H],
            'spinSystemGetCameras':[H,H],'spinCameraListClear':[H],
            'spinCameraListGetBySerial':[H,C.c_char_p,P(H)],
            'spinCameraInit':[H],'spinCameraDeInit':[H],'spinCameraRelease':[H],
            'spinCameraGetNodeMap':[H,P(H)],'spinCameraGetTLDeviceNodeMap':[H,P(H)],
            'spinCameraGetTLStreamNodeMap':[H,P(H)],
            'spinNodeMapGetNode':[H,C.c_char_p,P(H)],'spinNodeMapReleaseNode':[H,H],
            'spinNodeIsReadable':[H,P(B)],'spinNodeIsWritable':[H,P(B)],
            'spinNodeToString':[H,C.c_char_p,P(N)],'spinNodeFromString':[H,C.c_char_p],
            'spinFloatGetValue':[H,P(C.c_double)],'spinFloatSetValue':[H,C.c_double],
            'spinCameraBeginAcquisition':[H],'spinCameraEndAcquisition':[H],
            'spinCameraGetNextImageEx':[H,U,P(H)],'spinImageRelease':[H],
            'spinImageIsIncomplete':[H,P(B)],'spinImageGetData':[H,P(H)],
            'spinImageGetStatus':[H,P(C.c_int)],'spinImageGetStatusDescription':[C.c_int,C.c_char_p,P(N)],
            'spinImageGetPixelFormatName':[H,C.c_char_p,P(N)],
        }
        for name in ('Width','Height','Stride','BufferSize'):specs['spinImageGet'+name]=[H,P(N)]
        for name in ('FrameID','TimeStamp'):specs['spinImageGet'+name]=[H,P(U)]
        for name,types in specs.items():
            function=getattr(self.lib,name);function.argtypes=types;function.restype=C.c_int

    def call(self,name,*args):
        code=getattr(self.lib,name)(*args)
        if code:
            text=C.create_string_buffer(4096);size=N(len(text))
            self.lib.spinErrorGetLastMessage(text,C.byref(size))
            raise RuntimeError(f'{name}: {code}: {text.value.decode(errors="replace")}')

    @contextmanager
    def node(self,name,map=None):
        owner=self.map if map is None else map;handle=H()
        self.call('spinNodeMapGetNode',owner,name.encode(),C.byref(handle))
        try:yield handle
        finally:
            if handle:self.call('spinNodeMapReleaseNode',owner,handle)

    def read(self,name,map=None):
        with self.node(name,map) as node:
            valid=B();self.call('spinNodeIsReadable',node,C.byref(valid))
            if not valid.value:return None
            # GenApi ToString rounds the text, which can restore a float one
            # hardware increment low. Snapshot numeric values losslessly.
            if name in ('ExposureTime','Gain','BlackLevel'):
                value=C.c_double();self.call('spinFloatGetValue',node,C.byref(value))
                return repr(value.value)
            text=C.create_string_buffer(4096);length=N(len(text))
            self.call('spinNodeToString',node,text,C.byref(length))
            return text.value.decode(errors='replace')

    def set_temporary(self,name,value,map=None):
        if name not in ('ExposureAuto','GainAuto','AcquisitionMode','StreamBufferHandlingMode','ExposureTime','Gain','GevSCPD'):
            raise ValueError('diagnostic write whitelist excludes ROI, trigger and shutter changes')
        owner=self.map if map is None else map;previous=self.read(name,owner)
        if previous==value:return
        if previous is None:raise RuntimeError(f'cannot snapshot {name}')
        # Register rollback BEFORE writing, including partial failures.
        self.undo.append((owner,name,previous))
        self._write(name,value,owner)
        actual=self.read(name,owner)
        equal=(math.isclose(float(actual),float(value),rel_tol=.01,abs_tol=.001)
               if name in ('ExposureTime','Gain','BlackLevel') else actual==value)
        if not equal:raise RuntimeError(f'{name} readback differs: requested {value}, actual {actual}')

    def _write(self,name,value,map):
        with self.node(name,map) as node:
            valid=B();self.call('spinNodeIsWritable',node,C.byref(valid))
            if not valid.value:raise RuntimeError(f'{name} is not writable')
            if name in ('ExposureTime','Gain','BlackLevel'):
                self.call('spinFloatSetValue',node,C.c_double(float(value)))
            else:self.call('spinNodeFromString',node,str(value).encode())

    def parameters(self):
        names=('DeviceModelName','DeviceSerialNumber','PixelFormat','OffsetX','OffsetY','Width','Height',
               'ExposureMode','ExposureAuto','ExposureTime','GainAuto','Gain','BlackLevel','SensorShutterMode',
               'TriggerMode','TriggerSelector','TriggerSource','TriggerActivation','TriggerDelayEnabled',
               'TriggerDelay','AcquisitionMode','AcquisitionFrameRate','AcquisitionFrameRateEnable','GevSCPSPacketSize',
               'GevSCPD','GevTimestampTickFrequency')
        result={}
        for name in names:
            try:result[name]=self.read(name)
            except RuntimeError:result[name]=None
        return result

    def start(self):
        self.set_temporary('ExposureAuto','Off');self.set_temporary('GainAuto','Off')
        self.set_temporary('ExposureTime',str(min(2000.,float(self.before['ExposureTime']))))
        self.set_temporary('Gain','0')
        self.set_temporary('AcquisitionMode','Continuous')
        self.set_temporary('StreamBufferHandlingMode','NewestOnly',self.stream)
        # 1000 / 125MHz = 8us packet spacing. This reduces burst pressure
        # without changing any host sysctl, NIC MTU, ROI or trigger settings.
        # Full 5MP / 13FPS remains a diagnostic, not a realtime qualification.
        if self.before.get('GevTimestampTickFrequency')!='125000000':
            raise RuntimeError('unknown packet-delay clock; no guessed transport configuration')
        self.set_temporary('GevSCPD',str(max(1000,int(self.before['GevSCPD']))))
        self.active=self.parameters()
        self.call('spinCameraBeginAcquisition',self.camera);self.acquiring=True

    def capture(self,timeout_ms=1000):
        if not self.acquiring:raise RuntimeError('camera is not acquiring')
        image=H();start=time.time_ns()
        self.call('spinCameraGetNextImageEx',self.camera,U(timeout_ms),C.byref(image))
        try:
            incomplete=B();self.call('spinImageIsIncomplete',image,C.byref(incomplete))
            if incomplete.value:
                status=C.c_int();self.call('spinImageGetStatus',image,C.byref(status))
                description=C.create_string_buffer(1024);size=N(len(description))
                self.call('spinImageGetStatusDescription',status,description,C.byref(size))
                frame_id=U();self.call('spinImageGetFrameID',image,C.byref(frame_id))
                raise IncompleteFrame(dict(frame_id=frame_id.value,status=status.value,
                    description=description.value.decode(errors='replace'),host_received_unix_ns=time.time_ns()))
            text=C.create_string_buffer(128);length=N(len(text))
            self.call('spinImageGetPixelFormatName',image,text,C.byref(length))
            if text.value!=b'Mono8':raise RuntimeError('returned frame is not Mono8')
            values={}
            for name in ('Width','Height','Stride','BufferSize','FrameID','TimeStamp'):
                value=U() if name in ('FrameID','TimeStamp') else N()
                self.call('spinImageGet'+name,image,C.byref(value));values[name]=int(value.value)
            w,h,stride,size=(values[k] for k in ('Width','Height','Stride','BufferSize'))
            if not (0<w<=10000 and 0<h<=10000 and stride>=w and (h-1)*stride+w<=size<=100000000):
                raise RuntimeError('unsafe SDK buffer geometry')
            pointer=H();self.call('spinImageGetData',image,C.byref(pointer))
            if not pointer:raise RuntimeError('empty image data pointer')
            buffer=(C.c_uint8*size).from_address(pointer.value)
            array=np.ndarray((h,w),np.uint8,buffer=buffer,strides=(stride,1)).copy()
            if self.last_frame_id is not None and values['FrameID']<=self.last_frame_id:
                raise RuntimeError('duplicate/out-of-order camera frame')
            self.last_frame_id=values['FrameID']
            values.update(host_get_started_unix_ns=start,host_received_unix_ns=time.time_ns(),
                          camera_clock_relationship_verified=False,pixel_format='Mono8')
            return array,values
        finally:
            if image:self.call('spinImageRelease',image)

    def stream_statistics(self):
        result={}
        for name in ('StreamMode','StreamBufferHandlingMode','StreamBufferCountManual',
                     'StreamReceivedPacketCount','StreamMissedPacketCount','StreamIncompleteFrameCount',
                     'StreamLostFrameCount','StreamPacketResendRequestCount','StreamPacketResendReceivedPacketCount'):
            try:result[name]=self.read(name,self.stream)
            except RuntimeError:result[name]=None
        return result

    def close(self, expected_after=None):
        if self.closed:return
        # Only the explicit, no-acquisition failed-exposure recovery utility
        # may commit a verified restoration to a known previous baseline.
        # Normal diagnostics always compare against their own before snapshot.
        if expected_after is not None:
            changed={k for k in self.before.keys() | expected_after.keys()
                     if self.before.get(k)!=expected_after.get(k)}
            if self.acquiring or self.undo or changed-{'ExposureTime'}:
                raise RuntimeError('explicit recovery close permits only a verified ExposureTime correction')
            self.explicit_recovery_expected_after=dict(expected_after)
        errors=[]
        def cleanup(name,*args):
            try:self.call(name,*args)
            except Exception as error:errors.append(str(error))
        if self.acquiring:cleanup('spinCameraEndAcquisition',self.camera);self.acquiring=False
        for owner,name,value in reversed(self.undo):
            try:
                self._write(name,value,owner)
                if self.read(name,owner)!=value:raise RuntimeError('restore readback mismatch: '+name)
            except Exception as error:errors.append(str(error))
        self.undo.clear()
        if self.initialized:
            self.after=self.parameters()
            target=self.before if expected_after is None else expected_after
            if target!=self.after:errors.append('camera final parameters differ from declared restoration target')
            cleanup('spinCameraDeInit',self.camera);self.initialized=False
        if self.camera:cleanup('spinCameraRelease',self.camera);self.camera=H()
        if self.cameras:
            cleanup('spinCameraListClear',self.cameras);cleanup('spinCameraListDestroy',self.cameras);self.cameras=H()
        if self.system:cleanup('spinSystemReleaseInstance',self.system);self.system=H()
        if self.lock:self.lock.close();self.lock=None
        self.closed=True;self.close_errors=errors
        if errors:raise RuntimeError('; '.join(errors))
