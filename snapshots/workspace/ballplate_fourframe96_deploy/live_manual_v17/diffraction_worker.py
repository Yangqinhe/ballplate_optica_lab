"""Latest-only CPU simulation. No SDK, X11 calls, output windows or scans."""
from dataclasses import replace
import hashlib
import threading
import time

import cv2
from PyQt5 import QtCore

from diffraction_model import CachedASM,compare_shape,warp_to_camera,fit_zero_mapping,reference_signature
from realtime_model import patterns


class DiffractionWorker(QtCore.QThread):
    result=QtCore.pyqtSignal(object)
    failed=QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__();self.condition=threading.Condition();self.pending=None;self.reference=None
        self.stopping=False;self.engine=CachedASM();self.cache_key=None;self.cache_image=None

    def submit(self,request):
        with self.condition:
            if self.stopping:return
            # At most one current job + one latest preview + one explicit reference.
            if request.get('kind')=='zero':self.reference=request
            else:self.pending=request
            self.condition.notify()

    def stop(self):
        with self.condition:
            self.stopping=True;self.pending=None;self.reference=None;self.condition.notify_all()

    def target(self,r):
        s=r['settings']
        if r.get('centered',True):s=replace(s,x=0,y=0)
        if r.get('kind')=='zero':s=replace(s,dmd='fiducial',slm='zero',x=0,y=0)
        file_key=tuple(None if a is None else hashlib.sha256(a.tobytes()).hexdigest()
                       for a in (r.get('dmd_file'),r.get('slm_file')))
        key=(s,r['config'],file_key)
        if key!=self.cache_key:
            dmd,phase=patterns(s,r.get('dmd_file'),r.get('slm_file'))
            self.cache_image=self.engine.simulate(dmd,phase,r['config'])
            self.cache_key=key;self.target_dmd=dmd;self.target_phase=phase
        return self.cache_image

    def process(self,r):
        started=time.monotonic();sim=self.target(r)
        common=dict(epoch=r['epoch'],kind=r.get('kind','preview'),frame_id=r.get('frame_id'),
            frame_received=r.get('frame_received'),computed_monotonic=time.monotonic(),
            signature=reference_signature(r['config'],r['active']),active=r['active'],
            target_settings=self.cache_key[0],model_shape=list(sim.shape))
        if r.get('kind')=='zero':
            registration=fit_zero_mapping(r['measured'],sim,r['active'])
            registration.update(signature=common['signature'],reference_folder=r['folder'])
            common.update(registration=registration)
        else:
            warped,matrix=warp_to_camera(sim,r['active'],**r['mapping'])
            comparison=compare_shape(r['measured'],warped,r['active'])
            common.update(comparison,matrix_sensor=matrix.tolist(),
                measured_preview=r['measured'],
                # Compact unregistered view helps when the initial ROI/scale guess is poor.
                model_overview=cv2.resize(sim,(384,360),interpolation=cv2.INTER_AREA),
                target_dmd=self.target_dmd,target_phase=self.target_phase)
        common['compute_ms']=(time.monotonic()-started)*1000
        return common

    def run(self):
        while True:
            with self.condition:
                while not self.stopping and self.reference is None and self.pending is None:self.condition.wait()
                if self.stopping:return
                if self.reference is not None:r=self.reference;self.reference=None
                else:r=self.pending;self.pending=None
            try:
                result=self.process(r)
                with self.condition:
                    if self.stopping:return
                self.result.emit(result)
            except Exception as error:
                self.failed.emit(dict(epoch=r['epoch'],kind=r.get('kind','preview'),error=str(error)))
