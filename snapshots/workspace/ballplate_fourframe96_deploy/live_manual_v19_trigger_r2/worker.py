"""Same-owner worker with idempotent unchanged configuration requests."""
from live_manual_v19_trigger.worker import HardwareCameraWorker
from live_manual_v19_trigger.camera import WorkerGuard
from live_manual_v19_trigger_r2.camera import OrderedTriggerCamera
from realtime_worker import write_json


class OrderedTriggerWorker(HardwareCameraWorker):
    def __init__(self, video, directory, exposure=4000., roi=None, camera_factory=None):
        super().__init__(video, directory, exposure, roi,
            camera_factory=camera_factory or (lambda: OrderedTriggerCamera(WorkerGuard(self.check_stop))))

    def reconfigure(self, roi, exposure, initial=False, resume=True):
        if (not initial and self.camera.acquiring and not self.startup_dark_required
                and roi == self.requested_roi and exposure == self.exposure):
            self.show('hold_black')
            self.camera.revalidate(self.exposure, self.requested_roi)
            self.serial += 1
            write_json(self.directory/('configuration_noop_%04d.json'%self.serial),
                dict(actual_roi=self.camera.roi(), active=self.camera.active,
                     no_node_writes=True, same_SDK=True, no_acquisition_restart=True))
            self.message.emit('ROI／曝光未变：只读复核通过，未重启采集／重写ROI。')
            if resume:
                self.show('resume')
            return
        super().reconfigure(roi, exposure, initial=initial, resume=resume)
