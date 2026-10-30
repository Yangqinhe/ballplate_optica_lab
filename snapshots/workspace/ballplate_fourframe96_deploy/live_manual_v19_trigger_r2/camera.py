"""Stop triggers BEFORE ROI writes; unchanged ROI does not touch size nodes."""
from live_manual_v19_trigger.camera import HardwareTriggerCamera


class OrderedTriggerCamera(HardwareTriggerCamera):
    def set_roi(self, requested):
        if self.acquiring:
            raise RuntimeError('ROI变更前必须停止采集')
        # ROI writes while TriggerMode remains On are not part of the tested
        # stopped-rearm contract. In r1 the inherited free-run order did that.
        if self.read('TriggerMode') == 'On':
            if not self.trigger_before:
                raise RuntimeError('触发已开启却缺少原始回退快照，拒绝变更ROI')
            self._write('TriggerMode', 'Off', self.map)
        actual = self.roi()
        if all(requested[key] == actual[key] for key in actual):
            return actual
        return super().set_roi(requested)
