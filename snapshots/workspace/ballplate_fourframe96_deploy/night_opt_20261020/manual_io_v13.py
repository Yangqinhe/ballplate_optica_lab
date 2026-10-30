"""Causal new-display revision barrier after each acknowledged file load.

Framebuffer pixels may update before the next heartbeat publication. The
old unchanged revision must NEVER acknowledge that new input. Even repeated
identical bitmaps receive a new revision from original show_outputs().
"""
import time
from night_opt_20261020.manual_io_v11 import ManualIO as CommitIO


def next_revision_floor(before,previous_ack):
    return max(int(before),int(previous_ack))+1


class ManualIO(CommitIO):
    def __init__(self,bundle):
        self.minimum_revision=0;self.last_acknowledged_revision=-1
        super().__init__(bundle)
    def load(self,path,phase=False):
        heartbeat,_=self.guard(black=not bool(self.expected))
        floor=next_revision_floor(heartbeat['display_revision'],self.last_acknowledged_revision)
        result=super().load(path,phase=phase)
        self.minimum_revision=floor
        return dict(**result,required_new_display_revision_at_least=floor)
    def arm_and_verify(self):
        until=time.monotonic()+5
        while True:
            verified=super().arm_and_verify()
            if verified['revision']>=self.minimum_revision:
                self.last_acknowledged_revision=verified['revision']
                return dict(**verified,causal_new_revision_barrier_passed=True,
                    required_revision_at_least=self.minimum_revision)
            if time.monotonic()>=until:raise TimeoutError('new file must receive a NEW acknowledged display revision')
            time.sleep(.05)
