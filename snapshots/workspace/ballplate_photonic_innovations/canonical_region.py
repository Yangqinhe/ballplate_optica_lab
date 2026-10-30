"""Match pinned native ADC arithmetic exactly, including FP32 operation grouping."""
import types
import torch
from torch.nn import functional as F
from optical.devices import quantize_ste


def forward(self,field,seeds=None):
    cfg=self.cfg
    if cfg.noise_enabled:raise ValueError("noise must use the full native-count path")
    components=torch.stack((field.real,field.imag),dim=1);intensity=None
    for row in range(cfg.ccd_area_samples):
        for col in range(cfg.ccd_area_samples):
            count=cfg.ccd_area_samples
            offset=field.real.new_tensor([(col+.5)/count-.5,(row+.5)/count-.5])
            offset=-offset*cfg.ccd_pitch_m/self.plane_pitch*(2/(cfg.grid_size-1))
            sampled=F.grid_sample(components,self.grid.expand(len(field),-1,-1,-1)+offset,
                                 mode="bilinear",padding_mode="zeros",align_corners=True)
            value=sampled.square().sum(dim=1)[:,0]/count**2
            intensity=value if intensity is None else intensity+value
    electrons=intensity*cfg.electrons_per_intensity*(cfg.ccd_exposure_s/cfg.detector_reference_exposure_s)
    counts=electrons.clamp(0,cfg.full_well_electrons)*(cfg.mono8_counts_per_intensity/cfg.electrons_per_intensity)
    counts=counts.clamp(0,255)
    if not self.diagnostic_continuous:counts=quantize_ste(counts)
    return torch.stack([x.sum(1) for x in counts.split(self.lengths,dim=1)],1)/self.reference


def install(model):
    target=model.encoder.regions.inner if hasattr(model.encoder.regions,"inner") else model.encoder.regions
    target.forward=types.MethodType(forward,target)
