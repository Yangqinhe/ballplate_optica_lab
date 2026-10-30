"""Single phase plane and explicit physical binary DMD temporal layouts."""
import innov_runtime
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Independent, Normal
from torch.utils.checkpoint import checkpoint
from optical.encoder import OpticalFrameEncoder
from optical.devices import quantize_ste
from dashboard.acquisition import native_counts


def pack_history(frames, encoding):
    """[B,8,1,128,128] raw binary -> logical DMD, oldest to newest."""
    if tuple(frames.shape[1:]) != (8, 1, 128, 128):
        raise ValueError("exact eight raw binary images required")
    if encoding == "single":
        return frames[:, -1]
    if encoding == "mosaic2":
        # Only oldest and newest; intermediate frames retained in raw FIFO, not propagated.
        return torch.cat((frames[:, 0], frames[:, -1]), dim=-1)
    if encoding == "mosaic8":
        return torch.cat((torch.cat([frames[:, i] for i in range(4)], dim=-1),
                          torch.cat([frames[:, i] for i in range(4, 8)], dim=-1)), dim=-2)
    raise ValueError(encoding)


def physical_regions(cfg, encoding, readout):
    """Corrected CCD pixel rectangles. Geometric tiles, not ground-truth position."""
    p = cfg.ccd_roi_pixels
    if readout == "tiles8":
        return [[r*p//2, (r+1)*p//2, c*p//4, (c+1)*p//4]
                for r in range(2) for c in range(4)]
    if readout == "quad4":
        return [[r*p//2, (r+1)*p//2, c*p//2, (c+1)*p//2] for r in range(2) for c in range(2)]
    if readout in ("paired_quad8", "sparse_pair8"):
        if encoding != "mosaic2":
            raise ValueError("paired quadrants require two side-by-side image tiles")
        half_tile = 128*cfg.superpixel*cfg.dmd_pitch_m*cfg.magnification/2/cfg.ccd_pitch_m
        left, centre, right = round(p/2-half_tile), p//2, round(p/2+half_tile)
        if readout == "sparse_pair8":
            # Four 32x32 windows per image, total 8192 native CCD pixels.
            return [[y,y+32,x,x+32] for cx in (left,right)
                    for y in (p//2-32,p//2) for x in (cx-32,cx)]
        # Regions cover both ideal geometric images within the existing camera, no enlargement.
        y0, y1 = max(0, round(p/2-half_tile)), min(p, round(p/2+half_tile))
        result = []
        for x0, xc, x1 in ((max(0, round(left-half_tile)), left, centre),
                           (centre, right, min(p, round(right+half_tile)))):
            result.extend([[y0,p//2,x0,xc], [y0,p//2,xc,x1],
                           [p//2,y1,x0,xc], [p//2,y1,xc,x1]])
        return result
    if readout in ("sparse8", "sparse8_near"):
        # Fixed 32x32 windows, 8192 total pixels. Optical routing must bring signal here.
        radius, delta = 16, 55
        if readout == "sparse8_near": delta=16
        centres = [(p//2+dy, p//2+dx) for dy in (-delta, delta)
                   for dx in (-3*delta, -delta, delta, 3*delta)]
        return [[y-radius,y+radius,x-radius,x+radius] for y,x in centres]
    raise ValueError(readout)


class RegionDetector(nn.Module):
    """Same native pixel area integration/Mono8, selected fixed pixel rectangles."""
    def __init__(self, detector, regions, reference):
        super().__init__()
        self.cfg = detector.cfg
        self.plane_pitch = detector.plane_pitch
        self.regions = regions
        self.register_buffer("reference", torch.tensor(float(reference)))
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError("positive fixed simulation photon reference required")
        p = self.cfg.ccd_roi_pixels
        flat_indices, lengths = [], []
        occupied = set()
        for y0,y1,x0,x1 in regions:
            if not 0 <= y0 < y1 <= p or not 0 <= x0 < x1 <= p:
                raise ValueError("detector ROI outside existing native CCD")
            ids = (torch.arange(y0,y1)[:,None]*p + torch.arange(x0,x1)[None,:]).flatten()
            if occupied.intersection(ids.tolist()):
                raise ValueError("overlapping pixel ROIs must not be double counted")
            occupied.update(ids.tolist())
            flat_indices.append(ids)
            lengths.append(len(ids))
        self.lengths = lengths
        self.register_buffer("indices", torch.cat(flat_indices), persistent=False)
        grid = detector.sample_grid.reshape(1,-1,2)[:,self.indices]
        self.register_buffer("grid", grid.unsqueeze(1), persistent=False)
        self.diagnostic_continuous = False

    def forward(self, field, seeds=None):
        cfg = self.cfg
        if cfg.noise_enabled:
            # Sparse RNG order differs from a full sensor: deliberately fail until calibrated/tested.
            raise ValueError("sparse independent pixel noise contract not validated; use full-count noise path")
        components = torch.stack((field.real, field.imag), dim=1)
        intensity = None
        for r in range(cfg.ccd_area_samples):
            for c in range(cfg.ccd_area_samples):
                offset = field.real.new_tensor([(c+.5)/cfg.ccd_area_samples-.5,
                                                (r+.5)/cfg.ccd_area_samples-.5])
                offset = -offset*cfg.ccd_pitch_m/self.plane_pitch*2/(cfg.grid_size-1)
                samples = F.grid_sample(components, self.grid.expand(len(field),-1,-1,-1)+offset,
                                        mode="bilinear",padding_mode="zeros",align_corners=True)
                value = samples.square().sum(1)[:,0]/cfg.ccd_area_samples**2
                intensity = value if intensity is None else intensity+value
        electrons = intensity*cfg.electrons_per_intensity*(cfg.ccd_exposure_s/cfg.detector_reference_exposure_s)
        counts = electrons.clamp(0,cfg.full_well_electrons)*cfg.mono8_counts_per_intensity/cfg.electrons_per_intensity
        counts = counts.clamp(0,255)
        if not self.diagnostic_continuous:
            counts = quantize_ste(counts)
        return torch.stack([x.sum(1) for x in counts.split(self.lengths,dim=1)],dim=1)/self.reference

    def full_reference(self, full_counts):
        values = full_counts.flatten(1)[:,self.indices]
        return torch.stack([x.sum(1) for x in values.split(self.lengths,dim=1)],dim=1)/self.reference


class PhotonicEncoder(OpticalFrameEncoder):
    def __init__(self, cfg, encoding, readout, reference):
        super().__init__(cfg)
        self.encoding = encoding
        self.regions = RegionDetector(self.detector, physical_regions(cfg, encoding, readout), reference)

    def input_field(self, images):
        shapes = {"single": (128,128), "mosaic2": (128,256), "mosaic8": (256,512)}
        if images.ndim != 4 or images.shape[1] != 1 or tuple(images.shape[-2:]) != shapes[self.encoding]:
            raise ValueError("DMD input shape does not match physical temporal encoding")
        if images.dtype == torch.uint8:
            if bool(((images!=0)&(images!=255)).any()):
                raise ValueError("DMD input not binary")
            amplitude = images[:,0].float()/255
        else:
            if not torch.isfinite(images).all() or bool(((images!=0)&(images!=1)).any()):
                raise ValueError("DMD floating input not binary")
            amplitude = images[:,0].float()
        native_h, native_w = [n*self.cfg.superpixel for n in amplitude.shape[-2:]]
        if native_h > 1080 or native_w > 1920:
            raise ValueError("composite exceeds user's 1920x1080 DMD")
        factor = self.cfg.superpixel*self.cfg.samples_per_mirror
        amplitude = amplitude.repeat_interleave(factor,-2).repeat_interleave(factor,-1)
        amplitude = amplitude+(1-amplitude)*self.cfg.input_off_amplitude
        dh,dw = self.cfg.grid_size-amplitude.shape[-2], self.cfg.grid_size-amplitude.shape[-1]
        if min(dh,dw)<0 or dh%2 or dw%2:
            raise ValueError("ASM grid cannot contain centred DMD composite")
        return F.pad(amplitude,(dw//2,dw//2,dh//2,dh//2)).to(torch.complex64)

    def forward(self, image, noise_seeds=None):
        field = self.field(image)
        if self.cfg.noise_enabled:
            return self.regions.full_reference(native_counts(self.detector,field,noise_seeds))
        return self.regions(field,noise_seeds)


class PhotonicPolicy(nn.Module):
    architecture_id = "single_slm_photonic_temporal_actor_v1"
    def __init__(self,cfg,encoding="mosaic2",readout="paired_quad8",reference=342904.,
                 mode="learned",head="linear",phase_seed=93,phase_std=.3):
        super().__init__()
        self.options = dict(encoding=encoding,readout=readout,reference=reference,mode=mode,
                            head=head,phase_seed=phase_seed,phase_std=phase_std)
        self.mode,self.encoding,self.head_kind = mode,encoding,head
        self.encoder = PhotonicEncoder(cfg,encoding,readout,reference)
        self.feature_dim = len(self.encoder.regions.lengths)
        self.electric_history = encoding == "single"
        dim = self.feature_dim*(8 if self.electric_history else 1)
        if head == "linear":
            self.actor_head = nn.Linear(dim,2)
        elif head == "balanced":
            if encoding != "mosaic2" or readout not in ("paired_quad8","sparse_pair8"):
                raise ValueError("balanced PD wiring needs paired quadrant regions")
            # Fixed signs; 2x2 gain matrix and two bias values, no hidden nonlinear layer.
            self.actor_head = nn.Linear(2,2,bias=True)
            self.gains = nn.Parameter(torch.tensor([1.,1.]))
        else:
            raise ValueError(head)
        self.critic = nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,1))
        self.log_std = nn.Parameter(torch.full((2,),math.log(.025)))
        self.min_std,self.max_std = .005,.04
        with torch.no_grad():
            gen=torch.Generator().manual_seed(phase_seed)
            raw=torch.randn(self.encoder.slm.raw.shape,generator=gen)*phase_std
            self.encoder.slm.raw.copy_(raw if mode != "flat" else torch.zeros_like(raw))
            self.actor_head.weight.zero_(); self.actor_head.bias.zero_()
        self.encoder.slm.raw.requires_grad_(mode=="learned")

    def encode_indexed_history(self,frames,indices,seeds):
        if self.electric_history:
            unique,inverse=indices.flatten().unique(sorted=True,return_inverse=True)
            q=[self.encoder(frames[i:i+1],[seeds[i]]) for i in unique.tolist()]
            return torch.cat(q)[inverse].reshape(len(indices),8,self.feature_dim)
        outputs=[]
        for row in indices:
            image=pack_history(frames[row][None],self.encoding)
            fn=lambda x,seed=int(seeds[int(row[-1])]): self.encoder(x,[seed])
            q=checkpoint(fn,image,use_reentrant=False) if torch.is_grad_enabled() and self.encoder.slm.raw.requires_grad else fn(image)
            outputs.append(q)
        return torch.cat(outputs)

    def flat_features(self,q):
        return q.flatten(1)

    def actor_mean(self,q):
        if self.head_kind == "linear":
            return self.actor_head(self.flat_features(q))
        # Two image planes, four quadrant fluxes each. Derivative uses known 0.14 s interval.
        a,b=q[:,:4],q[:,4:]
        def signed(x):
            return torch.stack((x[:,1]+x[:,3]-x[:,0]-x[:,2],
                                x[:,2]+x[:,3]-x[:,0]-x[:,1]),dim=1)
        old,new=signed(a),signed(b)
        delta=(new-old)/.14
        return self.actor_head(new)+delta*self.gains

    def value(self,q):
        return self.critic(self.flat_features(q.detach())).squeeze(-1)

    def distribution(self,q):
        mu=self.actor_mean(q)
        return Independent(Normal(mu,self.log_std.exp().clamp(self.min_std,self.max_std).expand_as(mu)),1)

    @torch.no_grad()
    def project_std(self):
        self.log_std.clamp_(math.log(self.min_std),math.log(self.max_std))

    def actor_parameters(self):
        return sum(p.numel() for p in self.actor_head.parameters())+(self.gains.numel() if self.head_kind=="balanced" else 0)

    @torch.no_grad()
    def phase_stats(self):
        gray=self.encoder.slm.native_gray()
        return dict(raw_std=float(self.encoder.slm.raw.std(unbiased=False)),
                    gray_min=float(gray.min()),gray_max=float(gray.max()),gray_levels=int(gray.unique().numel()))
