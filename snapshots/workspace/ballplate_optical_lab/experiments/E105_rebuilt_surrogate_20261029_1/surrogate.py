"""Training-only hybrid optics. No GUI, SDK, hardware or file I/O on import.

ASM matches the existing zero-reference geometry convention. Camera RAW and
field features remain distinct; the real complex field has NOT been measured.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def centered(a,shape):
    h,w=a.shape[-2:];ny,nx=shape
    if h>ny or w>nx:raise ValueError('source larger than propagation grid')
    y,x=(ny-h)//2,(nx-w)//2
    return F.pad(a,(x,nx-w-x,y,ny-h-y))


def propagate(field,transfer):
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(torch.fft.fftshift(field,dim=(-2,-1)))*transfer),dim=(-2,-1))


def oriented(a,code):
    a=torch.rot90(a,int(code)%4,(-2,-1))
    return a.flip(-1) if int(code)>=4 else a


def discrete_gray(gray):
    value=gray.clamp(0,255)
    return value+(value.round()-value).detach()


class ASMRelay(nn.Module):
    """Known relay plus 13 bounded effective corrections; configurable grid.

    q: CCD xy, SLM xy, SLM scale xy, phase depth, blur, wavefront3,
    log photometric gain, additive ADU background. Sources may be CPU cached.
    """
    def __init__(self,baseline,layout,geometry,device='cpu',padding=None,phase_sign=1,slm_shift_bound=32.):
        super().__init__();self.device=torch.device(device)
        if phase_sign not in (-1,1):raise ValueError('phase sign must be a declared discrete hypothesis')
        self.phase_sign=phase_sign
        if not 0<slm_shift_bound<=320:raise ValueError('bounded effective SLM coordinate hypothesis required')
        self.slm_shift_bound=float(slm_shift_bound)
        self.expected=baseline['expected'];cfg=self.expected['optical_reference']
        self.config=dict(cfg);self.layout=layout;self.geometry=geometry
        self.roi=self.expected['roi'];self.native_shape=(self.roi['Height'],self.roi['Width'])
        from scipy.fft import next_fast_len
        pad=cfg['padding'] if padding is None else padding
        self.shape=(next_fast_len(max(600*pad,1080)),next_fast_len(max(600*pad,1920)))
        ny,nx=self.shape;wl=cfg['wavelength_m'];pitch=cfg['pitch_m'];k=2*np.pi/wl
        fy=np.fft.fftshift(np.fft.fftfreq(ny,pitch));fx=np.fft.fftshift(np.fft.fftfreq(nx,pitch))
        kz=np.sqrt(np.maximum(k*k-(2*np.pi*fy[:,None])**2-(2*np.pi*fx[None,:])**2,0.))
        yy=(np.arange(ny)-ny//2)*pitch;xx=(np.arange(nx)-nx//2)*pitch
        radius=yy[:,None]**2+xx[None,:]**2
        for i in (1,2):
            f=cfg[f'f{i}_m']
            self.register_buffer(f'h{i}',torch.tensor(np.fft.fftshift(np.exp(1j*f*(kz-k))).astype(np.complex64),device=self.device))
            self.register_buffer(f'l{i}',torch.tensor(np.exp(-1j*k*radius/(2*f)).astype(np.complex64),device=self.device))
        aperture=centered(torch.ones((1080,1920),device=self.device),self.shape)
        self.register_buffer('aperture',aperture)
        y,x=torch.meshgrid(torch.arange(ny,device=self.device)-ny//2,
                           torch.arange(nx,device=self.device)-nx//2,indexing='ij')
        self.register_buffer('wave_basis',torch.stack(((x/600).square(),x*y/600**2,(y/600).square())))
        self.q=nn.Parameter(torch.zeros(13,device=self.device))
        # Small initial blur mix (~0.009), fitted only as an effective response.
        self.register_buffer('photo_initial',torch.tensor([1.,10.],device=self.device))
        bounds=torch.tensor([r['bounds'] for r in layout['regions']],device=self.device,dtype=torch.long)
        self.register_buffer('bounds',bounds)
        self.register_buffer('areas',((bounds[:,2]-bounds[:,0])*(bounds[:,3]-bounds[:,1])).float())
        axis=torch.arange(-7,8,device=self.device,dtype=torch.float32)
        kernel=torch.exp(-.5*(axis/2.5)**2);kernel=kernel/kernel.sum()
        self.register_buffer('blur_kernel',(kernel[:,None]*kernel[None,:])[None,None])
        self.sources={}
        h,w=self.native_shape
        cy,cx=torch.meshgrid(torch.arange(h,device=self.device)+self.roi['OffsetY'],
                            torch.arange(w,device=self.device)+self.roi['OffsetX'],indexing='ij')
        self.register_buffer('sensor_coordinates',torch.stack((cx,cy),-1).float())
        self.register_buffer('matrix',torch.tensor(geometry['matrix_sensor'],device=self.device,dtype=torch.float32))

    @torch.no_grad()
    def source_field(self,key,dmd):
        if key not in self.sources:
            a=torch.as_tensor(dmd,dtype=torch.float32,device=self.device)/255
            if a.shape!=(600,600):raise ValueError('new platform uses actual 600-square DMD; no hidden scaling')
            u=centered(a.sqrt().to(torch.complex64),self.shape)
            u=propagate(u,self.h1)*self.l1
            self.sources[key]=propagate(u,self.h1).cpu()
        return self.sources[key].to(self.device)

    def modulation(self,gray,quantize=True):
        if gray.shape==(75,75):gray=gray.repeat_interleave(8,0).repeat_interleave(8,1)
        if gray.shape!=(600,600):raise ValueError('explicit macro or native SLM patch required')
        g=discrete_gray(gray) if quantize else gray.clamp(0,255)
        blur=F.conv2d(g[None,None],self.blur_kernel,padding=7)[0,0]
        # Linear smooth bounded mix keeps nonzero derivative at the initial value.
        mix=.5*torch.sigmoid(self.q[7]-4)
        g=(1-mix)*g+mix*blur
        settings=self.expected['slm'];x,y=660-settings['x'],240+settings['y']
        phase=g*(self.phase_sign*self.config['phase_depth_rad']/255)*(1+.4*torch.tanh(self.q[6]))
        panel=F.pad(phase,(x,1920-600-x,y,1080-600-y))
        if self.config['slm_flip_y']:panel=panel.flip(-2)
        phase=centered(panel,self.shape)
        if torch.is_grad_enabled() or torch.any(self.q[2:6]!=0):
            ny,nx=self.shape
            theta=torch.zeros((1,2,3),device=self.device,dtype=phase.dtype)
            theta[0,0,0]=1+.1*torch.tanh(self.q[4]);theta[0,1,1]=1+.1*torch.tanh(self.q[5])
            theta[0,:,2]=torch.tanh(self.q[2:4])*torch.tensor([2*self.slm_shift_bound/nx,2*self.slm_shift_bound/ny],device=self.device)
            grid=F.affine_grid(theta,(1,2,ny,nx),align_corners=False)
            values=F.grid_sample(torch.stack((phase.cos()-1,phase.sin()))[None],grid,align_corners=False)[0]
            modulation=torch.complex(values[0]+1,values[1])
        else:modulation=torch.exp(1j*phase)
        wave=(3*torch.tanh(self.q[8:11])[:,None,None]*self.wave_basis).sum(0)
        modulation=modulation*torch.exp(1j*wave)
        if self.config['panel_aperture']:modulation=modulation*self.aperture
        return modulation

    def camera_grid(self):
        coords=self.sensor_coordinates-self.matrix[:,2]-10*torch.tanh(self.q[:2])
        mapped=coords@torch.linalg.inv(self.matrix[:,:2]).T
        # Geometry was fitted to the padding3 reference grid. Larger numerical
        # grids shift the same optical centre but do not change hardware scale.
        oriented_shape=self.shape[::-1] if self.geometry['orientation']%2 else self.shape
        oldshape=(1800,1920);oldshape=oldshape[::-1] if self.geometry['orientation']%2 else oldshape
        h,w=oriented_shape
        mapped=mapped+torch.tensor([(w-oldshape[1])/2,(h-oldshape[0])/2],device=self.device)
        return (2*(mapped+.5)/torch.tensor([w,h],device=self.device)-1)[None]

    def forward(self,key,dmd,gray,quantize=True):
        source=self.source_field(key,dmd)
        u=source*self.modulation(gray,quantize=quantize)
        u=propagate(propagate(u,self.h2)*self.l2,self.h2)
        u=oriented(u,self.geometry['orientation']);grid=self.camera_grid()
        intensity=F.grid_sample(u.abs().square()[None,None],grid,align_corners=False)[0,0]*255
        parts=F.grid_sample(torch.stack((u.real,u.imag))[None],grid,align_corners=False)[0]
        gain=self.photo_initial[0]*torch.exp(self.q[11].clamp(-4,4))
        background=self.photo_initial[1]+3*torch.tanh(self.q[12])
        raw=(gain*intensity+background).clamp(0,255)
        return raw,parts*torch.sqrt(255*gain)

    def integrate(self,raw):
        ii=F.pad(raw.double().cumsum(-2).cumsum(-1),(1,0,1,0))
        x,y,u,v=self.bounds.T
        return (ii[...,v,u]-ii[...,y,u]-ii[...,v,x]+ii[...,y,x]).float()


class ResidualMean(nn.Module):
    """26,673 training-only parameters; physical RAW retained at native size."""
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(5,16,3,padding=1);self.c2=nn.Conv2d(16,16,3,padding=1)
        self.c3=nn.Conv2d(16,32,3,stride=2,padding=1);self.c4=nn.Conv2d(32,32,3,padding=1)
        self.c5=nn.Conv2d(48,16,3,padding=1);self.c6=nn.Conv2d(16,1,3,padding=1)
        self.condition=nn.Sequential(nn.Linear(16,32),nn.SiLU(),nn.Linear(32,64))
        nn.init.zeros_(self.c6.weight);nn.init.zeros_(self.c6.bias)

    def forward(self,features,coefficients):
        a=F.silu(self.c1(features));a=F.silu(self.c2(a));b=F.silu(self.c3(a))
        film=self.condition(coefficients);scale,bias=film.chunk(2,-1)
        b=F.silu(self.c4(b)*(1+.1*scale[:,:,None,None])+.1*bias[:,:,None,None])
        b=F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False)
        return self.c6(F.silu(self.c5(torch.cat((a,b),1))))


def residual_features(raw,field,size=256,raw_scale=16.,field_scale=4.):
    """Fixed scales, never per-image contrast normalisation or alignment."""
    r=F.interpolate(raw[None,None],size=(size,size),mode='area')/raw_scale
    f=F.interpolate(field[None],size=(size,size),mode='area')/field_scale
    y,x=torch.meshgrid(torch.linspace(-1,1,size,device=raw.device),torch.linspace(-1,1,size,device=raw.device),indexing='ij')
    return torch.cat((r,f,torch.stack((x,y))[None]),1)


def noise_covariance(grouped_residuals,shrinkage=.2):
    """Repeated same-condition effective-frame residuals only; not state variance."""
    a=np.asarray(grouped_residuals,np.float64)
    if a.ndim!=2 or len(a)<2 or not 0<=shrinkage<=1:raise ValueError('finite repeated residual matrix required')
    if not np.isfinite(a).all():raise ValueError('nonfinite residual')
    c=a.T@a/len(a)
    return (1-shrinkage)*c+shrinkage*np.diag(np.diag(c))
