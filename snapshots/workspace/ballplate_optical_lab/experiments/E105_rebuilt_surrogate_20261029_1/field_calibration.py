"""Offline bounded source illumination / gray-phase correction candidate.

This is an effective shared coherent model, not a component diagnosis. It never
reads a real CCD or accesses hardware. Original ASM and all old models stay intact.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from workflow import ROOT,read,write,ref,require
from dataset import checked
from surrogate import ASMRelay,centered,propagate,discrete_gray
from fit import gpu,groups,archive_sources


class SourceLUTRelay(ASMRelay):
    """13 frozen calibrated parameters + ten bounded effective corrections."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.extra=torch.nn.Parameter(torch.zeros(10,device=self.device))
        y,x=torch.meshgrid(torch.linspace(-1,1,600,device=self.device),
                           torch.linspace(-1,1,600,device=self.device),indexing='ij')
        self.register_buffer('illumination_basis',torch.stack((x,y,x.square()-1/3,x*y,y.square()-1/3)))
        self.dmd_pixels={}

    def source_field(self,key,dmd):
        if key not in self.dmd_pixels:
            require(dmd.shape==(600,600),'native600 DMD required')
            self.dmd_pixels[key]=torch.tensor(dmd,device=self.device,dtype=torch.float32)/255
        exponent=(.5*torch.tanh(self.extra[:5])[:,None,None]*self.illumination_basis).sum(0).clamp(-1.5,1.5)
        field=self.dmd_pixels[key].sqrt()*torch.exp(exponent)
        u=centered(field.to(torch.complex64),self.shape)
        return propagate(propagate(u,self.h1)*self.l1,self.h1)

    def modulation(self,gray,quantize=True):
        if gray.shape==(75,75):gray=gray.repeat_interleave(8,0).repeat_interleave(8,1)
        require(gray.shape==(600,600),'native600 SLM required')
        g=discrete_gray(gray) if quantize else gray.clamp(0,255)
        blur=F.conv2d(g[None,None],self.blur_kernel,padding=7)[0,0]
        mix=.5*torch.sigmoid(self.q[7]-4);g=(1-mix)*g+mix*blur
        phase=g*(self.phase_sign*self.config['phase_depth_rad']/255)*(1+.4*torch.tanh(self.q[6]))
        # Smooth endpoint-zero deviation; not an independently measured LUT.
        for k in range(3):phase=phase+(.25*np.pi*torch.tanh(self.extra[5+k]))*torch.sin((k+1)*np.pi*g/255)
        settings=self.expected['slm'];x,y=660-settings['x'],240+settings['y']
        phase=F.pad(phase,(x,1920-600-x,y,1080-600-y))
        if self.config['slm_flip_y']:phase=phase.flip(-2)
        phase=centered(phase,self.shape)
        ny,nx=self.shape;theta=torch.zeros((1,2,3),device=self.device,dtype=phase.dtype)
        theta[0,0,0]=1+.1*torch.tanh(self.q[4]);theta[0,1,1]=1+.1*torch.tanh(self.q[5])
        theta[0,:,2]=torch.tanh(self.q[2:4])*torch.tensor([2*self.slm_shift_bound/nx,2*self.slm_shift_bound/ny],device=self.device)
        grid=F.affine_grid(theta,(1,2,ny,nx),align_corners=False)
        values=F.grid_sample(torch.stack((phase.cos()-1,phase.sin()))[None],grid,align_corners=False)[0]
        modulation=torch.complex(values[0]+1,values[1])
        wave=(3*torch.tanh(self.q[8:11])[:,None,None]*self.wave_basis).sum(0)
        modulation=modulation*torch.exp(1j*wave)
        if self.config['panel_aperture']:modulation=modulation*self.aperture
        return modulation

    def forward(self,key,dmd,gray,quantize=True):
        raw,field=super().forward(key,dmd,gray,quantize)
        background=self.photo_initial[1]+3*torch.tanh(self.q[12]);gain=torch.exp(.5*torch.tanh(self.extra[8]))
        return ((raw-background)*gain+background+.1*torch.tanh(self.extra[9])).clamp(0,255),field*gain.sqrt()


def identity(initial,output):
    """Zero correction must recover calibrated ASM; check extra derivatives."""
    output.mkdir();cfg=read(initial/'MODEL.json');b=checked(cfg['baseline']);device=gpu()
    models=[c(b,cfg['layout'],checked(cfg['geometry']),device,phase_sign=cfg.get('phase_sign',1),
              slm_shift_bound=cfg.get('slm_shift_bound',32)) for c in (ASMRelay,SourceLUTRelay)]
    for model in models:
        with torch.no_grad():model.q.copy_(torch.tensor(cfg['q'],device=device));model.photo_initial[:]=torch.tensor(cfg['photometry'],device=device)
        model.q.requires_grad_(False)
    a=read(ROOT/'ASSETS_STRONG_01/PLAN.json');label=checked(a['catalogs']['PILOT'])['inputs'][0]
    dmd=checked(label['dmd'],True);g=torch.tensor(checked(a['assets']['X4P']['patch'],True),device=device,dtype=torch.float32)
    with torch.no_grad():
        expected,_=models[0](label['input_id'],dmd,g);actual,_=models[1](label['input_id'],dmd,g)
        maximum=float((actual-expected).abs().max());rms=float((actual-expected).square().mean().sqrt())
    model=models[1];weight=torch.tensor(np.random.default_rng(105).normal(size=64),device=device,dtype=torch.float32)
    def objective():
        raw,_=model(label['input_id'],dmd,g)
        return ((model.integrate(raw)/model.areas)*weight).sum()
    gradient=torch.autograd.grad(objective(),model.extra)[0].detach().cpu().numpy();fd=[]
    with torch.no_grad():
        for k in range(10):
            model.extra[k]=.02;plus=float(objective());model.extra[k]=-.02;minus=float(objective());model.extra[k]=0
            fd.append((plus-minus)/.04)
    relative=float(np.linalg.norm(gradient-fd)/max(np.linalg.norm(gradient),1e-12));passed=maximum<1e-4 and relative<.1
    write(output/'RESULT.json',dict(initial=ref(initial/'MODEL.json'),source=ref(__file__),identity_max_ADU=maximum,
        identity_RMS_ADU=rms,autograd=gradient.tolist(),finite_difference=fd,relative_gradient_error=relative,
        passed=passed,no_hardware=True,physical_mapping_NOT_verified=True))
    require(passed,'source/LUT numerical identity or gradient failed')
    print('SOURCE LUT NUMERICAL QA',maximum,relative,flush=True)


def train(dataset,initial,output,steps=1000):
    output.mkdir();start=time.monotonic();idx=read(dataset/'INDEX.json');cfg=read(initial/'MODEL.json')
    require(cfg['source']['sha256']==ref(ROOT/'surrogate.py')['sha256'],'original physical source changed')
    require(cfg['baseline']==idx['baseline'] and cfg['layout']==idx['layout'],'physical/data contract differs')
    samples=groups(idx)
    require(not any(r['label'].get('new_platform_split')=='TEST' or r['mask_split']=='TEST' for r in samples),'TEST forbidden')
    ids=[i for i,r in enumerate(samples) if r['label'].get('new_platform_split','PILOT') in ('TRAIN','PILOT') and r['mask_split']=='TRAIN']
    require(len(ids)>20,'joint TRAIN data required');device=gpu();b=checked(cfg['baseline'])
    relay=SourceLUTRelay(b,idx['layout'],checked(cfg['geometry']),device,
        phase_sign=cfg.get('phase_sign',1),slm_shift_bound=cfg.get('slm_shift_bound',32))
    with torch.no_grad():relay.q.copy_(torch.tensor(cfg['q'],device=device));relay.photo_initial[:]=torch.tensor(cfg['photometry'],device=device)
    relay.q.requires_grad_(False)
    archive_sources(output)
    source=output/'SOURCE/field_calibration.py'
    with source.open('xb') as f:f.write(Path(__file__).read_bytes())
    targets=[torch.tensor(s['image'],device=device) for s in samples]
    grays=[torch.tensor(s['gray'],device=device,dtype=torch.float32) for s in samples]
    z=torch.stack([relay.integrate(a)/relay.areas for a in targets]);scale=z[ids].std(0).clamp(min=.01)
    features=checked(idx['features'],True)['mean_sums']/relay.areas.cpu().numpy();noise=[]
    for i in ids:
        v=features[samples[i]['visits']];noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    from response import covariance
    _,w=covariance(noise);white=torch.tensor(w,device=device,dtype=torch.float32)
    lookup={(s['input_id'],tuple(s['coefficients'])):i for i,s in enumerate(samples)};keep=set(ids);pairs=[]
    for i in ids:
        c=samples[i]['coefficients'];other=lookup.get((samples[i]['input_id'],tuple(-x for x in c)))
        if any(c) and other in keep and other>i:pairs.append((i,other))
    write(output/'PLAN.json',dict(dataset=ref(dataset/'INDEX.json'),initial=ref(initial/'MODEL.json'),implementation=ref(source),
        source=ref(__file__),steps=steps,TRAIN_indices=ids,response_pairs=pairs,
        learned_extra_parameters=10,original13_frozen=True,heldout_labels_used=False,
        no_hardware=True,SLM_optimized=False,not_an_independently_calibrated_LUT=True))
    rng=np.random.default_rng(105430);optim=torch.optim.Adam([relay.extra],lr=.015);history=[];best=float('inf');best_extra=None
    def evaluate():
        with torch.no_grad():
            pp=[]
            for i in ids:
                rr=samples[i];raw,_=relay(rr['input_id'],rr['dmd'],grays[i]);pp.append(relay.integrate(raw)/relay.areas)
            vv=torch.stack(pp);mapping={j:k for k,j in enumerate(ids)}
            mean=((vv-z[ids])/scale).square().mean()
            response=[]
            for p,m in pairs:
                a=white@(z[p]-z[m]);e=white@(vv[mapping[p]]-vv[mapping[m]]-z[p]+z[m])
                floor=64*(1/len(samples[p]['visits'])+1/len(samples[m]['visits']))
                response.append(e.square().sum()/a.square().sum().clamp(min=floor))
            return float(mean+.5*torch.stack(response).mean()),float(mean)
    for step in range(steps):
        optim.zero_grad(set_to_none=True);i=int(rng.choice(ids));s=samples[i];raw,_=relay(s['input_id'],s['dmd'],grays[i])
        loss=(((relay.integrate(raw)/relay.areas-z[i])/scale).square()).mean()+.1*(raw-targets[i]).square().mean()+.001*relay.extra.square().mean()
        loss.backward();del raw,loss
        p,m=pairs[int(rng.integers(len(pairs)))];delta=[]
        for j in (p,m):
            s=samples[j];raw,_=relay(s['input_id'],s['dmd'],grays[j]);delta.append(relay.integrate(raw)/relay.areas)
        a=white@(z[p]-z[m]);error=white@(delta[0]-delta[1]-z[p]+z[m]);floor=64*(1/len(samples[p]['visits'])+1/len(samples[m]['visits']))
        response=.5*error.square().sum()/a.square().sum().clamp(min=floor);response.backward()
        torch.nn.utils.clip_grad_norm_([relay.extra],5);optim.step();del raw,response,delta
        if step%100==0 or step==steps-1:
            value,mean=evaluate();history.append(dict(step=step,TRAIN_score=value,TRAIN_scaled_MSE=mean,extra=relay.extra.detach().cpu().tolist(),seconds=time.monotonic()-start))
            if value<best:best=value;best_extra=relay.extra.detach().cpu().tolist()
            print('SOURCE LUT',step,'TRAIN',round(value,5),'mean',round(mean,5),flush=True)
    new=dict(cfg);new.update(implementation=ref(source),model_family='bounded_source_lut',extra=best_extra,
        initial_model=ref(initial/'MODEL.json'),calibration_dataset=ref(dataset/'INDEX.json'),
        added_effective_parameters=10,original13_frozen=True,not_measured_component_LUT=True)
    write(output/'MODEL.json',new);write(output/'HISTORY.json',history)
    write(output/'RESULT.json',dict(best_TRAIN_score=best,model=ref(output/'MODEL.json'),seconds=time.monotonic()-start,
        formal_TEST=False,SLM_optimized=False,deployed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');p.add_argument('--initial',required=True);p.add_argument('--identity',action='store_true')
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    require(Path(a.output).name==a.output and 1<=a.steps<=1500,'explicit fresh bounded offline fit')
    if a.identity:identity(ROOT/a.initial,ROOT/a.output)
    else:
        require(a.dataset is not None,'explicit dataset required')
        train(ROOT/a.dataset,ROOT/a.initial,ROOT/a.output,a.steps)
