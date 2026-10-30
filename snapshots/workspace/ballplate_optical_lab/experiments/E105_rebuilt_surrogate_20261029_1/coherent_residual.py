"""Offline static low-resolution complex pupil; never a deployed image network.

This is one effective coherent correction, not separately identified components.
Actual displayed SLM stays unchanged. TRAIN labels alone determine parameters.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from workflow import ROOT,read,write,ref,require
from dataset import checked,integrate
from surrogate import ASMRelay,centered,discrete_gray,propagate,oriented
from fit import gpu,groups,archive_sources


class PupilRelay(ASMRelay):
    def __init__(self,*args,pupil_size=8,**kwargs):
        super().__init__(*args,**kwargs)
        require(pupil_size in (4,8,16),'declared low-dimensional coherent pupil required')
        self.pupil_size=pupil_size
        self.pupil=torch.nn.Parameter(torch.zeros(2,pupil_size,pupil_size,device=self.device))

    def correction(self):
        # Bounded log-amplitude / phase on the physical 600-pixel SLM patch.
        low=torch.stack((.7*torch.tanh(self.pupil[0]),np.pi*torch.tanh(self.pupil[1])))
        maps=F.interpolate(low[None],size=(600,600),mode='bilinear',align_corners=False)[0]
        settings=self.expected['slm'];x,y=660-settings['x'],240+settings['y']
        maps=F.pad(maps,(x,1920-600-x,y,1080-600-y))
        if self.config['slm_flip_y']:maps=maps.flip(-2)
        maps=centered(maps,self.shape);gain=torch.exp(torch.complex(maps[0],maps[1]))
        ny,nx=self.shape;theta=torch.zeros((1,2,3),device=self.device)
        theta[0,0,0]=1+.1*torch.tanh(self.q[4]);theta[0,1,1]=1+.1*torch.tanh(self.q[5])
        theta[0,:,2]=torch.tanh(self.q[2:4])*torch.tensor([2*self.slm_shift_bound/nx,2*self.slm_shift_bound/ny],device=self.device)
        grid=F.affine_grid(theta,(1,2,ny,nx),align_corners=False)
        values=F.grid_sample(torch.stack((gain.real-1,gain.imag))[None],grid,align_corners=False)[0]
        return torch.complex(1+values[0],values[1])

    def modulation(self,gray,quantize=True):
        return super().modulation(gray,quantize)*self.correction()


class SamplingRelay(ASMRelay):
    """Offline effective SLM-grid hypothesis; does not alter any real display."""
    def __init__(self,*args,sampling_scale=1.,**kwargs):
        super().__init__(*args,**kwargs)
        require(.25<=sampling_scale<=5.,'bounded effective grid scale required')
        self.sampling_scale=float(sampling_scale)

    def modulation(self,gray,quantize=True):
        if gray.shape==(75,75):gray=gray.repeat_interleave(8,0).repeat_interleave(8,1)
        require(gray.shape==(600,600),'explicit SLM patch required')
        g=discrete_gray(gray) if quantize else gray.clamp(0,255)
        blur=F.conv2d(g[None,None],self.blur_kernel,padding=7)[0,0]
        mix=.5*torch.sigmoid(self.q[7]-4);g=(1-mix)*g+mix*blur
        settings=self.expected['slm'];x,y=660-settings['x'],240+settings['y']
        phase=g*(self.phase_sign*self.config['phase_depth_rad']/255)*(1+.4*torch.tanh(self.q[6]))
        phase=F.pad(phase,(x,1920-600-x,y,1080-600-y))
        if self.config['slm_flip_y']:phase=phase.flip(-2)
        phase=centered(phase,self.shape);ny,nx=self.shape;theta=torch.zeros((1,2,3),device=self.device)
        theta[0,0,0]=self.sampling_scale*(1+.1*torch.tanh(self.q[4]))
        theta[0,1,1]=self.sampling_scale*(1+.1*torch.tanh(self.q[5]))
        theta[0,:,2]=torch.tanh(self.q[2:4])*torch.tensor([2*self.slm_shift_bound/nx,2*self.slm_shift_bound/ny],device=self.device)
        grid=F.affine_grid(theta,(1,2,ny,nx),align_corners=False)
        values=F.grid_sample(torch.stack((phase.cos()-1,phase.sin()))[None],grid,align_corners=False)[0]
        modulation=torch.complex(values[0]+1,values[1])
        wave=(3*torch.tanh(self.q[8:11])[:,None,None]*self.wave_basis).sum(0)
        modulation=modulation*torch.exp(1j*wave)
        if self.config['panel_aperture']:modulation=modulation*self.aperture
        return modulation


def sampling_screen(initial,output):
    from response import covariance
    output.mkdir();cfg=read(initial/'MODEL.json');idx=read(ROOT/'JOINT_TRAIN_AUDIT_01/INDEX.json')
    samples=groups(idx);layout=cfg['layout'];area=np.array([r['pixels'] for r in layout['regions']])
    anchors=('CENTER','X_POS12','Y_NEG12','VY_POS');masks=('BASE','X4P','X4M','Y4P','Y4M','X8P','X8M','Y8P','Y8M')
    bykey={(s['input_id'],s['mask_name']):s for s in samples};selected=[bykey[(k,m)] for m in masks for k in anchors]
    truth=np.stack([integrate(s['image'],layout)/area for s in selected]);zf=checked(idx['features'],True)['mean_sums']/area;noise=[]
    for s in samples:
        v=zf[s['visits']];noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    _,white=covariance(noise);scales=np.stack([integrate(s['image'],layout)/area for s in samples if s['mask_name']=='BASE']).std(0,ddof=1).clip(min=.01)
    device=gpu();kwargs=dict(phase_sign=cfg.get('phase_sign',1),slm_shift_bound=cfg.get('slm_shift_bound',32))
    relay=SamplingRelay(checked(cfg['baseline']),layout,checked(cfg['geometry']),device,**kwargs)
    base=ASMRelay(checked(cfg['baseline']),layout,checked(cfg['geometry']),device,**kwargs)
    with torch.no_grad():
        for model in (relay,base):
            model.q.copy_(torch.tensor(cfg['q'],device=device));model.photo_initial[:]=torch.tensor(cfg['photometry'],device=device)
        gray={m:torch.tensor(bykey[(anchors[0],m)]['gray'],device=device,dtype=torch.float32) for m in masks}
        identity=float((relay.modulation(gray['X4P'])-base.modulation(gray['X4P'])).abs().max())
        require(identity<1e-4,'sampling identity failed')
    del base
    source=torch.stack([relay.source_field(k,bykey[(k,'BASE')]['dmd']) for k in anchors]);grid=relay.camera_grid().repeat(4,1,1,1)
    shift=relay.slm_shift_bound*np.tanh(np.array(cfg['q'][2:4]));results=[];all_values=[]
    pairs=[(4*m+k,4*(m+1)+k) for m in (1,3,5,7) for k in range(4)]
    scales_test=[.5,.75,1.,1.5,2.,2.75,8/2.2,4.]
    write(output/'PLAN.json',dict(model=ref(initial/'MODEL.json'),source=ref(__file__),dataset=ref(ROOT/'JOINT_TRAIN_AUDIT_01/INDEX.json'),
        anchors=anchors,masks=masks,sampling_scales=scales_test,phase_signs=[1,-1],centre_delta_pixels=[-96,0,96],
        identity_max=identity,TRAIN_only=True,actual_pixel_pitch_NOT_identified=True,no_hardware=True))
    with torch.no_grad():
        for scale in scales_test:
            relay.sampling_scale=scale
            for sign in (1,-1):
                for dx in (-96,0,96):
                    for dy in (-96,0,96):
                        ss=np.clip(shift+[dx,dy],-.99*relay.slm_shift_bound,.99*relay.slm_shift_bound)
                        relay.q[2:4]=torch.tensor(np.arctanh(ss/relay.slm_shift_bound),device=device);relay.phase_sign=sign;values=[]
                        for mask in masks:
                            u=source*relay.modulation(gray[mask]);u=propagate(propagate(u,relay.h2)*relay.l2,relay.h2)
                            u=oriented(u,relay.geometry['orientation']);gain=relay.photo_initial[0]*torch.exp(relay.q[11].clamp(-4,4))
                            raw=F.grid_sample(u.abs().square()[:,None],grid,align_corners=False)[:,0]*255*gain
                            raw=(raw+relay.photo_initial[1]+3*torch.tanh(relay.q[12])).clamp(0,255)
                            values.extend([(relay.integrate(im)/relay.areas).cpu().numpy() for im in raw])
                        v=np.asarray(values);mean=float(np.mean(((v-truth)/scales)**2));err=[];cos=[]
                        for p,m in pairs:
                            aa=white@(truth[p]-truth[m]);bb=white@(v[p]-v[m]);err.append(float(np.linalg.norm(aa-bb)/max(np.linalg.norm(aa),1e-12)))
                            cos.append(float(aa@bb/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-12)))
                        results.append(dict(scale=scale,phase_sign=sign,centre_delta=[dx,dy],actual_effective_shift=ss.tolist(),
                            mean_TRAIN_scaled_RMSE=float(np.sqrt(mean)),median_response_cosine=float(np.median(cos)),
                            median_response_relative_error=float(np.median(err)),score=float(mean+.5*np.mean(np.square(err)))))
                        all_values.append(v)
            print('SAMPLING SCREEN',scale,min(r['score'] for r in results),flush=True)
    with (output/'READOUTS.npz').open('xb') as f:np.savez_compressed(f,truth=truth,predictions=np.array(all_values))
    write(output/'RESULT.json',dict(hypotheses=results,top10=sorted(results,key=lambda r:r['score'])[:10],
        source=ref(__file__),no_actual_coordinate_change=True,formal_TEST=False,SLM_training_allowed=False))


def axial_screen(initial,output):
    """Move ONLY the virtual phase plane: P(-dz) M P(dz), fixed image plane.

    An effective axial hypothesis, not a measurement of any mirror distance.
    Unlike an extra quadratic pupil, propagation does not commute with M.
    """
    from response import covariance
    output.mkdir();cfg=read(initial/'MODEL.json');idx=read(ROOT/'JOINT_TRAIN_AUDIT_01/INDEX.json');samples=groups(idx)
    layout=cfg['layout'];area=np.array([r['pixels'] for r in layout['regions']]);device=gpu()
    anchors=('CENTER','X_POS12','Y_NEG12','VY_POS');masks=('BASE','X4P','X4M','Y4P','Y4M','X8P','X8M','Y8P','Y8M')
    bykey={(s['input_id'],s['mask_name']):s for s in samples};selected=[bykey[(k,m)] for m in masks for k in anchors]
    truth=np.stack([integrate(s['image'],layout)/area for s in selected]);zf=checked(idx['features'],True)['mean_sums']/area;noise=[]
    for s in samples:
        v=zf[s['visits']];noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    _,white=covariance(noise);scales=np.stack([integrate(s['image'],layout)/area for s in samples if s['mask_name']=='BASE']).std(0,ddof=1).clip(min=.01)
    relay=ASMRelay(checked(cfg['baseline']),layout,checked(cfg['geometry']),device,
        phase_sign=cfg.get('phase_sign',1),slm_shift_bound=cfg.get('slm_shift_bound',32))
    with torch.no_grad():relay.q.copy_(torch.tensor(cfg['q'],device=device));relay.photo_initial[:]=torch.tensor(cfg['photometry'],device=device)
    ny,nx=relay.shape;wl=relay.config['wavelength_m'];pitch=relay.config['pitch_m'];k=2*np.pi/wl
    fy=np.fft.fftshift(np.fft.fftfreq(ny,pitch));fx=np.fft.fftshift(np.fft.fftfreq(nx,pitch))
    kz=np.sqrt(np.maximum(k*k-(2*np.pi*fy[:,None])**2-(2*np.pi*fx[None,:])**2,0.))-k
    source=torch.stack([relay.source_field(key,bykey[(key,'BASE')]['dmd']) for key in anchors]);grid=relay.camera_grid().repeat(4,1,1,1)
    gray={m:torch.tensor(bykey[(anchors[0],m)]['gray'],device=device,dtype=torch.float32) for m in masks}
    pairs=[(4*m+j,4*(m+1)+j) for m in (1,3,5,7) for j in range(4)]
    distances=sorted(set([0.,-1.,1.,-2.,2.,-3.,3.,-5.,5.,-10.,10.,-20.,20.,-30.,30.,-50.,50.]))
    write(output/'PLAN.json',dict(initial=ref(initial/'MODEL.json'),source=ref(__file__),dataset=ref(ROOT/'JOINT_TRAIN_AUDIT_01/INDEX.json'),
        virtual_SLM_offsets_mm=distances,anchors=anchors,masks=masks,TRAIN_only=True,no_hardware=True,
        interpretation='P(-dz) M P(dz) effective non-Fourier-plane hypothesis; not measured physical distance'))
    results=[];all_values=[];identity=None
    with torch.no_grad():
        for mm in distances:
            hd=torch.tensor(np.fft.fftshift(np.exp(1j*(mm/1000)*kz)).astype(np.complex64),device=device)
            shifted=propagate(source,hd);values=[]
            for mask in masks:
                u=shifted*relay.modulation(gray[mask]);u=propagate(u,relay.h2*hd.conj())*relay.l2;u=propagate(u,relay.h2)
                u=oriented(u,relay.geometry['orientation']);gain=relay.photo_initial[0]*torch.exp(relay.q[11].clamp(-4,4))
                raw=F.grid_sample(u.abs().square()[:,None],grid,align_corners=False)[:,0]*255*gain
                raw=(raw+relay.photo_initial[1]+3*torch.tanh(relay.q[12])).clamp(0,255)
                values.extend([(relay.integrate(im)/relay.areas).cpu().numpy() for im in raw])
                if mm==0 and mask=='BASE':
                    ordinary,_=relay(anchors[0],bykey[(anchors[0],'BASE')]['dmd'],gray[mask])
                    identity=float((raw[0]-ordinary).abs().max());require(identity<1e-4,'zero axial offset numerical identity failed')
            v=np.array(values);mean=np.mean(((v-truth)/scales)**2);err=[];cos=[]
            for p,m in pairs:
                a=white@(truth[p]-truth[m]);b=white@(v[p]-v[m]);err.append(float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12)))
                cos.append(float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12)))
            results.append(dict(dz_mm=mm,mean_scaled_RMSE=float(np.sqrt(mean)),median_response_cosine=float(np.median(cos)),
                median_relative_response_error=float(np.median(err)),score=float(mean+.5*np.mean(np.square(err)))))
            all_values.append(v);print('AXIAL SCREEN',results[-1],flush=True)
    with (output/'READOUTS.npz').open('xb') as f:np.savez_compressed(f,truth=truth,predictions=all_values)
    write(output/'RESULT.json',dict(hypotheses=results,top5=sorted(results,key=lambda r:r['score'])[:5],identity_max_ADU=identity,
        no_parameter_fitting=True,no_hardware=True,actual_axial_position_NOT_identified=True,SLM_training_allowed=False))


def initialize(cfg,size,device):
    model=PupilRelay(checked(cfg['baseline']),cfg['layout'],checked(cfg['geometry']),device,
        pupil_size=size,phase_sign=cfg.get('phase_sign',1),slm_shift_bound=cfg.get('slm_shift_bound',32))
    with torch.no_grad():
        model.q.copy_(torch.tensor(cfg['q'],device=device));model.photo_initial[:]=torch.tensor(cfg['photometry'],device=device)
    model.q.requires_grad_(False)
    return model


def identity(initial,output,size):
    output.mkdir();cfg=read(initial/'MODEL.json');device=gpu();model=initialize(cfg,size,device)
    base=ASMRelay(checked(cfg['baseline']),cfg['layout'],checked(cfg['geometry']),device,
        phase_sign=cfg.get('phase_sign',1),slm_shift_bound=cfg.get('slm_shift_bound',32))
    with torch.no_grad():base.q.copy_(model.q);base.photo_initial.copy_(model.photo_initial)
    a=read(ROOT/'ASSETS_STRONG_01/PLAN.json');row=checked(a['catalogs']['PILOT'])['inputs'][0]
    dmd=checked(row['dmd'],True);gray=torch.tensor(checked(a['assets']['X4P']['patch'],True),device=device,dtype=torch.float32)
    with torch.no_grad():
        p,_=base(row['input_id'],dmd,gray);q,_=model(row['input_id'],dmd,gray);maximum=float((p-q).abs().max())
    weight=torch.tensor(np.random.default_rng(105).normal(size=64),device=device,dtype=torch.float32)
    def objective():
        raw,_=model(row['input_id'],dmd,gray)
        return ((model.integrate(raw)/model.areas)*weight).sum()
    gradient=torch.autograd.grad(objective(),model.pupil)[0];direction=torch.tensor(np.random.default_rng(7).normal(size=gradient.shape),device=device,dtype=torch.float32)
    direction=direction/direction.norm();expected=float((gradient*direction).sum())
    with torch.no_grad():
        model.pupil.copy_(.05*direction);plus=float(objective());model.pupil.copy_(-.05*direction);minus=float(objective());model.pupil.zero_()
    fd=(plus-minus)/.1;relative=abs(fd-expected)/max(abs(expected),1e-8)
    passed=maximum<1e-4 and relative<.1
    write(output/'RESULT.json',dict(initial=ref(initial/'MODEL.json'),source=ref(__file__),pupil_size=size,
        parameters=model.pupil.numel(),identity_max_ADU=maximum,directional_gradient=expected,finite_difference=fd,
        relative_gradient_error=relative,passed=passed,real_mapping_NOT_verified=True,no_hardware=True))
    require(passed,'coherent numerical QA failed');print('COHERENT QA',maximum,relative,flush=True)


def train(dataset,initial,output,size,steps):
    output.mkdir();start=time.monotonic();index=read(dataset/'INDEX.json');cfg=read(initial/'MODEL.json')
    require(index['baseline']==cfg['baseline'] and index['layout']==cfg['layout'],'coherent/data contract differs')
    require(cfg['source']['sha256']==ref(ROOT/'surrogate.py')['sha256'],'original physics source changed')
    samples=groups(index)
    require(not any(s['label'].get('new_platform_split')=='TEST' or s['mask_split']=='TEST' for s in samples),'TEST cannot enter fitting')
    ids=[i for i,s in enumerate(samples) if s['label'].get('new_platform_split','PILOT') in ('PILOT','TRAIN') and s['mask_split']=='TRAIN']
    device=gpu();relay=initialize(cfg,size,device);area=relay.areas
    targets=[torch.tensor(s['image'],device=device) for s in samples]
    gray=[torch.tensor(s['gray'],device=device,dtype=torch.float32) for s in samples]
    z=torch.stack([relay.integrate(t)/area for t in targets]);scale=z[[i for i in ids if samples[i]['mask_name']=='BASE']].std(0).clamp(min=.01)
    values=checked(index['features'],True)['mean_sums']/area.cpu().numpy();noise=[]
    for i in ids:
        v=values[samples[i]['visits']];noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    from response import covariance
    _,white=covariance(noise);white=torch.tensor(white,device=device,dtype=torch.float32)
    lookup={(s['input_id'],tuple(s['coefficients'])):i for i,s in enumerate(samples)};keep=set(ids);pairs=[]
    for i in ids:
        c=samples[i]['coefficients'];j=lookup.get((samples[i]['input_id'],tuple(-x for x in c)))
        if any(c) and j in keep and j>i:pairs.append((i,j))
    require(len(ids)>20 and len(pairs)>8,'adequate TRAIN-only mean and response coverage required')
    archive_sources(output);src=output/'SOURCE/coherent_residual.py'
    with src.open('xb') as f:f.write(Path(__file__).read_bytes())
    write(output/'PLAN.json',dict(dataset=ref(dataset/'INDEX.json'),initial=ref(initial/'MODEL.json'),source=ref(src),
        pupil_size=size,additional_parameters=2*size*size,steps=steps,TRAIN=ids,response_pairs=pairs,
        TRAIN_only=True,physical_original13_frozen=True,no_SLM_change=True,no_hardware=True,
        effective_pupil_NOT_uniquely_identified_component=True))
    rng=np.random.default_rng(105430);optimizer=torch.optim.Adam([relay.pupil],lr=.015)
    best=float('inf');state=None;history=[]
    def predict(i):
        r=samples[i];return relay(r['input_id'],r['dmd'],gray[i])[0]
    def score():
        with torch.no_grad():
            zz={i:relay.integrate(predict(i))/area for i in ids}
            mean=torch.stack([((zz[i]-z[i])/scale).square().mean() for i in ids]).mean()
            response=[]
            for p,m in pairs:
                a=white@(z[p]-z[m]);b=white@(zz[p]-zz[m]-z[p]+z[m])
                floor=64*(1/len(samples[p]['visits'])+1/len(samples[m]['visits']))
                response.append(b.square().sum()/a.square().sum().clamp(min=floor))
            return float(mean+.5*torch.stack(response).mean())
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True);i=int(rng.choice(ids));raw=predict(i)
        mean=((relay.integrate(raw)/area-z[i])/scale).square().mean()
        pixel=((1+4*(targets[i]>11).float())*(raw-targets[i]).square()).mean()
        smooth=(relay.pupil[:,1:]-relay.pupil[:,:-1]).square().mean()+(relay.pupil[:,:,1:]-relay.pupil[:,:,:-1]).square().mean()
        loss=mean+.2*pixel+.001*smooth+.0001*relay.pupil.square().mean();loss.backward();del loss,raw
        p,m=pairs[int(rng.integers(len(pairs)))];dp=relay.integrate(predict(p))/area-relay.integrate(predict(m))/area
        a=white@(z[p]-z[m]);b=white@(dp-z[p]+z[m]);floor=64*(1/len(samples[p]['visits'])+1/len(samples[m]['visits']))
        response=.5*b.square().sum()/a.square().sum().clamp(min=floor);response.backward()
        torch.nn.utils.clip_grad_norm_([relay.pupil],5);optimizer.step();del response,dp
        if step%100==0 or step==steps-1:
            value=score();history.append(dict(step=step,TRAIN_score=value,seconds=time.monotonic()-start))
            if value<best:best=value;state=relay.pupil.detach().cpu().clone()
            print('COHERENT PUPIL',size,step,value,flush=True)
    with torch.no_grad():relay.pupil.copy_(state)
    new=dict(cfg);new.update(model_family='bounded_complex_pupil',implementation=ref(src),pupil=state.tolist(),
        pupil_size=size,initial_model=ref(initial/'MODEL.json'),calibration_dataset=ref(dataset/'INDEX.json'))
    write(output/'MODEL.json',new);write(output/'HISTORY.json',history)
    with torch.no_grad():prediction=np.stack([predict(i).cpu().numpy() for i in range(len(samples))])
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,predicted=prediction,truth=np.stack([s['image'] for s in samples]))
    write(output/'RESULT.json',dict(model=ref(output/'MODEL.json'),predictions=ref(output/'PREDICTIONS.npz'),
        best_TRAIN_score=best,seconds=time.monotonic()-start,no_SLM_optimization=True,no_deployment=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dataset');p.add_argument('--identity',action='store_true');p.add_argument('--size',type=int,default=8)
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--sampling-screen',action='store_true');p.add_argument('--axial-screen',action='store_true');a=p.parse_args()
    require(Path(a.output).name==a.output and 1<=a.steps<=1600,'explicit fresh bounded offline experiment')
    if a.axial_screen:axial_screen(ROOT/a.initial,ROOT/a.output)
    elif a.sampling_screen:sampling_screen(ROOT/a.initial,ROOT/a.output)
    elif a.identity:identity(ROOT/a.initial,ROOT/a.output,a.size)
    else:
        require(a.dataset is not None,'explicit dataset required');train(ROOT/a.dataset,ROOT/a.initial,ROOT/a.output,a.size,a.steps)
