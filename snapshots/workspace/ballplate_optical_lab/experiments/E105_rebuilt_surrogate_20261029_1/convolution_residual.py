"""Offline, shared shift-equivariant intensity residual over full ASM.

The physical coherent model remains the baseline. A learned convolution acts
on a simulated zero-phase focused input, approximating only the residual mean
for this isolated four-ball family. It is NOT a claim that coherent optics is
linear in intensity, and it never runs in the deployed control path.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from workflow import ROOT,read,write,ref,require
from dataset import checked,integrate
from fit import gpu,archive_sources
from residual_fit import relay_from_config,region_means,pilot_split,signed_pairs


class FocusConvolution(torch.nn.Module):
    def __init__(self,size=65,parameter_size=None):
        super().__init__();require(size in (5,33,65,129,257),'declared odd residual support required')
        parameter_size=size if parameter_size is None else parameter_size
        require(parameter_size in (5,17,33,65,129) and parameter_size<=size,'bounded coarse kernel required')
        self.size=size;self.parameter_size=parameter_size
        self.kernels=torch.nn.Parameter(torch.zeros(5,parameter_size,parameter_size))

    def forward(self,spectra,coefficients,native_shape,fft_shape):
        c=torch.cat((torch.ones_like(coefficients[:,:1]),coefficients[:,:4]),1)
        # Parameter-step integrated gain is independent of kernel support.
        kernels=self.kernels
        if self.parameter_size!=self.size:
            kernels=F.interpolate(kernels[:,None],size=(self.size,self.size),mode='bilinear',align_corners=False)[:,0]
        k=torch.einsum('bk,khw->bhw',c,kernels)/self.size**2
        fk=torch.fft.rfft2(k,s=fft_shape);out=torch.fft.irfft2(spectra*fk,s=fft_shape)
        r=self.size//2;h,w=native_shape
        return out[:,r:r+h,r:r+w]


def focus_cache(cache,output):
    output.mkdir();index=read(cache/'INDEX.json');cfg=checked(index['physical_model']);relay=relay_from_config(cfg,gpu())
    require('implementation' not in cfg,'focus diagnostic currently uses the original calibrated 13-parameter model')
    lookup={};images=[];rowids=[]
    with torch.no_grad():
        relay.q[8:11]=0 # Remove fitted defocus/astigmatism ONLY from the known input template.
        bg=float(relay.photo_initial[1]+3*torch.tanh(relay.q[12]));gray=torch.zeros(75,75,device=relay.device)
        for row in index['rows']:
            sha=row['dmd_reference']['sha256']
            if sha not in lookup:
                raw,_=relay(row['input_id'],checked(row['dmd_reference'],True),gray)
                images.append((raw-bg).clamp(min=0).cpu().numpy());lookup[sha]=len(images)-1
            rowids.append(lookup[sha])
    with (output/'FOCUS.npz').open('xb') as f:np.savez_compressed(f,images=np.asarray(images),row_indices=rowids)
    write(output/'INDEX.json',dict(cache=ref(cache/'INDEX.json'),focus=ref(output/'FOCUS.npz'),source=ref(__file__),
        input_SHA_order=list(lookup),simulated_zero_phase_only=True,measured_CCD_NOT_input=True,
        wavefront_removed_from_template_only=True,baseline_physics_unchanged=True,no_hardware=True))


def train(cache,focus,output,size=65,steps=3000,seed=105,parameter_size=None):
    from scipy.fft import next_fast_len
    from response import covariance
    output.mkdir();start=time.monotonic();device=gpu();torch.manual_seed(seed);rng=np.random.default_rng(seed)
    idx=read(cache/'INDEX.json');fi=read(focus/'INDEX.json');require(fi['cache']==ref(cache/'INDEX.json'),'focus/cache contract differs')
    arrays=checked(idx['cache'],True);ff=checked(fi['focus'],True);rows=idx['rows'];layout=idx['layout'];split=pilot_split(rows)
    train=np.array([i for i,s in enumerate(split) if s=='TRAIN']);select=np.array([i for i,s in enumerate(split) if s!='TRAIN'])
    true=torch.tensor(arrays['truth'],device=device);physical=torch.tensor(arrays['physics'],device=device)
    ztrue=torch.cat([region_means(true[i:i+4],layout) for i in range(0,len(rows),4)])
    base=[i for i in train if rows[i]['mask']=='BASE'];scales=ztrue[base].std(0).clamp(min=.01)
    native=true.shape[-2:];fftshape=tuple(next_fast_len(n+size-1) for n in native)
    focusimages=torch.tensor(ff['images'],device=device);trainfocus=sorted(set(ff['row_indices'][train]))
    focus_scale=float(focusimages[trainfocus].square().mean().sqrt().clamp(min=1e-6))
    spectra=torch.fft.rfft2(focusimages/focus_scale,s=fftshape);rowids=torch.tensor(ff['row_indices'],device=device)
    coefficients=torch.tensor([r['coefficients_raw_gray'] for r in rows],device=device,dtype=torch.float32)/24
    data=checked(idx['dataset']);area=np.array([r['pixels'] for r in layout['regions']]);zz=checked(data['features'],True)['mean_sums']/area;noise=[]
    for i in train:
        v=zz[rows[i]['visits']];noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    cov,white=covariance(noise);white=torch.tensor(white,device=device,dtype=torch.float32)
    tset=set(train);sset=set(select);pairs=signed_pairs(rows)
    tp=[(p,m) for p,m in pairs if p in tset and m in tset];sp=[(p,m) for p,m in pairs if p in sset and m in sset]
    require(len(tp)>10 and len(sp)>5,'paired TRAIN and SELECT responses required')
    net=FocusConvolution(size,parameter_size).to(device);optimizer=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=1e-5)
    archive_sources(output);source=output/'SOURCE/convolution_residual.py'
    with source.open('xb') as f:f.write(Path(__file__).read_bytes())
    write(output/'PLAN.json',dict(cache=ref(cache/'INDEX.json'),focus=ref(focus/'INDEX.json'),fit_source=ref(source),
        train=train.tolist(),select=select.tolist(),split=split,seed=seed,steps=steps,architecture='focus_convolution',
        kernel_size=size,kernel_parameter_size=net.parameter_size,kernel_parameter_scale='1 / size**2',learning_rate=.002,
        fft_shape=list(fftshape),focus_scale=focus_scale,selection_criterion='mean_response',
        state_scales_ADU=scales.cpu().tolist(),response_pairs=tp,selection_response_pairs=sp,
        assumption='shared shift-equivariant residual for isolated four-ball images; must pass held-input validation',
        physical_coherent_model_retained=True,no_state_truth_inputs=True,no_online_image_network=True,formal_TEST=False))
    def predict(ids):
        residual=net(spectra[rowids[ids]],coefficients[ids],native,fftshape)
        return (physical[ids]+residual).clamp(0,255),residual
    def response_loss(v,p,m):
        a=white@(ztrue[p]-ztrue[m]);e=white@(v[0]-v[1]-ztrue[p]+ztrue[m])
        floor=64*(1/len(rows[p]['visits'])+1/len(rows[m]['visits']))
        return e.square().sum()/a.square().sum().clamp(min=floor)
    def score():
        with torch.no_grad():
            images=[];vectors=[]
            for i in range(0,len(rows),4):
                p=predict(list(range(i,min(i+4,len(rows)))))[0]
                vectors.append(region_means(p,layout));images.append(p.cpu().numpy())
            pred=np.concatenate(images);v=torch.cat(vectors);error=((v-ztrue)/scales).square().mean(-1)
            response=torch.stack([response_loss(v[[p,m]],p,m) for p,m in sp]).mean()
            return float(error[select].mean()+.5*response),pred,error,float(response)
    value,_,error,rr=score()
    history=[dict(step=-1,SELECT_criterion=value,SELECT_scaled_MSE=float(error[select].mean()),
        SELECT_relative_response_MSE=rr,TRAIN_scaled_MSE=float(error[train].mean()),seconds=time.monotonic()-start)]
    best=value;beststep=-1;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    for step in range(steps):
        ids=rng.choice(train,size=4,replace=False).tolist();optimizer.zero_grad(set_to_none=True)
        pred,residual=predict(ids);v=region_means(pred,layout)
        mean=((v-ztrue[ids])/scales).square().mean()
        pixel=((1+4*(true[ids]>11).float())*(pred-true[ids]).square()).mean()
        p,m=tp[int(rng.integers(len(tp)))];pr,_=predict([p,m]);resp=response_loss(region_means(pr,layout),p,m)
        smooth=(net.kernels[:,1:]-net.kernels[:,:-1]).square().mean()+(net.kernels[:,:,1:]-net.kernels[:,:,:-1]).square().mean()
        loss=mean+.2*pixel+.5*resp+.001*smooth;loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
        if step%100==0 or step==steps-1:
            value,_,error,rr=score();history.append(dict(step=step,SELECT_criterion=value,SELECT_scaled_MSE=float(error[select].mean()),
                SELECT_relative_response_MSE=rr,TRAIN_scaled_MSE=float(error[train].mean()),seconds=time.monotonic()-start))
            if value<best:best=value;beststep=step;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            print('FOCUS CONV',size,seed,step,'SELECT',round(value,6),flush=True)
    net.load_state_dict(state);_,pred,_,_=score()
    with (output/'STATE.pt').open('xb') as f:torch.save(state,f)
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,predicted=pred,physics=arrays['physics'],truth=arrays['truth'])
    write(output/'HISTORY.json',history);metrics=[]
    for role in dict.fromkeys(split):
        ids=[i for i,s in enumerate(split) if s==role];item=dict(split=role,conditions=len(ids))
        for name,est in [('physics',arrays['physics']),('hybrid',pred)]:
            e=est[ids]-arrays['truth'][ids];ze=np.stack([integrate(im,layout)/area for im in e])
            item[name]=dict(region_RMSE_ADU=float(np.sqrt(np.mean(ze**2))),TRAIN_scaled_region_RMSE=float(np.sqrt(np.mean((ze/scales.cpu().numpy())**2))),
                pixel_RMSE_ADU=float(np.sqrt(np.mean(e**2))))
        metrics.append(item)
    write(output/'RESULT.json',dict(state=ref(output/'STATE.pt'),predictions=ref(output/'PREDICTIONS.npz'),seed=seed,best_step=beststep,
        metrics=metrics,model_parameters=sum(p.numel() for p in net.parameters()),seconds=time.monotonic()-start,
        deployed=False,SLM_optimized=False,formal_TEST=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--build-focus',action='store_true');p.add_argument('--focus');p.add_argument('--size',type=int,default=65)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--seed',type=int,default=105);p.add_argument('--parameter-size',type=int);a=p.parse_args()
    require(Path(a.output).name==a.output and 1<=a.steps<=4000,'bounded explicit offline experiment')
    if a.build_focus:focus_cache(ROOT/a.cache,ROOT/a.output)
    else:
        require(a.focus is not None,'explicit focused input cache required');train(ROOT/a.cache,ROOT/a.focus,ROOT/a.output,a.size,a.steps,a.seed,a.parameter_size)
