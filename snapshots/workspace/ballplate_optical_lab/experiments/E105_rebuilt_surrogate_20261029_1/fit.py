"""Offline physics calibration and conditional mean fit; explicit dataset only.

No hardware imports. Calibration pilot scores are not deployment acceptance.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from workflow import ROOT,LAB,read,write,ref,require
from dataset import checked
from surrogate import ASMRelay,ResidualMean,residual_features


def gpu():
    ids=[i for i in range(torch.cuda.device_count()) if '3070' in torch.cuda.get_device_name(i)]
    require(len(ids)==1,'RTX3070 required');torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.65,ids[0]);return 'cuda:'+str(ids[0])


def groups(index):
    raw=checked(index['means'],True)['images'];grouped=defaultdict(list)
    inputs={};masks={}
    for i,row in enumerate(index['rows']):
        identity=row['input_id'];digest=row['label']['dmd']['sha256']
        require(identity not in inputs or inputs[identity]==digest,'input ID aliases different DMD pixels')
        inputs[identity]=digest
        if row.get('mask_name'):
            name=row['mask_name'];sha=row['mask']['sha256']
            require(name not in masks or masks[name]==sha,'mask name aliases different amplitudes/bytes')
            masks[name]=sha
        grouped[(identity,row['mask']['sha256'])].append(i)
    result=[]
    assets=read(ROOT/'ASSETS_02/PLAN.json')['assets']
    bysha={v['patch']['sha256']:(k,v) for k,v in assets.items()}
    for key,ids in grouped.items():
        row=index['rows'][ids[0]]
        if row.get('assets_plan') is not None:
            assetplan=checked(row['assets_plan']);name=row['mask_name'];asset=assetplan['assets'][name]
            require(asset['patch']==row['mask'] and asset['coefficients']==row['coefficients'],'mask coefficient contract differs')
        else:name,asset=bysha[row['mask']['sha256']]
        result.append(dict(input_id=key[0],mask_name=name,mask=row['mask'],label=row['label'],
            coefficients=asset['coefficients'],mask_split=asset.get('split','TRAIN'),image=raw[ids].mean(0),visits=ids,
            dmd=checked(row['label']['dmd'],True),gray=checked(row['mask'],True)))
    return result


def archive_sources(output):
    folder=output/'SOURCE';folder.mkdir();sources=[]
    for name in ('fit.py','surrogate.py','dataset.py','workflow.py','residual_fit.py','response.py','design.py','series.py','head.py','validate_physics.py'):
        p=ROOT/name;sources.append(ref(p))
        with (folder/name).open('xb') as f:f.write(p.read_bytes())
    return sources


def physical(dataset,output,steps,seed=105,phase_sign=1,response_weight=1.,initial_model=None,relative_response=False,lr=.012):
    require(1<=steps<=3000,'bounded fit steps');output.mkdir();start=time.monotonic();device=gpu()
    index=read(dataset/'INDEX.json');baseline=checked(index['baseline']);geometry=checked(baseline['geometry'])
    initial=None if initial_model is None else read(initial_model/'MODEL.json')
    if initial is not None:
        require(initial['baseline']==index['baseline'] and initial['layout']==index['layout'],'initial model contract differs')
        phase_sign=initial.get('phase_sign',1)
    samples=groups(index);relay=ASMRelay(baseline,index['layout'],geometry,device,phase_sign=phase_sign,
        slm_shift_bound=32. if initial is None else initial.get('slm_shift_bound',32.))
    # First calibration uses four named development anchors. Rest is a pilot
    # diagnostic, NOT untouched formal TEST (already observed at baseline).
    formal=any(r['label'].get('new_platform_split') in ('TRAIN','SELECT','TEST') for r in samples)
    if formal:
        require(not any(r['label'].get('new_platform_split')=='TEST' or r['mask_split']=='TEST' for r in samples),
                'formal TEST cannot enter physical calibration')
        train=[i for i,r in enumerate(samples) if r['label'].get('new_platform_split','PILOT') in ('TRAIN','PILOT')
               and r['mask_split']=='TRAIN']
        train_names={samples[i]['input_id'] for i in train}
    else:
        train_names={'CENTER','X_POS12','Y_NEG12','VY_POS'}
        train=[i for i,r in enumerate(samples) if r['input_id'] in train_names]
    lookup={(r['input_id'],r['mask_name']):i for i,r in enumerate(samples)}
    pairs=[]
    for input_id in sorted(train_names):
        for axis in ('X4','Y4','X8','Y8'):
            p,m=lookup.get((input_id,axis+'P')),lookup.get((input_id,axis+'M'))
            if p in train and m in train:pairs.append((p,m))
    require(len(train)>=4,'four explicit anchors required')
    targets=[torch.tensor(r['image'],device=device) for r in samples]
    grays=[torch.tensor(r['gray'],dtype=torch.float32,device=device) for r in samples]
    area=relay.areas
    zz=torch.stack([relay.integrate(targets[i])/area for i in train])
    scales=zz.std(0).clamp(min=.01).detach()
    whitener=None
    if relative_response:
        from response import covariance
        features=checked(index['features'],True)['mean_sums']/area.cpu().numpy();res=[]
        for j in train:
            visits=samples[j]['visits'];z=features[visits];res.extend((z-z.mean(0))*np.sqrt(len(z)/(len(z)-1)))
        _,w=covariance(res);whitener=torch.tensor(w,device=device,dtype=torch.float32)
    # Fit only shared light gain / background before bounded spatial fitting.
    with torch.no_grad():
        relay.photo_initial[:]=torch.tensor([.001,0.],device=device)
        sx=sy=sxx=sxy=total=0.
        for i in train:
            r=samples[i];signal,_=relay(r['input_id'],r['dmd'],grays[i]);x=(signal/.001)[::3,::3].double();y=targets[i][::3,::3].double()
            sx+=x.sum().item();sy+=y.sum().item();sxx+=x.square().sum().item();sxy+=(x*y).sum().item();total+=x.numel()
        gain=max(1e-5,(sxy-sx*sy/total)/max(sxx-sx*sx/total,1e-12));bg=(sy-gain*sx)/total
        relay.photo_initial[:]=torch.tensor([gain,bg],device=device)
        if initial is not None:
            relay.q.copy_(torch.tensor(initial['q'],device=device));relay.photo_initial[:]=torch.tensor(initial['photometry'],device=device)
    initial_photo=relay.photo_initial.detach().cpu().tolist()
    write(output/'PLAN.json',dict(dataset=ref(dataset/'INDEX.json'),sources=archive_sources(output),
        model='same-grid two-lens ASM, 13 bounded effective corrections',steps=steps,seed=seed,
        train_input_ids=sorted(train_names),initial_photometry=initial_photo,
        calibration_not_independent_TEST=True,no_hardware=True,no_SLM_training=True,
        formal_train_partition=bool(formal),
        phase_sign_hypothesis=phase_sign,response_pairs=pairs,response_weight=response_weight,
        learning_rate=lr,relative_response=relative_response,
        initial_model=None if initial_model is None else ref(initial_model/'MODEL.json'),
        prior_old_measurement_labels_used=False,physics_identity=ref(ROOT/'PHYSICS_QA_01/RESULT.json')))
    def predict():
        with torch.no_grad():
            return np.stack([relay(r['input_id'],r['dmd'],grays[i])[0].cpu().numpy() for i,r in enumerate(samples)])
    before=predict();rng=np.random.default_rng(seed);torch.manual_seed(seed)
    optim=torch.optim.Adam([relay.q],lr=lr);history=[];best=float('inf');bestq=None
    def response_loss(pair):
        predictions=[];truth=[]
        for j in pair:
            rr=samples[j];pp=relay(rr['input_id'],rr['dmd'],grays[j])[0]
            predictions.append(relay.integrate(pp)/area);truth.append(relay.integrate(targets[j])/area)
        delta=predictions[0]-predictions[1];real=truth[0]-truth[1]
        if whitener is None:return ((delta-real)/scales).square().mean()
        dr=whitener@real;error=whitener@(delta-real)
        denominator=dr.square().sum()+64*(1/len(samples[pair[0]]['visits'])+1/len(samples[pair[1]]['visits']))
        return error.square().sum()/denominator
    for step in range(steps):
        i=int(rng.choice(train));r=samples[i];optim.zero_grad(set_to_none=True)
        predicted,_=relay(r['input_id'],r['dmd'],grays[i])
        region=((relay.integrate(predicted)/area-relay.integrate(targets[i])/area)/scales).square().mean()
        # TRAIN target-dependent weighting belongs to loss only, never input.
        weights=1+4*(targets[i]>initial_photo[1]+.5).float()
        pixel=(weights*(predicted-targets[i]).square()).mean()
        response=torch.zeros((),device=device)
        if pairs:
            response=response_loss(pairs[int(rng.integers(len(pairs)))])
        loss=region+.2*pixel+response_weight*response+.0001*relay.q[:11].square().mean()
        require(torch.isfinite(loss).item(),'nonfinite physics loss');loss.backward()
        torch.nn.utils.clip_grad_norm_([relay.q],5);optim.step()
        if step%20==0 or step==steps-1:
            with torch.no_grad():
                score=np.mean([float((((relay.integrate(relay(samples[j]['input_id'],samples[j]['dmd'],grays[j])[0])
                    -relay.integrate(targets[j]))/area)/scales).square().mean()) for j in train])
                if relative_response and pairs:score+=response_weight*np.mean([float(response_loss(pair)) for pair in pairs])
            row=dict(step=step,loss=float(loss.detach()),train_region_scaled_MSE=float(score),q=relay.q.detach().cpu().tolist(),seconds=time.monotonic()-start)
            history.append(row);print('physics',step,'TRAIN scaled',round(score,6),'seconds',round(row['seconds'],1),flush=True)
            if score<best:best=score;bestq=relay.q.detach().clone()
    with torch.no_grad():relay.q.copy_(bestq)
    after=predict();truth=np.stack([r['image'] for r in samples])
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,before=before,after=after,truth=truth)
    write(output/'MODEL.json',dict(q=relay.q.detach().cpu().tolist(),photometry=relay.photo_initial.cpu().tolist(),
        baseline=index['baseline'],layout=index['layout'],geometry=baseline['geometry'],source=ref(output/'SOURCE/surrogate.py'),
        phase_LUT_calibrated=False,phase_sign=phase_sign,slm_shift_bound=relay.slm_shift_bound,noise_fitted=False,local_mask_space_only=True))
    write(output/'HISTORY.json',history)
    metrics=[]
    from dataset import integrate
    area_np=area.cpu().numpy()
    for i,r in enumerate(samples):
        def measure(pred):
            e=pred-truth[i];de=integrate(e,index['layout'])/area_np
            return dict(pixel_RMSE_ADU=float(np.sqrt(np.mean(e**2))),region_mean_RMSE_ADU=float(np.sqrt(np.mean(de**2))))
        metrics.append(dict(input_id=r['input_id'],mask=r['mask_name'],is_fit_anchor=r['input_id'] in train_names,
            before=measure(before[i]),after=measure(after[i])))
    write(output/'RESULT.json',dict(completed=True,metrics=metrics,seconds=time.monotonic()-start,
        best_train_scaled_MSE=best,model=ref(output/'MODEL.json'),predictions=ref(output/'PREDICTIONS.npz'),
        deployment_passed=False,SLM_optimized=False))
    plots(output,metrics,before,after,truth,samples)


def plots(out,metrics,before,after,truth,samples):
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(3,4,figsize=(12,8))
    for i in range(min(3,len(samples))):
        for j,a in enumerate((truth[i],before[i],after[i],after[i]-truth[i])):
            axes[i,j].imshow(a,cmap='RdBu_r' if j==3 else 'gray',vmin=-5 if j==3 else 8,vmax=5 if j==3 else 25)
            axes[i,j].axis('off');axes[i,j].set_title(samples[i]['input_id']+' / '+('real','physics seed','physics fit','fit - real')[j])
    fig.suptitle('New platform DEVELOPMENT calibration | native ADU, no image-wise rescale')
    fig.tight_layout();save_figure(fig,out/'01_physics_images',180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3));x=np.arange(len(metrics))
    for key,color in [('before','#D55E00'),('after','#0072B2')]:
        ax.plot(x,[m[key]['region_mean_RMSE_ADU'] for m in metrics],'o-',color=color,label=key)
    ax.set_xticks(x,[m['input_id']+'/'+m['mask'] for m in metrics],rotation=70,fontsize=6)
    ax.set_ylabel('64 region mean RMSE / ADU');ax.legend();fig.tight_layout();save_figure(fig,out/'02_region_error',180);plt.close(fig)


def scan_centers(dataset,seed_model,output):
    """File-only discrete effective-frequency-origin hypotheses, not alignment.

    A centered illumination wave is an assumption, not a measured carrier.
    Change only virtual frequency origin/sign; never the displayed SLM position.
    """
    from response import covariance
    output.mkdir();start=time.monotonic();device=gpu();index=read(dataset/'INDEX.json');cfg=read(seed_model/'MODEL.json')
    require(index['baseline']==cfg['baseline'] and index['layout']==cfg['layout'],'scan contracts differ')
    samples=groups(index);b=checked(index['baseline']);layout=index['layout'];geometry=checked(cfg['geometry'])
    relay=ASMRelay(b,layout,geometry,device,slm_shift_bound=256.)
    names={'CENTER','X_POS12','Y_NEG12','VY_POS'};ids=[i for i,r in enumerate(samples) if r['input_id'] in names]
    area=relay.areas.cpu().numpy();truth=np.stack([integrate_numpy(r['image'],layout)/area for r in samples])
    single=checked(index['features'],True)['mean_sums']/area;res=[]
    for i in ids:
        visits=samples[i]['visits'];v=single[visits];res.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    cov,white=covariance(res);lookup={(r['input_id'],r['mask_name']):i for i,r in enumerate(samples)}
    pairs=[(lookup[(inp,axis+'P')],lookup[(inp,axis+'M')]) for inp in sorted(names) for axis in ('X4','Y4','X8','Y8')]
    expected=np.stack([white@(truth[p]-truth[m]) for p,m in pairs]);den=np.sum(expected**2,axis=1)+64
    grid=list(range(-240,241,60));write(output/'PLAN.json',dict(dataset=ref(dataset/'INDEX.json'),seed_model=ref(seed_model/'MODEL.json'),
        sources=archive_sources(output),train_ids=ids,train_input_ids=sorted(names),shift_grid_px=grid,
        phase_signs=[1,-1],response_pair_count=len(pairs),only_virtual_coordinates=True,no_hardware=True,
        actual_illumination_tilt_or_SLMCentre_NOT_identified=True))
    grays=[torch.tensor(r['gray'],device=device,dtype=torch.float32) for r in samples]
    results=[];best=None
    with torch.no_grad():
        relay.q.copy_(torch.tensor(cfg['q'],device=device));relay.photo_initial[:]=torch.tensor(cfg['photometry'],device=device)
        for sign in (1,-1):
            relay.phase_sign=sign
            for y in grid:
                for x in grid:
                    relay.q[2:4]=torch.atanh(torch.tensor([x/256,y/256],device=device))
                    prediction={i:relay.integrate(relay(samples[i]['input_id'],samples[i]['dmd'],grays[i])[0]).cpu().numpy()/area for i in ids}
                    delta=np.stack([white@(prediction[p]-prediction[m]) for p,m in pairs])
                    response=float(np.mean(np.sum((delta-expected)**2,axis=1)/den))
                    mean=float(np.mean([(prediction[i]-truth[i])**2 for i in ids]))
                    cos=np.sum(delta*expected,axis=1)/np.maximum(np.linalg.norm(delta,axis=1)*np.linalg.norm(expected,axis=1),1e-12)
                    score=response+.1*mean/.01
                    row=dict(sign=sign,x=x,y=y,score=score,response_error=response,pixel_region_MSE=mean,
                        median_response_cosine=float(np.median(cos)))
                    results.append(row)
                    if best is None or score<best['score']:
                        best=dict(row=row,q=relay.q.cpu().tolist(),score=score)
                print('virtual frequency scan',sign,y,'best',best['row'],flush=True)
    write(output/'SCAN.json',results)
    final=dict(cfg,q=best['q'],phase_sign=best['row']['sign'],slm_shift_bound=256.,source=ref(output/'SOURCE/surrogate.py'))
    write(output/'MODEL.json',final)
    write(output/'RESULT.json',dict(best=best['row'],model=ref(output/'MODEL.json'),seconds=time.monotonic()-start,
        discrete_candidates=len(results),development_ONLY=True,physical_validity_NOT_passed=True,no_hardware=True))


def integrate_numpy(image,layout):
    from dataset import integrate
    return integrate(image,layout)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='BASE_AUDIT_01');p.add_argument('--output',default='PHYSICAL_01')
    p.add_argument('--steps',type=int,default=200);p.add_argument('--phase-sign',type=int,choices=(-1,1),default=1)
    p.add_argument('--response-weight',type=float,default=1.);p.add_argument('--scan-centers',action='store_true')
    p.add_argument('--seed-model',default='PHYSICAL_PLUS_02');p.add_argument('--initial-model')
    p.add_argument('--relative-response',action='store_true');p.add_argument('--lr',type=float,default=.012);a=p.parse_args()
    require(Path(a.dataset).name==a.dataset and Path(a.output).name==a.output,'explicit direct folders required')
    if a.scan_centers:scan_centers(ROOT/a.dataset,ROOT/a.seed_model,ROOT/a.output)
    else:physical(ROOT/a.dataset,ROOT/a.output,a.steps,phase_sign=a.phase_sign,response_weight=a.response_weight,
        initial_model=None if a.initial_model is None else ROOT/a.initial_model,relative_response=a.relative_response,lr=a.lr)
