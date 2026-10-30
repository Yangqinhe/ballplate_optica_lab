"""Offline conditional mean modeling; never an online controller or camera owner."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from workflow import ROOT,read,write,ref,require,load
from dataset import checked,integrate
from fit import gpu,groups,archive_sources
from surrogate import ASMRelay,ResidualMean,residual_features


class ResponseMean(torch.nn.Module):
    """Local residual: input-conditioned offset + four explicit response maps.

    Uses BASE simulated field only, never measured state/CCD as an input. The
    nonlinear physical prediction is retained separately. Coefficients arriving
    from the cache are raw gray / 8; response units below are raw gray / 24.
    """
    def __init__(self,quadratic=False,context=False,full_quadratic=False):
        super().__init__()
        self.quadratic=quadratic or full_quadratic;self.context=context;self.full_quadratic=full_quadratic
        self.c1=torch.nn.Conv2d(14 if context else 5,16,3,padding=1);self.c2=torch.nn.Conv2d(16,16,3,padding=1)
        self.c3=torch.nn.Conv2d(16,32,3,stride=2,padding=1);self.c4=torch.nn.Conv2d(32,32,3,padding=1)
        self.c5=torch.nn.Conv2d(48,16,3,padding=1);self.c6=torch.nn.Conv2d(16,15 if full_quadratic else 9 if self.quadratic else 5,3,padding=1)
        torch.nn.init.zeros_(self.c6.weight);torch.nn.init.zeros_(self.c6.bias)

    def forward(self,features,coefficients):
        if self.context:
            h,w=features.shape[-2:];channels=[features]
            for scale in (8,32,128):
                low=F.adaptive_avg_pool2d(features[:,:3],(max(1,h//scale),max(1,w//scale)))
                channels.append(F.interpolate(low,size=(h,w),mode='bilinear',align_corners=False))
            features=torch.cat(channels,1)
        a=F.silu(self.c2(F.silu(self.c1(features))))
        b=F.silu(self.c4(F.silu(self.c3(a))))
        b=F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False)
        maps=self.c6(F.silu(self.c5(torch.cat((a,b),1))))
        c=coefficients[:,:4,None,None]/3
        residual=maps[:,:1]+(maps[:,1:5]*c).sum(1,keepdim=True)
        if self.quadratic:residual=residual+(maps[:,5:9]*c.square()).sum(1,keepdim=True)
        if self.full_quadratic:
            cross=torch.stack([c[:,i]*c[:,j] for i in range(4) for j in range(i+1,4)],1)
            residual=residual+(maps[:,9:15]*cross).sum(1,keepdim=True)
        return residual


ARCHITECTURES=('film','response','response_quadratic','response_context','response_cross_context')


def make_network(kind,module=None):
    require(kind in ARCHITECTURES,'unknown frozen architecture')
    klass=ResponseMean if module is None else module.ResponseMean
    if kind=='film':return ResidualMean() if module is None else module.ResidualMean()
    if kind=='response':return klass()
    if kind=='response_quadratic':return klass(quadratic=True)
    if kind=='response_context':return klass(context=True)
    return klass(context=True,full_quadratic=True)


def relay_from_config(cfg,device,padding=None):
    """Instantiate the explicitly archived effective-physics implementation."""
    require(ref(ROOT/'surrogate.py')['sha256']==cfg['source']['sha256'],'original ASM source changed')
    klass=ASMRelay
    if cfg.get('implementation') is not None:
        source=cfg['implementation'];require(ref(source['path'])==source,'physical extension source changed')
        require(cfg.get('model_family')=='bounded_source_lut','unknown physical extension')
        module=load(Path(source['path']),'e105_field_'+source['sha256'][:12]);klass=module.SourceLUTRelay
    relay=klass(checked(cfg['baseline']),cfg['layout'],checked(cfg['geometry']),device,padding=padding,
        phase_sign=cfg.get('phase_sign',1),slm_shift_bound=cfg.get('slm_shift_bound',32))
    with torch.no_grad():
        relay.q.copy_(torch.tensor(cfg['q'],device=device));relay.photo_initial[:]=torch.tensor(cfg['photometry'],device=device)
        if cfg.get('implementation') is not None:
            require(len(cfg['extra'])==10 and np.isfinite(cfg['extra']).all(),'finite frozen physical correction required')
            relay.extra.copy_(torch.tensor(cfg['extra'],device=device))
    return relay


class FrozenProxy:
    """Differentiable frozen optical surrogate; no control policy, SDK or display.

    The BASE feature cache is independent of the mask being optimized. Actual
    mask-dependent propagation is recomputed every call, never stale-cached.
    """
    def __init__(self,model,device=None):
        self.model=Path(model);self.plan=read(self.model/'PLAN.json');result=read(self.model/'RESULT.json')
        self.cache=checked(self.plan['cache']);cfg=checked(self.cache['physical_model'])
        require(ref(ROOT/'surrogate.py')['sha256']==cfg['source']['sha256'],'frozen physical implementation changed')
        src=self.model/'SOURCE/residual_fit.py'
        require(ref(src)['sha256']==self.plan['fit_source']['sha256'],'frozen residual source missing')
        module=load(src,'e105_proxy_frozen_'+str(self.plan['seed']))
        self.device=device or gpu();self.kind=self.plan.get('architecture','film')
        self.net=make_network(self.kind,module).to(self.device)
        require(ref(result['state']['path'])==result['state'],'frozen proxy weights changed')
        self.net.load_state_dict(torch.load(result['state']['path'],map_location=self.device,weights_only=True));self.net.eval()
        baseline=checked(cfg['baseline']);self.layout=cfg['layout']
        self.relay=relay_from_config(cfg,self.device)
        for p in (*self.net.parameters(),*self.relay.parameters()):p.requires_grad_(False)
        self.base=torch.tensor(checked(baseline['files']['slm'],True)[::8,::8],device=self.device,dtype=torch.float32)
        assets=read(ROOT/'ASSETS_02/PLAN.json');self.bases=torch.tensor(checked(assets['bases'],True),device=self.device)
        norm=self.plan['feature_normalization']
        self.mu=torch.tensor(norm['mean'],device=self.device)[None,:,None,None]
        self.sd=torch.tensor(norm['std'],device=self.device)[None,:,None,None]
        self.max_l1=max(sum(abs(x) for x in r['coefficients_raw_gray']) for r in self.cache['rows'])
        self.source_keys={};self.base_features={}

    def gray(self,coefficients):
        require(coefficients.shape==(4,) and torch.isfinite(coefficients).all().item(),'finite four-dimensional mask required')
        require(float(coefficients.detach().abs().sum())<=self.max_l1+1e-5,'outside observed local gray envelope')
        return self.base+torch.einsum('k,khw->hw',coefficients,self.bases)

    def __call__(self,key,dmd,coefficients,quantize=True):
        import hashlib
        digest=hashlib.sha256(np.ascontiguousarray(dmd).tobytes()).hexdigest()
        require(key not in self.source_keys or self.source_keys[key]==digest,'DMD input ID reused with changed pixels')
        self.source_keys[key]=digest
        raw,field=self.relay(key,dmd,self.gray(coefficients),quantize=quantize)
        if self.kind.startswith('response'):
            if key not in self.base_features:
                with torch.no_grad():
                    br,bf=self.relay(key,dmd,self.base)
                    self.base_features[key]=residual_features(br,bf).detach().cpu()
            features=self.base_features[key].to(self.device)
        else:features=residual_features(raw,field)
        encoded=F.pad(coefficients/8,(0,12))[None]
        residual=self.net((features-self.mu)/self.sd,encoded)
        native=F.interpolate(residual,size=raw.shape,mode='bilinear',align_corners=False)[0,0]
        return (raw+native).clamp(0,255)


def signed_pairs(rows):
    """Match exact opposite coefficients for the SAME DMD input, including mixed masks."""
    lookup={};pairs=[]
    for i,r in enumerate(rows):
        c=tuple(float(x) for x in r['coefficients_raw_gray'])
        key=(r['input_id'],c)
        require(key not in lookup,'duplicate condition / coefficient identifiers')
        lookup[key]=i
    for (input_id,c),i in lookup.items():
        if not any(c):continue
        j=lookup.get((input_id,tuple(-x for x in c)))
        if j is not None and i<j:pairs.append((i,j))
    return pairs


def build_cache(dataset,physical,output):
    output.mkdir();index=read(dataset/'INDEX.json');config=read(physical/'MODEL.json')
    require(config['baseline']==index['baseline'] and config['layout']==index['layout'],'model/data contract differs')
    require(ref(ROOT/'surrogate.py')['sha256']==config['source']['sha256'],'physical model implementation changed')
    b=checked(index['baseline']);device=gpu();relay=relay_from_config(config,device)
    samples=groups(index);images=[];features=[];base_features=[];truth=[];coefficients=[];rows=[]
    base_gray=checked(read(ROOT/'ASSETS_02/PLAN.json')['assets']['BASE']['patch'],True);base_cache={}
    with torch.no_grad():
        relay.q.copy_(torch.tensor(config['q'],device=device));relay.photo_initial[:]=torch.tensor(config['photometry'],device=device)
        for s in samples:
            if s['input_id'] not in base_cache:
                br,bf=relay(s['input_id'],s['dmd'],torch.tensor(base_gray,device=device,dtype=torch.float32))
                base_cache[s['input_id']]=residual_features(br,bf)[0].cpu().numpy()
            raw,field=relay(s['input_id'],s['dmd'],torch.tensor(s['gray'],device=device,dtype=torch.float32))
            images.append(raw.cpu().numpy());features.append(residual_features(raw,field)[0].cpu().numpy());truth.append(s['image'])
            base_features.append(base_cache[s['input_id']])
            c=np.zeros(16,np.float32);c[:len(s['coefficients'])]=np.array(s['coefficients'])/8;coefficients.append(c)
            rows.append(dict(input_id=s['input_id'],mask=s['mask_name'],mask_reference=s['mask'],
                dmd_reference=s['label']['dmd'],visits=s['visits'],coefficients_raw_gray=s['coefficients'],
                input_split=s['label'].get('new_platform_split','PILOT'),mask_split=s['mask_split']))
    with (output/'CACHE.npz').open('xb') as f:
        np.savez_compressed(f,physics=np.array(images),features=np.array(features),base_features=np.array(base_features),truth=np.array(truth),coefficients=coefficients)
    write(output/'INDEX.json',dict(dataset=ref(dataset/'INDEX.json'),physical_model=ref(physical/'MODEL.json'),
        cache=ref(output/'CACHE.npz'),rows=rows,layout=index['layout'],source=ref(__file__),
        feature_scales=dict(raw_ADU=16.,model_field=4.,coordinates='[-1,1]',coefficients_raw_gray=8.),
        target_is_mean_of_independent_effective_frames=True,states_not_model_inputs=True,
        scope='development pilot cache; not for differentiable SLM optimization'))


def region_means(images,layout):
    ii=F.pad(images.double().cumsum(-2).cumsum(-1),(1,0,1,0))
    bounds=torch.tensor([r['bounds'] for r in layout['regions']],device=images.device,dtype=torch.long)
    x,y,u,v=bounds.T;area=((u-x)*(v-y)).double()
    return ((ii[...,v,u]-ii[...,y,u]-ii[...,v,x]+ii[...,y,x])/area).float()


def pilot_split(rows):
    if any(r.get('input_split') in ('TRAIN','SELECT','TEST') for r in rows):
        require(not any(r.get('input_split')=='TEST' or r.get('mask_split')=='TEST' for r in rows),
                'formal TEST cannot be opened for fitting or model selection')
        return ['SELECT_BOTH' if r.get('input_split')=='SELECT' and r.get('mask_split')=='SELECT' else
                'SELECT_INPUT' if r.get('input_split')=='SELECT' else 'SELECT_MASK' if r.get('mask_split')=='SELECT' else 'TRAIN' for r in rows]
    fit_inputs={'CENTER','X_POS12','Y_NEG12','VY_POS'};held_masks={'X4M','Y8M'}
    labels=[]
    for row in rows:
        newinput=row['input_id'] not in fit_inputs;newmask=row['mask'] in held_masks
        labels.append('SELECT_BOTH' if newinput and newmask else 'SELECT_INPUT' if newinput else
                      'SELECT_MASK' if newmask else 'TRAIN')
    return labels


def fit(cache,output,seed=105,steps=1500,normalize=False,response_mode='state',architecture='film',selection='mean'):
    output.mkdir();start=time.monotonic();device=gpu();torch.manual_seed(seed);rng=np.random.default_rng(seed)
    index=read(cache/'INDEX.json');arrays=checked(index['cache'],True);rows=index['rows'];layout=index['layout']
    split=pilot_split(rows);train=np.array([i for i,s in enumerate(split) if s=='TRAIN']);select=np.array([i for i,s in enumerate(split) if s!='TRAIN'])
    require(len(train)>=8 and len(select)>=4,'development input/mask selection partition too small')
    tensors={k:torch.tensor(arrays[k],dtype=torch.float32,device=device) for k in ('physics','features','truth','coefficients')}
    if architecture.startswith('response'):
        require('base_features' in arrays,'explicit BASE-field cache required')
        tensors['features']=torch.tensor(arrays['base_features'],dtype=torch.float32,device=device)
    feature_mu=torch.zeros(5,device=device);feature_sd=torch.ones(5,device=device)
    if normalize:
        # Single TRAIN-frozen normalization. The raw cache is unchanged; no
        # target/state/time labels or per-image contrast normalization enter.
        feature_mu[:3]=tensors['features'][train,:3].mean((0,2,3))
        feature_sd[:3]=tensors['features'][train,:3].std((0,2,3)).clamp(min=1e-5)
        tensors['features']=(tensors['features']-feature_mu[None,:,None,None])/feature_sd[None,:,None,None]
    truth=tensors['truth'];physical=tensors['physics'];ztruth=region_means(truth,layout)
    # Scales learned on TRAIN only. No per-image/per-mask normalization.
    base=[i for i in train if rows[i]['mask']=='BASE'];require(len(base)>=3,'TRAIN baseline state diversity needed')
    scales=ztruth[base].std(0).clamp(min=.01).detach()
    target_shape=truth.shape[-2:]
    if architecture=='response_cross_context':
        c=np.array([rows[i]['coefficients_raw_gray'] for i in train])/24
        cross=np.stack([c[:,i]*c[:,j] for i in range(4) for j in range(i+1,4)],1)
        require(np.linalg.matrix_rank(cross)==6,'six cross interactions need TRAIN mixed probes; axis data insufficient')
    net=make_network(architecture).to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
    lookup={(r['input_id'],r['mask']):i for i,r in enumerate(rows)};trainset=set(train.tolist())
    allpairs=signed_pairs(rows)
    pairs=[(p,m) for p,m in allpairs if p in trainset and m in trainset]
    selection_pairs=[(p,m) for p,m in allpairs if p not in trainset and m not in trainset]
    if response_mode=='relative':
        for i in train:
            if rows[i]['mask']=='BASE':continue
            base=lookup.get((rows[i]['input_id'],'BASE'))
            if base in trainset:pairs.append((int(i),base))
    # TRAIN repeated effective frames bound response denominators; state changes
    # and model residuals are not used as sensor noise samples.
    data_index=checked(index['dataset']);zf=checked(data_index['features'],True)['mean_sums']
    areas=np.array([r['pixels'] for r in layout['regions']]);noise=[]
    for i in train:
        visits=rows[i]['visits'];zz=zf[visits]/areas
        noise.extend((zz-zz.mean(0))*np.sqrt(len(zz)/(len(zz)-1)))
    noise_variance=torch.tensor(np.mean(np.square(noise),axis=0),device=device,dtype=torch.float32)
    from response import covariance
    _,w=covariance(noise);white=torch.tensor(w,device=device,dtype=torch.float32)
    static_residual=(truth[train]-physical[train]).mean(0).detach()
    write(output/'PLAN.json',dict(cache=ref(cache/'INDEX.json'),sources=archive_sources(output),
        fit_source=ref(__file__),seed=seed,steps=steps,split=split,train=train.tolist(),select=select.tolist(),
        response_pairs=pairs,state_scales_ADU=scales.cpu().tolist(),
        response_mode=response_mode,TRAIN_noise_variance_region_mean_ADU=noise_variance.cpu().tolist(),
        architecture=architecture,selection_criterion=selection,selection_response_pairs=selection_pairs,
        response_coefficient_unit_raw_gray=24 if architecture.startswith('response') else None,
        residual_cross_quadratic_terms=architecture=='response_cross_context',
        fixed_multiscale_context=(8,32,128) if architecture.endswith('context') else None,
        feature_normalization=dict(TRAIN_only=True,enabled=normalize,mean=feature_mu.cpu().tolist(),std=feature_sd.cpu().tolist()),
        loss='TRAIN-scaled region + .2 native image + paired mask response + tiny residual L2',
        SELECT_used_for_model_choice=True,formal_TEST=False,
        raw_background_is_not_error_denominator=True,no_hardware=True,online_electronic_network_added=False))
    def predict(ids):
        residual=net(tensors['features'][ids],tensors['coefficients'][ids])
        native=F.interpolate(residual,size=target_shape,mode='bilinear',align_corners=False)[:,0]
        return (physical[ids]+native).clamp(0,255),native
    def response_errors(vectors,pairlist):
        if not pairlist:return torch.zeros(1,device=device)
        p,m=torch.tensor(pairlist,device=device).T
        real=(ztruth[p]-ztruth[m])@white.T
        error=(vectors[p]-vectors[m]-ztruth[p]+ztruth[m])@white.T
        floor=torch.tensor([64*(1/len(rows[int(a)]['visits'])+1/len(rows[int(b)]['visits'])) for a,b in pairlist],device=device)
        return error.square().sum(-1)/torch.maximum(real.square().sum(-1),floor).clamp(min=1e-6)
    def score():
        with torch.no_grad():
            allpred=torch.cat([predict(list(range(i,min(i+4,len(rows)))))[0] for i in range(0,len(rows),4)])
            vectors=region_means(allpred,layout);errors=((vectors-ztruth)/scales).square().mean(-1)
            response_score=float(response_errors(vectors,selection_pairs).mean())
            value=float(errors[select].mean())+(.5*response_score if selection=='mean_response' else 0.)
        return value,allpred,errors,response_score
    best=float('inf');history=[];best_state=None;best_step=None
    for step in range(steps):
        ids=rng.choice(train,size=min(4,len(train)),replace=False).tolist()
        optim.zero_grad(set_to_none=True);pred,resid=predict(ids);z=region_means(pred,layout)
        region=((z-ztruth[ids])/scales).square().mean()
        pixel=((1+4*(truth[ids]>11).float())*(pred-truth[ids]).square()).mean()
        response=torch.zeros((),device=device)
        if pairs:
            pair=pairs[int(rng.integers(len(pairs)))];p,_=predict(list(pair));zp=region_means(p,layout)
            real_delta=ztruth[pair[0]]-ztruth[pair[1]]
            response=(((zp[0]-zp[1]-real_delta)/scales).square()).mean()
            if response_mode=='relative':
                real=white@real_delta;err=white@(zp[0]-zp[1]-real_delta)
                floor=64*(1/len(rows[pair[0]]['visits'])+1/len(rows[pair[1]]['visits']))
                response=.5*err.square().sum()/real.square().sum().clamp(min=floor)
        loss=region+.2*pixel+response+1e-5*resid.square().mean()
        require(torch.isfinite(loss).item(),'nonfinite residual fit');loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5);optim.step()
        if step%100==0 or step==steps-1:
            value,_,error,response_score=score();item=dict(step=step,loss=float(loss.detach()),SELECT_criterion=value,
                SELECT_scaled_MSE=float(error[select].mean()),SELECT_relative_response_MSE=response_score,
                TRAIN_scaled_MSE=float(error[train].mean()),seconds=time.monotonic()-start)
            history.append(item);print('residual',seed,step,'train',round(item['TRAIN_scaled_MSE'],5),'select',round(value,5),flush=True)
            if value<best:
                best=value;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(best_state);_,pred,errors,_=score();pred=pred.cpu().numpy()
    with (output/'STATE.pt').open('xb') as f:torch.save(best_state,f)
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,predicted=pred,physics=arrays['physics'],truth=arrays['truth'],
        static_residual=static_residual.cpu().numpy())
    write(output/'HISTORY.json',history)
    metrics=[]
    for label in dict.fromkeys(split):
        ids=[i for i,s in enumerate(split) if s==label]
        item=dict(split=label,conditions=len(ids))
        for name,est in [('physics',arrays['physics']),('static_TRAIN_residual',arrays['physics']+static_residual.cpu().numpy()),('hybrid',pred)]:
            e=est[ids]-arrays['truth'][ids];area=np.array([r['pixels'] for r in layout['regions']]);ze=np.stack([integrate(a,layout)/area for a in e])
            item[name]=dict(region_RMSE_ADU=float(np.sqrt(np.mean(ze**2))),
                TRAIN_scaled_region_RMSE=float(np.sqrt(np.mean((ze/scales.cpu().numpy())**2))),
                pixel_RMSE_ADU=float(np.sqrt(np.mean(e**2))))
        metrics.append(item)
    write(output/'RESULT.json',dict(seed=seed,best_step=best_step,metrics=metrics,seconds=time.monotonic()-start,
        state=ref(output/'STATE.pt'),predictions=ref(output/'PREDICTIONS.npz'),model_parameters=sum(p.numel() for p in net.parameters()),
        deployed=False,SLM_optimized=False,formal_TEST=False))
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(metrics))
    for j,(name,color) in enumerate([('physics','#D55E00'),('static_TRAIN_residual','#999999'),('hybrid','#0072B2')]):
        ax.bar(x+(j-1)*.25,[m[name]['TRAIN_scaled_region_RMSE'] for m in metrics],.25,label=name,color=color)
    ax.set_xticks(x,[m['split'] for m in metrics]);ax.set_ylabel('64-region RMSE / TRAIN state scale');ax.legend()
    ax.set_title('Development selection, not independent TEST');fig.tight_layout();save_figure(fig,output/'01_model_comparison',180);plt.close(fig)


def evaluate(model,cache,output,mode='test',freeze=None):
    """Frozen evaluation only. No optimizer, no fitting, no threshold adjustment."""
    output.mkdir();plan=read(model/'PLAN.json');trained=read(model/'RESULT.json')
    train_cache=checked(plan['cache']);test_cache=read(cache/'INDEX.json')
    require(train_cache['physical_model']==test_cache['physical_model'],'TEST physical model differs from frozen TRAIN')
    require(train_cache['layout']==test_cache['layout'],'TEST layout changed')
    rows=test_cache['rows'];layout=test_cache['layout']
    require(mode in ('test','time'),'explicit frozen evaluation mode required')
    if mode=='test':
        require(all(r.get('input_split')=='TEST' or r.get('mask_split')=='TEST' for r in rows),'TEST-only evaluation expected')
    else:
        require(freeze is not None,'pre-measurement model freeze required')
        marker=read(freeze)
        require(marker['model']==ref(model/'RESULT.json'),'time evaluation model differs from pre-capture freeze')
        temporal_data=checked(test_cache['dataset'])
        require(all(r['host_times_ns'][0]>marker['created_unix_ns'] for r in temporal_data['rows']),
                'time holdout contains pre-freeze frames')
        known={(r['input_id'],r['mask_reference']['sha256']) for r in train_cache['rows']}
        require(all((r['input_id'],r['mask_reference']['sha256']) in known for r in rows),'time revisit identity differs')
    frozen_source=model/'SOURCE/residual_fit.py'
    require(ref(frozen_source)['sha256']==plan['fit_source']['sha256'],'frozen fit implementation missing')
    frozen=load(frozen_source,'e105_frozen_residual_'+str(plan['seed']))
    device=gpu();net=make_network(plan['architecture'],frozen).to(device)
    require(ref(trained['state']['path'])==trained['state'],'frozen checkpoint changed')
    net.load_state_dict(torch.load(trained['state']['path'],map_location=device,weights_only=True));net.eval()
    for p in net.parameters():p.requires_grad_(False)
    arrays=checked(test_cache['cache'],True);physical=arrays['physics'];truth=arrays['truth']
    features=arrays['base_features' if plan['architecture'].startswith('response') else 'features']
    norm=plan['feature_normalization'];mu=np.array(norm['mean'],np.float32);sd=np.array(norm['std'],np.float32)
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(rows),4):
            v=torch.tensor((features[start:start+4]-mu[None,:,None,None])/sd[None,:,None,None],device=device)
            c=torch.tensor(arrays['coefficients'][start:start+4],device=device)
            residual=F.interpolate(net(v,c),size=truth.shape[-2:],mode='bilinear',align_corners=False)[:,0].cpu().numpy()
            predictions.extend(np.clip(physical[start:start+4]+residual,0,255))
    pred=np.asarray(predictions);areas=np.array([r['pixels'] for r in layout['regions']]);scales=np.array(plan['state_scales_ADU'])
    z={name:np.stack([integrate(im,layout)/areas for im in a]) for name,a in
       [('truth',truth),('physics',physical),('hybrid',pred)]}
    roles=(['TIME_HOLDOUT']*len(rows) if mode=='time' else
           ['TEST_BOTH' if r.get('input_split')=='TEST' and r.get('mask_split')=='TEST' else
            'TEST_INPUT' if r.get('input_split')=='TEST' else 'TEST_MASK' for r in rows])
    metrics=[]
    for role in dict.fromkeys(roles):
        ids=[i for i,s in enumerate(roles) if s==role];item=dict(role=role,conditions=len(ids))
        for name in ('physics','hybrid'):
            e=z[name][ids]-z['truth'][ids]
            item[name]=dict(TRAIN_scaled_RMSE=float(np.sqrt(np.mean((e/scales)**2))),region_mean_RMSE_ADU=float(np.sqrt(np.mean(e**2))),
                per_region_RMSE_ADU=np.sqrt(np.mean(e**2,axis=0)).tolist(),
                per_region_bias_ADU=e.mean(0).tolist(),
                worst_region_TRAIN_scaled_RMSE=float(np.sqrt(np.mean((e/scales)**2,axis=0)).max()))
        item['improvement_fraction']=1-item['hybrid']['TRAIN_scaled_RMSE']/max(item['physics']['TRAIN_scaled_RMSE'],1e-12)
        metrics.append(item)
    # Covariance strictly from the already frozen TRAIN repeated-frame residuals.
    from response import covariance
    data_index=checked(train_cache['dataset']);zz=checked(data_index['features'],True)['mean_sums']/areas;noise=[]
    for i in plan['train']:
        visits=train_cache['rows'][i]['visits'];v=zz[visits]
        noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    cov,white=covariance(noise);pairs=[]
    test_index=checked(test_cache['dataset']);test_z=checked(test_index['features'],True)['mean_sums']/areas
    test_noise=[]
    for row in rows:
        v=test_z[row['visits']];test_noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    tcov,_=covariance(test_noise)
    noise_diagnostic=dict(TRAIN_rms_region_ADU=float(np.sqrt(np.trace(cov)/64)),
        TEST_rms_region_ADU=float(np.sqrt(np.trace(tcov)/64)),
        TEST_over_TRAIN_RMS_ratio=float(np.sqrt(np.trace(tcov)/np.trace(cov))),
        TEST_covariance_NOT_used_for_fit_or_primary_whitening=True,
        TRAIN_per_region_SD_ADU=np.sqrt(np.diag(cov)).tolist(),
        TEST_per_region_SD_ADU=np.sqrt(np.diag(tcov)).tolist())
    for p,m in signed_pairs(rows):
        real=z['truth'][p]-z['truth'][m];wa=white@real
        snr=np.linalg.norm(real)/np.sqrt(np.trace(cov)*(1/len(rows[p]['visits'])+1/len(rows[m]['visits'])))
        item=dict(input_id=rows[p]['input_id'],mask_pair=[rows[p]['mask'],rows[m]['mask']],role=roles[p],
            SNR=float(snr),identifiable=bool(snr>=3),
            SNR_TEST_noise_diagnostic=float(np.linalg.norm(real)/np.sqrt(np.trace(tcov)*(1/len(rows[p]['visits'])+1/len(rows[m]['visits'])))))
        require(roles[p]==roles[m],'opposite masks must share TEST partition')
        for name in ('physics','hybrid'):
            wb=white@(z[name][p]-z[name][m]);norm=max(np.linalg.norm(wa),1e-12)
            item[name]=dict(cosine=float(wa@wb/max(norm*np.linalg.norm(wb),1e-12)),
                amplitude_ratio=float(np.linalg.norm(wb)/norm),relative_error=float(np.linalg.norm(wb-wa)/norm))
        pairs.append(item)
    summaries=[]
    for role in ('ALL',*dict.fromkeys(roles)):
        pp=[p for p in pairs if role=='ALL' or p['role']==role];good=[p for p in pp if p['identifiable']]
        coverage=len(good)/len(pp) if pp else 0.
        s=dict(role=role,total_pairs=len(pp),identifiable_pairs=len(good),identifiable_fraction=coverage)
        for name in ('physics','hybrid'):
            s[name]=None if not good else dict(median_cosine=float(np.median([p[name]['cosine'] for p in good])),
                direction_fraction=float(np.mean([p[name]['cosine']>0 for p in good])),
                amplitude_fraction_05_20=float(np.mean([.5<=p[name]['amplitude_ratio']<=2 for p in good])),
                median_relative_error=float(np.median([p[name]['relative_error'] for p in good])))
        v=s['hybrid'];s['response_gate_passed']=bool(v and coverage>=.8 and v['median_cosine']>=.7 and
            v['direction_fraction']>=.8 and v['amplitude_fraction_05_20']>=.8 and v['median_relative_error']<=.75)
        summaries.append(s)
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,predicted=pred,physics=physical,truth=truth)
    write(output/'RESULT.json',dict(model=ref(model/'RESULT.json'),cache=ref(cache/'INDEX.json'),source=ref(__file__),
        frozen_fit_source=ref(frozen_source),metrics=metrics,response_pairs=pairs,response_summary=summaries,
        noise_diagnostic=noise_diagnostic,
        model_changed=False,thresholds_changed=False,formal_TEST=mode=='test',
        evaluation_mode=mode,pre_capture_freeze=None if freeze is None else ref(freeze),
        static_model_gate_passed=bool(all(x['improvement_fraction']>=.2 for x in metrics) and
                                     all(x['response_gate_passed'] for x in summaries)),
        time_holdout_NOT_evaluated=mode!='time',SLM_training_allowed=False,control_qualified=False))
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,ax=plt.subplots(figsize=(8,4));xx=np.arange(len(metrics))
    for j,(name,color) in enumerate((('physics','#D55E00'),('hybrid','#0072B2'))):
        ax.bar(xx+(j-.5)*.3,[r[name]['TRAIN_scaled_RMSE'] for r in metrics],.3,label=name,color=color)
    ax.set_xticks(xx,[r['role'] for r in metrics]);ax.set_ylabel('64-region RMSE / frozen TRAIN scale');ax.legend()
    fig.tight_layout();save_figure(fig,output/'01_frozen_TEST',180);plt.close(fig)
    print('FROZEN',mode.upper(),'mean improvements',[(m['role'],m['improvement_fraction']) for m in metrics],summaries,flush=True)


def forecast(model,design,output):
    """Seal native predictions for explicit TEST inputs/masks BEFORE measurement."""
    output.mkdir();plan=read(design/'PLAN.json');proxy=FrozenProxy(model);rows=[];means=[];physics=[]
    seen=set()
    with torch.no_grad():
        for job in plan['jobs']:
            if job['role'] not in ('TEST_INPUT','TEST_MASK','TEST_BOTH'):continue
            mask=plan['assets'][job['mask']];coeff=torch.tensor(mask['coefficients'],device=proxy.device,dtype=torch.float32)
            patch=checked(mask['patch'],True)
            exported=proxy.gray(coeff).round().clamp(0,255).byte().repeat_interleave(8,0).repeat_interleave(8,1).cpu().numpy()
            require(np.array_equal(exported,patch),'forecast physical SLM bytes differ from planned file')
            for label in checked(plan['catalogs'][job['catalog']])['inputs']:
                key=(label['input_id'],mask['patch']['sha256']);require(key not in seen,'duplicate forecast condition');seen.add(key)
                dmd=checked(label['dmd'],True)
                pred=proxy(label['input_id'],dmd,coeff)
                raw,_=proxy.relay(label['input_id'],dmd,proxy.gray(coeff))
                means.append(pred.cpu().numpy());physics.append(raw.cpu().numpy())
                rows.append(dict(input_id=label['input_id'],mask=job['mask'],mask_reference=mask['patch'],
                    dmd_reference=label['dmd'],role=job['role'],coefficients_raw_gray=mask['coefficients']))
    require(len(rows)==80,'explicit frozen TEST budget is 80 conditions')
    path=output/'PREDICTIONS.npz'
    with path.open('xb') as f:np.savez_compressed(f,predicted=np.asarray(means),physics=np.asarray(physics))
    write(output/'FORECAST.json',dict(model=ref(model/'RESULT.json'),plan=ref(model/'PLAN.json'),
        design=ref(design/'PLAN.json'),source=ref(__file__),created_unix_ns=time.time_ns(),
        physical_model=proxy.cache['physical_model'],layout=proxy.layout,rows=rows,predictions=ref(path),
        actual_TEST_CCD_NOT_read=True,model_changed=False,source_mask_bytes_verified=True))
    print('PRE-CAPTURE FORECAST SEALED',len(rows),'native CCD predictions',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-cache',action='store_true');p.add_argument('--dataset')
    p.add_argument('--physical',default='PHYSICAL_01');p.add_argument('--cache');p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=105);p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--normalize-train',action='store_true');p.add_argument('--response-mode',choices=('state','relative'),default='state')
    p.add_argument('--architecture',choices=ARCHITECTURES,default='film')
    p.add_argument('--selection',choices=('mean','mean_response'),default='mean');p.add_argument('--evaluate')
    p.add_argument('--forecast-design');p.add_argument('--model')
    p.add_argument('--evaluation-mode',choices=('test','time'),default='test');p.add_argument('--freeze-file');a=p.parse_args()
    require(Path(a.output).name==a.output,'fresh explicit output required')
    if a.forecast_design:
        require(a.model is not None,'explicit frozen model required')
        forecast(ROOT/a.model,ROOT/a.forecast_design,ROOT/a.output)
    elif a.evaluate:evaluate(ROOT/a.evaluate,ROOT/a.cache,ROOT/a.output,a.evaluation_mode,
                            None if a.freeze_file is None else Path(a.freeze_file))
    elif a.build_cache:build_cache(ROOT/a.dataset,ROOT/a.physical,ROOT/a.output)
    else:fit(ROOT/a.cache,ROOT/a.output,a.seed,a.steps,a.normalize_train,a.response_mode,a.architecture,a.selection)
