"""Explicit offline GPU check against the pinned existing CPU ASM."""
import argparse
import sys
import time
import subprocess
from pathlib import Path
import numpy as np
from workflow import ROOT,LAB,read,ref,write,load,require


class FrozenFocusProxy:
    """Numerical adapter for the archived convolution candidate; offline only."""
    def __init__(self,model,device=None):
        import torch
        from dataset import checked
        from residual_fit import relay_from_config
        from fit import gpu
        self.model=Path(model);self.plan=read(self.model/'PLAN.json');result=read(self.model/'RESULT.json')
        require(self.plan['architecture']=='focus_convolution','explicit convolution model required')
        self.cache=checked(self.plan['cache']);cfg=checked(self.cache['physical_model']);self.layout=cfg['layout']
        self.device=device or gpu();src=self.model/'SOURCE/convolution_residual.py'
        require(ref(src)['sha256']==self.plan['fit_source']['sha256'],'archived convolution source changed')
        module=load(src,'e105_focus_proxy_'+self.model.name)
        size=self.plan['kernel_size'];ps=self.plan.get('kernel_parameter_size')
        self.net=(module.FocusConvolution(size) if ps is None else module.FocusConvolution(size,ps)).to(self.device)
        require(ref(result['state']['path'])==result['state'],'archived convolution state changed')
        self.net.load_state_dict(torch.load(result['state']['path'],map_location=self.device,weights_only=True));self.net.eval()
        self.relay=relay_from_config(cfg,self.device);self.template_relay=relay_from_config(cfg,self.device)
        with torch.no_grad():self.template_relay.q[8:11]=0
        for obj in (self.net,self.relay,self.template_relay):
            for p in obj.parameters():p.requires_grad_(False)
        baseline=checked(cfg['baseline']);assets=read(ROOT/'ASSETS_02/PLAN.json')
        self.base=torch.tensor(checked(baseline['files']['slm'],True)[::8,::8],device=self.device,dtype=torch.float32)
        self.bases=torch.tensor(checked(assets['bases'],True),device=self.device)
        self.max_l1=max(sum(abs(x) for x in r['coefficients_raw_gray']) for r in self.cache['rows'])
        self.spectra={};self.source_keys={}

    def gray(self,coefficients):
        from residual_fit import FrozenProxy
        return FrozenProxy.gray(self,coefficients)

    def __call__(self,key,dmd,coefficients,quantize=True):
        import torch,hashlib
        digest=hashlib.sha256(np.ascontiguousarray(dmd).tobytes()).hexdigest()
        require(key not in self.source_keys or self.source_keys[key]==digest,'DMD input ID aliases different pixels')
        self.source_keys[key]=digest
        raw,_=self.relay(key,dmd,self.gray(coefficients),quantize=quantize)
        if key not in self.spectra:
            with torch.no_grad():
                t,_=self.template_relay(key,dmd,torch.zeros_like(self.base))
                bg=self.template_relay.photo_initial[1]+3*torch.tanh(self.template_relay.q[12])
                self.spectra[key]=torch.fft.rfft2((t-bg).clamp(min=0)/self.plan['focus_scale'],s=self.plan['fft_shape']).detach().cpu()
        residual=self.net(self.spectra[key].to(self.device)[None],coefficients[None]/24,
            raw.shape,self.plan['fft_shape'])[0]
        return (raw+residual).clamp(0,255)


def frozen_proxy(model,device=None):
    from residual_fit import FrozenProxy
    return (FrozenFocusProxy if read(model/'PLAN.json')['architecture']=='focus_convolution' else FrozenProxy)(model,device)


def cpu_reference(path):
    baseline=read(ROOT/'BASELINE_01/BASELINE.json')
    source=LAB.parent/'ballplate_fourframe96_deploy/live_manual_v17/diffraction_model.py'
    require(ref(source)['sha256']=='7c69033621ed3f0709c84e5698daf8f80e4b2cd85c6687e2d4bd0db7bf481edb','reference changed')
    spec=__import__('importlib.util').util.spec_from_file_location('e105_cpu_asm',source)
    cpu=__import__('importlib.util').util.module_from_spec(spec);sys.modules[spec.name]=cpu;spec.loader.exec_module(cpu)
    gray=np.load(baseline['files']['slm']['path'],allow_pickle=False)
    dmd=np.load(read(ROOT/'ASSETS_02/SMOKE_CATALOG.json')['inputs'][0]['dmd']['path'],allow_pickle=False)
    settings=baseline['expected']['slm'];panel=np.zeros((1080,1920),np.uint8)
    x,y=660-settings['x'],240+settings['y'];panel[y:y+600,x:x+600]=gray
    reference=cpu.CachedASM().simulate(dmd,panel,cpu.OpticalReference(**baseline['expected']['optical_reference']))
    with path.open('xb') as f:np.save(f,reference,allow_pickle=False)


def run(output):
    import torch
    import torch.nn.functional as F
    from surrogate import ASMRelay,oriented
    baseline=read(ROOT/'BASELINE_01/BASELINE.json');geometry=read(baseline['geometry']['path'])
    index=read(ROOT/'SMOKE_AUDIT_02/INDEX.json');layout=index['layout']
    source=LAB.parent/'ballplate_fourframe96_deploy/live_manual_v17/diffraction_model.py'
    require(ref(source)['sha256']=='7c69033621ed3f0709c84e5698daf8f80e4b2cd85c6687e2d4bd0db7bf481edb', 'reference ASM changed')
    devices=[i for i in range(torch.cuda.device_count()) if '3070' in torch.cuda.get_device_name(i)]
    require(len(devices)==1,'explicit RTX3070 required');device='cuda:'+str(devices[0])
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.65,devices[0])
    output.mkdir();start=time.monotonic()
    relay=ASMRelay(baseline,layout,geometry,device)
    gray=np.load(baseline['files']['slm']['path'],allow_pickle=False)
    dmd=np.load(read(ROOT/'ASSETS_02/SMOKE_CATALOG.json')['inputs'][0]['dmd']['path'],allow_pickle=False)
    source_file=output/'CPU_REFERENCE.npy'
    subprocess.run([str(LAB.parent/'ballplate_fourframe96_deploy/.calibration-venv/bin/python'),'-B',__file__,
        '--cpu-reference',str(source_file)],check=True)
    reference=np.load(source_file,allow_pickle=False)
    with torch.no_grad():
        relay.q[7]=-30;relay.photo_initial[:]=torch.tensor([1.,0.],device=device)
        expected=F.grid_sample(oriented(torch.tensor(reference,device=device),geometry['orientation'])[None,None],
            relay.camera_grid(),align_corners=False)[0,0].clamp(0,255).cpu().numpy()
        actual,_=relay('CENTER',dmd,torch.tensor(gray,device=device,dtype=torch.float32))
        actual=actual.cpu().numpy()
    delta=actual-expected;relative=float(np.linalg.norm(delta)/max(np.linalg.norm(expected),1e-12))
    passed=relative<2e-4
    write(output/'RESULT.json',dict(passed=passed,reference=ref(source),source=ref(__file__),
        model_source=ref(ROOT/'surrogate.py'),device=torch.cuda.get_device_name(devices[0]),torch=torch.__version__,
        shape=relay.shape,native_shape=relay.native_shape,relative_native_L2_error=relative,
        max_abs_ADU=float(np.abs(delta).max()),rms_ADU=float(np.sqrt(np.mean(delta**2))),
        physical_parameters=relay.q.numel(),elapsed_s=time.monotonic()-start,hardware_actions=0,
        comparison_scope='Numerical reproduction of old ASM conventions under NEW geometry; not physical validity',
        both_models_same_bilinear_camera_sampling=True,blur_disabled_for_identity_test=True))
    require(passed, 'numerical physics identity test failed')
    print('GPU ASM reproduction PASS',relative,flush=True)


def proxy_check(model,output):
    import torch
    from dataset import checked
    from residual_fit import FrozenProxy,region_means
    output.mkdir();proxy=frozen_proxy(model);plan=read(model/'PLAN.json');result=read(model/'RESULT.json')
    cache=checked(plan['cache']);pred=checked(result['predictions'],True)['predicted'];rows=cache['rows']
    # TRAIN-only numerical QA; this does not consume TEST labels or optimize SLM.
    ids=[i for i in plan['train'] if rows[i]['input_id'] in ('CENTER','VY_POS')][:4]
    require(len(ids)==4,'four named TRAIN numerical probes required');checks=[]
    with torch.no_grad():
        for i in ids:
            r=rows[i];c=torch.tensor(r['coefficients_raw_gray'],device=proxy.device,dtype=torch.float32)
            out=proxy(r['input_id'],checked(r['dmd_reference'],True),c).cpu().numpy();e=out-pred[i]
            checks.append(dict(input_id=r['input_id'],mask=r['mask'],native_max_abs_error_ADU=float(abs(e).max()),
                native_RMS_error_ADU=float(np.sqrt(np.mean(e**2)))))
    row=rows[ids[0]];dmd=checked(row['dmd_reference'],True)
    c=torch.tensor([4.,-2.,1.,0.],device=proxy.device,requires_grad=True)
    weight=torch.tensor(np.random.default_rng(1058).normal(size=64),device=proxy.device,dtype=torch.float32)
    def objective(cc,quantize):
        return (region_means(proxy(row['input_id'],dmd,cc,quantize=quantize),proxy.layout)*weight).sum()
    exact=torch.autograd.grad(objective(c,False),c)[0].detach().cpu().numpy()
    differences={}
    with torch.no_grad():
        for label,eps,quantize in [('continuous',.1,False),('discrete_step2',2.,True),('discrete_step4',4.,True)]:
            values=[]
            for k in range(4):
                step=torch.zeros_like(c);step[k]=eps
                values.append(float((objective(c+step,quantize)-objective(c-step,quantize))/(2*eps)))
            v=np.array(values);relative=float(np.linalg.norm(v-exact)/max(np.linalg.norm(exact),1e-12))
            cosine=float(v@exact/max(np.linalg.norm(v)*np.linalg.norm(exact),1e-12))
            differences[label]=dict(gradient=values,relative_L2_error=relative,cosine=cosine,step_gray=eps)
    passed=max(r['native_max_abs_error_ADU'] for r in checks)<1e-4 and differences['continuous']['relative_L2_error']<.1
    write(output/'RESULT.json',dict(model=ref(model/'RESULT.json'),source=ref(__file__),proxy_source=ref(ROOT/'residual_fit.py'),
        forward_checks=checks,autograd_continuous=exact.tolist(),finite_differences=differences,numerical_passed=passed,
        real_optical_gradient_NOT_measured=True,hardware_actions=0,SLM_optimized=False,
        quantized_forward_is_exact_gray_round_clip=True,discrete_derivative_is_only_an_approximation=True))
    require(passed,'frozen differentiable proxy numerical QA failed')
    print('FROZEN PROXY QA',checks,differences,flush=True)


def grid_check(model,output):
    """Numerical padding convergence only; physical parameters stay frozen."""
    import gc
    import torch
    from dataset import checked
    from surrogate import ASMRelay
    from fit import gpu
    output.mkdir();cfg=read(model/'MODEL.json');baseline=checked(cfg['baseline']);device=gpu()
    assets=read(ROOT/'ASSETS_STRONG_01/PLAN.json');catalog=checked(assets['catalogs']['PILOT'])
    inputs=[r for r in catalog['inputs'] if r['input_id'] in ('CENTER','X_POS12','Y_POS12','VY_POS')]
    require(len(inputs)==4,'four explicit TRAIN diagnostic inputs required')
    pads=(3,4,5);rows=[(r,name) for r in inputs for name in assets['assets']]
    write(output/'PLAN.json',dict(model=ref(model/'MODEL.json'),source=ref(__file__),model_source=ref(ROOT/'surrogate.py'),
        pads=list(pads),inputs=[r['input_id'] for r in inputs],mask_assets=ref(ROOT/'ASSETS_STRONG_01/PLAN.json'),
        CCD_grid_fixed=True,SLM_physical_pixels_fixed=True,physical_parameters_fixed=True,
        screening_relative_mean_and_response_L2_max=.05,screening_response_cosine_min=.99,
        physical_mapping_NOT_verified=True,hardware_actions=0))
    values={};background=cfg['photometry'][1]+3*np.tanh(cfg['q'][12])
    for padding in pads:
        relay=ASMRelay(baseline,cfg['layout'],checked(cfg['geometry']),device,padding=padding,
            phase_sign=cfg.get('phase_sign',1),slm_shift_bound=cfg.get('slm_shift_bound',32))
        vectors=[]
        with torch.no_grad():
            relay.q.copy_(torch.tensor(cfg['q'],device=device));relay.photo_initial[:]=torch.tensor(cfg['photometry'],device=device)
            for row,name in rows:
                dmd=checked(row['dmd'],True);gray=torch.tensor(checked(assets['assets'][name]['patch'],True),device=device,dtype=torch.float32)
                raw,_=relay(row['input_id'],dmd,gray)
                vectors.append((relay.integrate(raw)/relay.areas).cpu().numpy())
        values[padding]=np.asarray(vectors);print('NUMERICAL GRID',padding,relay.shape,len(rows),'conditions',flush=True)
        del relay,raw;gc.collect();torch.cuda.empty_cache()
    comparisons=[];lookup={(r['input_id'],name):i for i,(r,name) in enumerate(rows)}
    for a,b in ((3,4),(4,5),(3,5)):
        means=[];responses=[]
        for i,(row,name) in enumerate(rows):
            x=values[a][i]-background;y=values[b][i]-background
            means.append(dict(input_id=row['input_id'],mask=name,
                net_region_relative_L2=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12)),
                region_RMS_ADU=float(np.sqrt(np.mean((x-y)**2)))))
        for row in inputs:
            for axis in ('X4','Y4','X8','Y8'):
                p=lookup[(row['input_id'],axis+'P')];m=lookup[(row['input_id'],axis+'M')]
                x=values[a][p]-values[a][m];y=values[b][p]-values[b][m]
                responses.append(dict(input_id=row['input_id'],axis=axis,
                    relative_L2=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12)),
                    cosine=float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-12))))
        maximum=max(r['net_region_relative_L2'] for r in means)
        response_max=max(r['relative_L2'] for r in responses);cosine_min=min(r['cosine'] for r in responses)
        comparisons.append(dict(padding=[a,b],mean=means,response=responses,max_mean_relative_L2=maximum,
            max_response_relative_L2=response_max,min_response_cosine=cosine_min,
            numerical_screen_passed=bool(maximum<=.05 and response_max<=.05 and cosine_min>=.99)))
    with (output/'READOUTS.npz').open('xb') as f:np.savez_compressed(f,**{'padding%d'%k:v for k,v in values.items()})
    write(output/'RESULT.json',dict(comparisons=comparisons,all_passed=all(r['numerical_screen_passed'] for r in comparisons),
        raw_background_NOT_error_denominator=True,no_physical_or_control_qualification=True))
    print('GRID QA',[(r['padding'],r['max_mean_relative_L2'],r['max_response_relative_L2'],r['min_response_cosine'],r['numerical_screen_passed']) for r in comparisons],flush=True)


def response_geometry_check(model,output):
    """Diagnostic only: compare whitened and raw signed responses, no new gate."""
    from dataset import checked,integrate
    from residual_fit import signed_pairs
    from response import covariance
    output.mkdir();plan=read(model/'PLAN.json');result=read(model/'RESULT.json')
    cache=checked(plan['cache']);data=checked(cache['dataset']);rows=cache['rows'];layout=cache['layout']
    area=np.array([r['pixels'] for r in layout['regions']]);raw=checked(data['features'],True)['mean_sums']/area
    pred=checked(result['predictions'],True);noise=[];train=set(plan['train'])
    for i in train:
        v=raw[rows[i]['visits']];noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    cov,white=covariance(noise);values={k:np.stack([integrate(a,layout)/area for a in pred[k]]) for k in ('truth','physics','predicted')}
    comparisons=[]
    for p,m in signed_pairs(rows):
        real=values['truth'][p]-values['truth'][m];est=values['predicted'][p]-values['predicted'][m]
        scope='TRAIN' if p in train and m in train else 'SELECT'
        item=dict(scope=scope,input_id=rows[p]['input_id'],masks=[rows[p]['mask'],rows[m]['mask']])
        for name,a,b in [('raw',real,est),('TRAIN_whitened',white@real,white@est)]:
            item[name]=dict(cosine=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12)),
                relative_L2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12)))
        comparisons.append(item)
    summary=[]
    for scope in ('TRAIN','SELECT'):
        rr=[r for r in comparisons if r['scope']==scope]
        summary.append(dict(scope=scope,pairs=len(rr),**{k:dict(median_cosine=float(np.median([r[k]['cosine'] for r in rr])),
            median_relative_L2=float(np.median([r[k]['relative_L2'] for r in rr]))) for k in ('raw','TRAIN_whitened')}))
    write(output/'RESULT.json',dict(model=ref(model/'RESULT.json'),source=ref(__file__),comparisons=comparisons,summary=summary,
        diagnostic_only=True,no_threshold_or_control_change=True,formal_TEST=False))
    print('RESPONSE METRIC DIAGNOSTIC',summary,flush=True)


def compare_physical_caches(caches,output):
    """Fixed same-data physical candidates; no additional fitting or TEST."""
    from dataset import checked,integrate
    from residual_fit import pilot_split,signed_pairs
    from response import covariance
    output.mkdir();reference=read(caches[0]/'INDEX.json');layout=reference['layout'];rows=reference['rows']
    labels=pilot_split(rows);train=[i for i,s in enumerate(labels) if s=='TRAIN'];area=np.array([r['pixels'] for r in layout['regions']])
    data=checked(reference['dataset']);zf=checked(data['features'],True)['mean_sums']/area;noise=[]
    for i in train:
        v=zf[rows[i]['visits']];noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    cov,white=covariance(noise);summaries=[];details=[]
    for cache in caches:
        idx=read(cache/'INDEX.json');require(idx['dataset']==reference['dataset'] and idx['rows']==rows,'same optical conditions required')
        a=checked(idx['cache'],True);z={k:np.stack([integrate(im,layout)/area for im in a[k]]) for k in ('physics','truth')}
        base=[i for i in train if rows[i]['mask']=='BASE'];scales=z['truth'][base].std(0,ddof=1).clip(min=.01)
        pairs=[]
        for p,m in signed_pairs(rows):
            real=z['truth'][p]-z['truth'][m];aa=white@real;bb=white@(z['physics'][p]-z['physics'][m])
            snr=float(np.linalg.norm(real)/np.sqrt(np.trace(cov)*(1/len(rows[p]['visits'])+1/len(rows[m]['visits']))))
            pairs.append(dict(input_id=rows[p]['input_id'],masks=[rows[p]['mask'],rows[m]['mask']],split=labels[p],SNR=snr,
                cosine=float(aa@bb/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-12)),
                relative_error=float(np.linalg.norm(aa-bb)/max(np.linalg.norm(aa),1e-12))))
        for role in dict.fromkeys(labels):
            ii=[i for i,s in enumerate(labels) if s==role];err=z['physics'][ii]-z['truth'][ii];rr=[r for r in pairs if r['split']==role and r['SNR']>=3]
            summaries.append(dict(cache=cache.name,split=role,conditions=len(ii),
                region_RMS_ADU=float(np.sqrt(np.mean(err**2))),TRAIN_scaled_region_RMSE=float(np.sqrt(np.mean((err/scales)**2))),
                identifiable_response_pairs=len(rr),median_white_cosine=float(np.median([r['cosine'] for r in rr])) if rr else None,
                median_relative_response_error=float(np.median([r['relative_error'] for r in rr])) if rr else None))
        details.append(dict(cache=ref(cache/'INDEX.json'),physical_model=idx['physical_model'],pairs=pairs))
    write(output/'RESULT.json',dict(source=ref(__file__),summary=summaries,details=details,formal_TEST=False,
        no_new_fitting=True,no_hardware=True,SLM_training_allowed=False))
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(10,4));roles=list(dict.fromkeys(labels));width=.7/len(caches)
    for k,cache in enumerate(caches):
        rr=[next(r for r in summaries if r['cache']==cache.name and r['split']==s) for s in roles]
        for ax,key in zip(axes,('TRAIN_scaled_region_RMSE','median_white_cosine')):
            ax.bar(np.arange(len(roles))+(k-(len(caches)-1)/2)*width,[r[key] for r in rr],width,label=cache.name)
            ax.set_xticks(np.arange(len(roles)),roles)
    axes[0].set_ylabel('64-region RMSE / TRAIN state scale');axes[1].set_ylabel('Response whitened cosine');axes[1].axhline(.7,color='#999999',ls='--')
    axes[0].legend(fontsize=7);fig.tight_layout();save_figure(fig,output/'01_physical_comparison',180);plt.close(fig)
    print('PHYSICAL CANDIDATES',summaries,flush=True)


def coordinate_screen(model,output):
    """TRAIN-only discrete SLM-coordinate hypotheses; no hardware transform."""
    import torch
    import torch.nn.functional as F
    from dataset import checked,integrate
    from surrogate import ASMRelay,oriented,propagate
    from fit import gpu,groups
    from response import covariance
    output.mkdir();cfg=read(model/'MODEL.json');b=checked(cfg['baseline']);device=gpu()
    require(cfg['source']['sha256']==ref(ROOT/'surrogate.py')['sha256'],'ASM implementation changed')
    idx=read(ROOT/'JOINT_TRAIN_AUDIT_01/INDEX.json');samples=groups(idx);layout=cfg['layout'];area=np.array([r['pixels'] for r in layout['regions']])
    anchors=('CENTER','X_POS12','Y_NEG12','VY_POS');masks=('BASE','X4P','X4M','Y4P','Y4M','X8P','X8M','Y8P','Y8M')
    bykey={(s['input_id'],s['mask_name']):s for s in samples};selected=[bykey[(k,m)] for m in masks for k in anchors]
    truth=np.stack([integrate(s['image'],layout)/area for s in selected]);zf=checked(idx['features'],True)['mean_sums']/area;noise=[]
    for s in samples:
        v=zf[s['visits']];noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    cov,white=covariance(noise);scales=np.stack([integrate(s['image'],layout)/area for s in samples if s['mask_name']=='BASE']).std(0,ddof=1).clip(min=.01)
    relay=ASMRelay(b,layout,checked(cfg['geometry']),device,padding=4,
        phase_sign=cfg.get('phase_sign',1),slm_shift_bound=cfg.get('slm_shift_bound',32))
    with torch.no_grad():relay.q.copy_(torch.tensor(cfg['q'],device=device));relay.photo_initial[:]=torch.tensor(cfg['photometry'],device=device)
    require(relay.shape[0]==relay.shape[1],'rotation hypothesis requires square numerical grid')
    source=torch.stack([relay.source_field(k,bykey[(k,'BASE')]['dmd']) for k in anchors]);grid=relay.camera_grid().repeat(4,1,1,1)
    gray={m:torch.tensor(bykey[(anchors[0],m)]['gray'],device=device,dtype=torch.float32) for m in masks}
    shift=relay.slm_shift_bound*np.tanh(np.array(cfg['q'][2:4]));results=[];all_values=[]
    pairs=[(4*m+k,4*(m+1)+k) for m in (1,3,5,7) for k in range(4)]
    write(output/'PLAN.json',dict(model=ref(model/'MODEL.json'),source=ref(__file__),dataset=ref(ROOT/'JOINT_TRAIN_AUDIT_01/INDEX.json'),
        anchors=anchors,masks=masks,relative_transform_codes=list(range(8)),phase_signs=[1,-1],
        centre_delta_pixels=[-96,0,96],numerical_padding=4,current_hardware_config_unchanged=True,
        current_SLM_transform_is_delta0=True,TRAIN_only_screen_not_independent_validation=True,
        affine_zero_reference_does_NOT_identify_SLM_phase_plane_orientation=True))
    with torch.no_grad():
        for code in range(8):
            for sign in (1,-1):
                for dx in (-96,0,96):
                    for dy in (-96,0,96):
                        ss=shift+np.array([dx,dy]);ss=np.clip(ss,-.99*relay.slm_shift_bound,.99*relay.slm_shift_bound)
                        relay.q[2:4]=torch.tensor(np.arctanh(ss/relay.slm_shift_bound),device=device);relay.phase_sign=sign
                        values=[]
                        for mask in masks:
                            mod=oriented(relay.modulation(gray[mask]),code)
                            u=source*mod;u=propagate(propagate(u,relay.h2)*relay.l2,relay.h2)
                            u=oriented(u,relay.geometry['orientation']);gain=relay.photo_initial[0]*torch.exp(relay.q[11].clamp(-4,4))
                            raw=F.grid_sample(u.abs().square()[:,None],grid,align_corners=False)[:,0]*255*gain
                            raw=(raw+relay.photo_initial[1]+3*torch.tanh(relay.q[12])).clamp(0,255)
                            values.extend([(relay.integrate(im)/relay.areas).cpu().numpy() for im in raw])
                        v=np.asarray(values);mean=float(np.mean(((v-truth)/scales)**2));response=[];cos=[]
                        for p,m in pairs:
                            aa=white@(truth[p]-truth[m]);bb=white@(v[p]-v[m]);response.append(float(np.linalg.norm(aa-bb)/max(np.linalg.norm(aa),1e-12)))
                            cos.append(float(aa@bb/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-12)))
                        result=dict(transform_delta=code,phase_sign=sign,centre_delta=[dx,dy],actual_effective_shift=ss.tolist(),
                            mean_TRAIN_scaled_RMSE=float(np.sqrt(mean)),median_response_cosine=float(np.median(cos)),
                            median_response_relative_error=float(np.median(response)),score=float(mean+.5*np.mean(np.square(response))))
                        results.append(result);all_values.append(v)
                print('COORDINATE SCREEN',code,sign,min(r['score'] for r in results),flush=True)
    best=sorted(results,key=lambda r:r['score'])[:10]
    with (output/'READOUTS.npz').open('xb') as f:np.savez_compressed(f,truth=truth,predictions=np.array(all_values))
    write(output/'RESULT.json',dict(hypotheses=results,top10=best,source=ref(__file__),
        no_actual_coordinate_change=True,no_model_refinement=True,formal_TEST=False,SLM_training_allowed=False))
    print('COORDINATE TOP10',best,flush=True)


def selection_review(models,output):
    """Predeclared SELECT criterion and per-role gates; never reads TEST labels."""
    import csv
    from dataset import checked
    output.mkdir();candidates=[];dataset=None
    for model in models:
        plan=read(model/'PLAN.json');result=read(model/'RESULT.json');cache=checked(plan['cache'])
        if dataset is None:dataset=cache['dataset']
        require(cache['dataset']==dataset,'candidate data differ; no automatic comparison')
        require(plan['selection_criterion']=='mean_response','same declared SELECT criterion required')
        audit_path=ROOT/(model.name+'_AUDIT')/'RESULT.json';audit=read(audit_path)
        require(audit['model']==ref(model/'RESULT.json') and audit['plan']==ref(model/'PLAN.json'),'stale response audit')
        rows=cache['rows'];roles={(r['input_id'],r['mask']):plan['split'][i] for i,r in enumerate(rows)}
        pairs=[]
        for pair in audit['pairs']:
            pm=pair['axis'].split(' vs ');require(len(pm)==2,'explicit signed mask pair required')
            a,b=(roles[(pair['input_id'],m)] for m in pm);require(a==b,'paired partition differs')
            pairs.append(dict(pair,role=a))
        groups=[]
        for metric in result['metrics']:
            role=metric['split']
            if role=='TRAIN':continue
            pp=[p for p in pairs if p['role']==role];good=[p for p in pp if p['identifiable']]
            coverage=len(good)/len(pp) if pp else 0.
            values=None if not good else dict(
                median_cosine=float(np.median([p['predicted']['cosine'] for p in good])),
                direction_fraction=float(np.mean([p['predicted']['cosine']>0 for p in good])),
                amplitude_fraction_05_20=float(np.mean([.5<=p['predicted']['amplitude_ratio']<=2 for p in good])),
                median_relative_error=float(np.median([p['predicted']['relative_error'] for p in good])))
            improvement=1-metric['hybrid']['TRAIN_scaled_region_RMSE']/metric['physics']['TRAIN_scaled_region_RMSE']
            response_pass=bool(values and coverage>=.8 and values['median_cosine']>=.7 and
                values['direction_fraction']>=.8 and values['amplitude_fraction_05_20']>=.8 and values['median_relative_error']<=.75)
            groups.append(dict(role=role,conditions=metric['conditions'],pairs=len(pp),identifiable_fraction=coverage,
                mean_improvement_fraction=improvement,mean_pass=improvement>=.2,response=values,
                response_pass=response_pass,passed=bool(improvement>=.2 and response_pass)))
        require({r['role'] for r in groups}=={'SELECT_INPUT','SELECT_MASK','SELECT_BOTH'},'three disentangled selection roles required')
        history=read(model/'HISTORY.json');best=next(h for h in history if h['step']==result['best_step'])
        candidates.append(dict(model=model.name,model_reference=ref(model/'RESULT.json'),plan=ref(model/'PLAN.json'),
            audit=ref(audit_path),selection_score=best['SELECT_criterion'],groups=groups,
            SELECT_gate_passed=all(g['passed'] for g in groups)))
    selected=min(candidates,key=lambda r:r['selection_score'])
    write(output/'RESULT.json',dict(candidates=candidates,selected_model=selected['model'],
        selected_by='minimum predeclared SELECT scaled MSE + 0.5 relative response MSE',
        selected_gate_passed=selected['SELECT_gate_passed'],dataset=dataset,source=ref(__file__),
        created_unix_ns=time.time_ns(),formal_TEST_NOT_read=True,time_holdout_NOT_passed=True,
        SLM_training_allowed=False,closed_loop_qualified=False))
    flat=[dict(model=c['model'],score=c['selection_score'],role=g['role'],mean_improvement=g['mean_improvement_fraction'],
        coverage=g['identifiable_fraction'],**(g['response'] or {}),passed=g['passed']) for c in candidates for g in c['groups']]
    with (output/'metrics.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(1,3,figsize=(13,5));xx=np.arange(len(candidates))
    for j,role in enumerate(('SELECT_INPUT','SELECT_MASK','SELECT_BOTH')):
        gg=[next(g for g in c['groups'] if g['role']==role) for c in candidates]
        for ax,key in zip(axes,('mean_improvement','median_cosine','median_relative_error')):
            vals=[g['mean_improvement_fraction'] if key=='mean_improvement' else g['response'][key] if g['response'] else np.nan for g in gg]
            ax.plot(xx,vals,'o-',lw=.8,label=role)
    for ax,threshold,title in zip(axes,(.2,.7,.75),('Mean improvement (higher)','Response cosine (higher)','Response relative error (lower)')):
        ax.axhline(threshold,color='gray',ls='--');ax.set_ylabel(title)
        ax.set_xticks(xx,[c['model'] for c in candidates],rotation=75,ha='right',fontsize=6)
    axes[0].legend(fontsize=7);fig.suptitle('Disentangled development selection | not deployment qualification')
    fig.tight_layout();save_figure(fig,output/'01_selection_roles',180);plt.close(fig)
    print('SELECT REVIEW',selected['model'],selected['selection_score'],selected['SELECT_gate_passed'],selected['groups'],flush=True)


def forecast_evidence_check(forecast,cache,evaluation,output):
    """Verify pre-capture prediction commitment, exact bytes and chronology."""
    from dataset import checked
    output.mkdir();sealed=read(forecast/'FORECAST.json');index=read(cache/'INDEX.json')
    result=read(evaluation/'RESULT.json');data=checked(index['dataset'])
    known=sealed.get('known_inputs_only',False)
    require((result.get('evaluation_mode')=='time' if known else result['formal_TEST']) and result['model']==sealed['model'],
            'different model or evaluation partition')
    require(result['cache']==ref(cache/'INDEX.json'),'evaluation cache differs')
    require(index['physical_model']==sealed['physical_model'] and index['layout']==sealed['layout'],'forecast physics or readout changed')
    require(all(r['host_times_ns'][0]>sealed['created_unix_ns'] for r in data['rows']),'TEST capture predates sealed forecast')
    pred=checked(sealed['predictions'],True);after=np.load(evaluation/'PREDICTIONS.npz',allow_pickle=False)
    def key(row):return row['input_id'],row['dmd_reference']['sha256'],row['mask_reference']['sha256']
    lookup={key(r):i for i,r in enumerate(sealed['rows'])}
    count=len(sealed['rows']) if known else 80
    require(count>0 and len(lookup)==len(sealed['rows'])==count,'duplicate or incomplete pre-capture forecast')
    require(len(index['rows'])==count and {key(r) for r in index['rows']}==set(lookup),'capture input/mask pixels differ from commitment')
    maximum=0.
    for i,row in enumerate(index['rows']):
        j=lookup[key(row)]
        for name in ('predicted','physics'):
            maximum=max(maximum,float(np.max(np.abs(after[name][i]-pred[name][j]))))
    require(maximum<1e-4,'post-capture predictions differ from sealed forward model')
    write(output/'RESULT.json',dict(forecast=ref(forecast/'FORECAST.json'),cache=ref(cache/'INDEX.json'),
        evaluation=ref(evaluation/'RESULT.json'),source=ref(__file__),passed=True,
        conditions=count,known_time_revisit=known,max_prediction_difference_ADU=maximum,all_capture_after_freeze=True,
        exact_input_and_mask_SHA=True,model_NOT_changed=True,does_NOT_imply_metric_pass=True))


def known_forecast(model,assets,masks,output):
    """CPU-only pre-capture commitment for explicitly known time-revisit inputs."""
    import torch
    from dataset import checked
    from residual_fit import FrozenProxy
    output.mkdir();torch.set_num_threads(2);proxy=frozen_proxy(model,device='cpu')
    plan=read(assets/'PLAN.json');catalog=checked(plan['catalogs']['TIME_ANCHORS']);rows=[];images=[];physics=[]
    known={(r['input_id'],r['dmd_reference']['sha256'],r['mask_reference']['sha256']) for r in proxy.cache['rows']}
    with torch.no_grad():
        for name in masks:
            asset=plan['assets'][name];coeff=torch.tensor(asset['coefficients'],dtype=torch.float32)
            expected=proxy.gray(coeff).round().clamp(0,255).byte().repeat_interleave(8,0).repeat_interleave(8,1).numpy()
            require(np.array_equal(expected,checked(asset['patch'],True)),'time forecast SLM bytes differ')
            for row in catalog['inputs']:
                require((row['input_id'],row['dmd']['sha256'],asset['patch']['sha256']) in known,'time revisit must use known input/mask pixels')
                dmd=checked(row['dmd'],True);pred=proxy(row['input_id'],dmd,coeff)
                raw,_=proxy.relay(row['input_id'],dmd,proxy.gray(coeff))
                rows.append(dict(input_id=row['input_id'],dmd_reference=row['dmd'],mask_reference=asset['patch'],mask=name))
                images.append(pred.numpy());physics.append(raw.numpy())
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,predicted=images,physics=physics)
    write(output/'FORECAST.json',dict(model=ref(model/'RESULT.json'),plan=ref(model/'PLAN.json'),source=ref(__file__),
        physical_model=proxy.cache['physical_model'],layout=proxy.layout,rows=rows,created_unix_ns=time.time_ns(),
        predictions=ref(output/'PREDICTIONS.npz'),known_inputs_only=True,no_actual_CCD_read=True,
        purpose='fixed-candidate time diagnostic; not final model selection or complete 4D deployment qualification'))
    print('TIME FORECAST SEALED',len(rows),'conditions',flush=True)


def seal_test_predictions(model,design,output):
    """Exact future TEST bytes, immutable predictions, no measured TEST input."""
    import torch
    from dataset import checked
    output.mkdir();plan=read(design/'PLAN.json');proxy=frozen_proxy(model);rows=[];images=[];physics=[];seen=set()
    with torch.no_grad():
        for job in plan['jobs']:
            if job['role'] not in ('TEST_INPUT','TEST_MASK','TEST_BOTH'):continue
            asset=plan['assets'][job['mask']];coeff=torch.tensor(asset['coefficients'],device=proxy.device,dtype=torch.float32)
            exported=proxy.gray(coeff).round().clamp(0,255).byte().repeat_interleave(8,0).repeat_interleave(8,1).cpu().numpy()
            require(np.array_equal(exported,checked(asset['patch'],True)),'forecast SLM bytes differ')
            for row in checked(plan['catalogs'][job['catalog']])['inputs']:
                key=(row['input_id'],asset['patch']['sha256']);require(key not in seen,'duplicate TEST condition');seen.add(key)
                dmd=checked(row['dmd'],True);pred=proxy(row['input_id'],dmd,coeff);raw,_=proxy.relay(row['input_id'],dmd,proxy.gray(coeff))
                rows.append(dict(input_id=row['input_id'],dmd_reference=row['dmd'],mask_reference=asset['patch'],mask=job['mask'],
                    role=job['role'],coefficients_raw_gray=asset['coefficients']))
                images.append(pred.cpu().numpy());physics.append(raw.cpu().numpy())
    require(len(rows)==80,'frozen TEST protocol requires eighty conditions')
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,predicted=images,physics=physics)
    write(output/'FORECAST.json',dict(model=ref(model/'RESULT.json'),plan=ref(model/'PLAN.json'),design=ref(design/'PLAN.json'),
        source=ref(__file__),created_unix_ns=time.time_ns(),physical_model=proxy.cache['physical_model'],layout=proxy.layout,
        rows=rows,predictions=ref(output/'PREDICTIONS.npz'),actual_TEST_CCD_NOT_read=True,known_inputs_only=False,
        model_changed=False,source_mask_bytes_verified=True))
    print('FROZEN TEST SEALED',model.name,len(rows),flush=True)


def evaluate_sealed_predictions(forecast,dataset,output):
    """Score PRE-capture sealed native predictions, not post-capture recomputation."""
    from dataset import checked,integrate
    from fit import groups
    from residual_fit import signed_pairs
    from response import covariance
    output.mkdir();sealed=read(forecast/'FORECAST.json');plan=checked(sealed['plan']);checked(sealed['model'])
    index=read(dataset/'INDEX.json');train=checked(plan['cache']);layout=sealed['layout']
    require(index['layout']==layout and index['baseline']==checked(train['dataset'])['baseline'],'data contract changed')
    require(all(r['host_times_ns'][0]>sealed['created_unix_ns'] for r in index['rows']),'capture not later than frozen prediction')
    samples=groups(index);key=lambda r:(r['input_id'],r['dmd_reference']['sha256'],r['mask_reference']['sha256'])
    lookup={key(r):i for i,r in enumerate(sealed['rows'])};require(len(lookup)==len(sealed['rows']),'duplicate sealed condition')
    area=np.array([r['pixels'] for r in layout['regions']]);pred=checked(sealed['predictions'],True);rows=[];order=[];truth=[]
    for s in samples:
        k=(s['input_id'],s['label']['dmd']['sha256'],s['mask']['sha256']);require(k in lookup,'uncommitted input/mask pixels')
        j=lookup[k];record=sealed['rows'][j];known=sealed.get('known_inputs_only',False)
        role='TIME_HOLDOUT' if known else record['role']
        if not known:require(s['label'].get('new_platform_split')=='TEST' or s['mask_split']=='TEST','non-TEST condition')
        rows.append(dict(input_id=s['input_id'],mask=s['mask_name'],coefficients_raw_gray=s['coefficients'],role=role,visits=s['visits']))
        order.append(j);truth.append(s['image'])
    require(len(samples)==len(lookup) and set(order)==set(range(len(lookup))),'incomplete sealed conditions')
    arrays=dict(truth=np.stack(truth),physics=pred['physics'][order],predicted=pred['predicted'][order])
    z={k:np.stack([integrate(im,layout)/area for im in v]) for k,v in arrays.items()};scale=np.array(plan['state_scales_ADU'])
    td=checked(train['dataset']);tz=checked(td['features'],True)['mean_sums']/area;noise=[]
    for i in plan['train']:
        v=tz[train['rows'][i]['visits']];noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    cov,white=covariance(noise);pairs=[];metrics=[]
    for p,m in signed_pairs(rows):
        require(rows[p]['role']==rows[m]['role'],'pair roles differ');a=white@(z['truth'][p]-z['truth'][m]);norm=max(np.linalg.norm(a),1e-12)
        snr=np.linalg.norm(z['truth'][p]-z['truth'][m])/np.sqrt(np.trace(cov)*(1/len(rows[p]['visits'])+1/len(rows[m]['visits'])))
        item=dict(input_id=rows[p]['input_id'],mask_pair=[rows[p]['mask'],rows[m]['mask']],role=rows[p]['role'],SNR=float(snr),identifiable=bool(snr>=3))
        for name in ('physics','predicted'):
            b=white@(z[name][p]-z[name][m]);item[name]=dict(cosine=float(a@b/max(norm*np.linalg.norm(b),1e-12)),
                amplitude_ratio=float(np.linalg.norm(b)/norm),relative_error=float(np.linalg.norm(a-b)/norm))
        pairs.append(item)
    for role in dict.fromkeys(r['role'] for r in rows):
        ids=[i for i,r in enumerate(rows) if r['role']==role];item=dict(role=role,conditions=len(ids))
        for name in ('physics','predicted'):
            e=z[name][ids]-z['truth'][ids];item[name]=dict(TRAIN_scaled_RMSE=float(np.sqrt(np.mean((e/scale)**2))),
                region_RMS_ADU=float(np.sqrt(np.mean(e**2))),worst_region_scaled_RMSE=float(np.sqrt(np.mean((e/scale)**2,0)).max()))
        item['mean_improvement']=1-item['predicted']['TRAIN_scaled_RMSE']/max(item['physics']['TRAIN_scaled_RMSE'],1e-12)
        pp=[p for p in pairs if p['role']==role];good=[p for p in pp if p['identifiable']];coverage=len(good)/len(pp) if pp else 0
        values=None if not good else dict(median_cosine=float(np.median([p['predicted']['cosine'] for p in good])),
            direction_fraction=float(np.mean([p['predicted']['cosine']>0 for p in good])),
            amplitude_fraction_05_20=float(np.mean([.5<=p['predicted']['amplitude_ratio']<=2 for p in good])),
            median_relative_error=float(np.median([p['predicted']['relative_error'] for p in good])))
        passed=bool(values and coverage>=.8 and values['median_cosine']>=.7 and values['direction_fraction']>=.8 and
            values['amplitude_fraction_05_20']>=.8 and values['median_relative_error']<=.75)
        item.update(response_pairs=len(pp),identifiable_fraction=coverage,response=values,response_passed=passed,
            passed=bool(passed and item['mean_improvement']>=.2));metrics.append(item)
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    write(output/'RESULT.json',dict(forecast=ref(forecast/'FORECAST.json'),dataset=ref(dataset/'INDEX.json'),source=ref(__file__),
        metrics=metrics,response_pairs=pairs,scored_exact_PRE_capture_arrays=True,no_post_capture_forward_or_fit=True,
        formal_TEST=not sealed.get('known_inputs_only',False),static_model_gate_passed=all(m['passed'] for m in metrics),
        deployment_qualified=False,SLM_training_allowed=False))
    print('SEALED PREDICTION EVALUATION',metrics,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='PHYSICS_QA_01');p.add_argument('--cpu-reference');p.add_argument('--proxy')
    p.add_argument('--grid');p.add_argument('--response-geometry');p.add_argument('--physics-caches',nargs='+');p.add_argument('--coordinate-screen')
    p.add_argument('--selection-review',nargs='+');p.add_argument('--forecast-check');p.add_argument('--cache');p.add_argument('--evaluation')
    p.add_argument('--seal-test-model');p.add_argument('--design');p.add_argument('--score-sealed');p.add_argument('--dataset');a=p.parse_args()
    if a.cpu_reference:
        cpu_reference(Path(a.cpu_reference));sys.exit(0)
    require(Path(a.output).name==a.output,'fresh direct output required')
    if a.seal_test_model:
        require(a.design,'explicit frozen design required');seal_test_predictions(ROOT/a.seal_test_model,ROOT/a.design,ROOT/a.output)
    elif a.score_sealed:
        require(a.dataset,'explicit actual dataset required');evaluate_sealed_predictions(ROOT/a.score_sealed,ROOT/a.dataset,ROOT/a.output)
    elif a.selection_review:selection_review([ROOT/n for n in a.selection_review],ROOT/a.output)
    elif a.forecast_check:
        require(a.cache and a.evaluation,'explicit TEST cache/evaluation required')
        forecast_evidence_check(ROOT/a.forecast_check,ROOT/a.cache,ROOT/a.evaluation,ROOT/a.output)
    elif a.coordinate_screen:coordinate_screen(ROOT/a.coordinate_screen,ROOT/a.output)
    elif a.physics_caches:compare_physical_caches([ROOT/n for n in a.physics_caches],ROOT/a.output)
    elif a.response_geometry:response_geometry_check(ROOT/a.response_geometry,ROOT/a.output)
    elif a.grid:grid_check(ROOT/a.grid,ROOT/a.output)
    elif a.proxy:proxy_check(ROOT/a.proxy,ROOT/a.output)
    else:run(ROOT/a.output)
