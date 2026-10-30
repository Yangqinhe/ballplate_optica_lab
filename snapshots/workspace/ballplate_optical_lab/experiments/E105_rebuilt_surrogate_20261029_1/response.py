"""Offline response observability / direction audit in native readout units."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from workflow import ROOT,read,write,ref,require
from dataset import checked,integrate


def covariance(residuals,shrinkage=.2):
    a=np.asarray(residuals,np.float64)
    require(a.ndim==2 and len(a)>=2 and np.isfinite(a).all(),'finite repeated residuals required')
    c=a.T@a/len(a)
    c=(1-shrinkage)*c+shrinkage*np.diag(np.diag(c))
    eigen,vectors=np.linalg.eigh(c)
    floor=max(float(np.trace(c)/len(c))*.01,1e-10)
    return c,(vectors/np.sqrt(np.maximum(eigen,floor)))@vectors.T


def audit(dataset,output,physical=None):
    output.mkdir();index=read(dataset/'INDEX.json');features=checked(index['features'],True)
    layout=index['layout'];area=np.array([r['pixels'] for r in layout['regions']])
    z=features['mean_sums']/area
    assets=read(ROOT/'ASSETS_02/PLAN.json')['assets'];names={v['patch']['sha256']:k for k,v in assets.items()}
    groups=defaultdict(list)
    for i,r in enumerate(index['rows']):groups[(r['input_id'],r.get('mask_name') or names[r['mask']['sha256']])].append(i)
    residual=[]
    for ids in groups.values():
        require(len(ids)>=2,'repeat count must be >=2')
        residual.extend((z[ids]-z[ids].mean(0))*np.sqrt(len(ids)/(len(ids)-1)))
    cov,white=covariance(residual);predictions={}
    if physical is not None:
        import torch
        from fit import gpu,groups as samples
        from surrogate import ASMRelay
        config=read(physical/'MODEL.json');b=checked(index['baseline'])
        require(config['baseline']==index['baseline'] and config['layout']==layout,'model contract differs')
        device=gpu();relay=ASMRelay(b,layout,checked(config['geometry']),device,phase_sign=config.get('phase_sign',1),
            slm_shift_bound=config.get('slm_shift_bound',32.))
        with torch.no_grad():
            relay.q.copy_(torch.tensor(config['q'],device=device));relay.photo_initial[:]=torch.tensor(config['photometry'],device=device)
            for sample in samples(index):
                image,_=relay(sample['input_id'],sample['dmd'],torch.tensor(sample['gray'],device=device,dtype=torch.float32))
                predictions[(sample['input_id'],sample['mask_name'])]=relay.integrate(image).cpu().numpy()/area
    pairs=[]
    for input_id in dict.fromkeys(r['input_id'] for r in index['rows']):
        for axis in ('X4','Y4','X8','Y8'):
            plus=(input_id,axis+'P');minus=(input_id,axis+'M')
            if plus not in groups or minus not in groups:continue
            p,m=groups[plus],groups[minus];d=z[p].mean(0)-z[m].mean(0)
            noise=np.sqrt(np.trace(cov)*(1/len(p)+1/len(m)))
            snr=float(np.linalg.norm(d)/max(noise,1e-12))
            item=dict(input_id=input_id,axis=axis,signal_rms_ADU=float(np.sqrt(np.mean(d*d))),
                noise_of_mean_difference_rms_ADU=float(noise/8),signal_to_repeat_noise=snr,
                identifiable=snr>=3.,independent_visits=[len(p),len(m)],
                wallclock_gap_seconds=float((np.mean([index['rows'][i]['mean_host_ns'] for i in p])-
                                             np.mean([index['rows'][i]['mean_host_ns'] for i in m]))/1e9))
            if predictions:
                dh=predictions[plus]-predictions[minus];a=white@d;b=white@dh
                item.update(physical_whitened_cosine=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12)),
                    physical_relative_response_error=float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-12)),
                    physical_whitened_amplitude_ratio=float(np.linalg.norm(b)/max(np.linalg.norm(a),1e-12)))
            pairs.append(item)
    with (output/'NOISE.npz').open('xb') as f:np.savez_compressed(f,covariance_region_mean_ADU=cov,whitener=white,residuals_region_mean_ADU=residual)
    usable=[p for p in pairs if p['identifiable']]
    summary=dict(dataset=ref(dataset/'INDEX.json'),source=ref(__file__),
        model=None if physical is None else ref(physical/'MODEL.json'),pairs=pairs,
        identifiable_fraction=len(usable)/len(pairs) if pairs else None,
        physical_median_cosine=float(np.median([p['physical_whitened_cosine'] for p in usable])) if predictions and usable else None,
        physical_direction_fraction=float(np.mean([p['physical_whitened_cosine']>0 for p in usable])) if predictions and usable else None,
        pooled_effective_noise_rms_ADU=float(np.sqrt(np.trace(cov)/64)),
        threshold_SNR=3.,shrinkage=.2,whitening_eigenfloor_fraction=.01,
        scope='development pilot; sequential masks/time confounding retained; not a deployment gate',
        deployment_passed=False,noise=ref(output/'NOISE.npz'))
    write(output/'RESULT.json',summary)
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    x=np.arange(len(pairs));labels=[p['input_id']+'/'+p['axis'] for p in pairs]
    axes[0].bar(x,[p['signal_to_repeat_noise'] for p in pairs],color='#0072B2')
    axes[0].axhline(3,color='#D55E00',ls='--',label='development identifiable threshold');axes[0].set_ylabel('Response / repeat noise');axes[0].legend()
    if predictions:
        axes[1].plot(x,[p['physical_whitened_cosine'] for p in pairs],'o',color='#009E73')
        axes[1].axhline(.7,color='#D55E00',ls='--');axes[1].axhline(0,color='gray',lw=.6)
    axes[1].set_ylim(-1.05,1.05);axes[1].set_ylabel('Whitened response cosine')
    axes[1].set_xticks(x,labels,rotation=80,fontsize=6);fig.tight_layout();save_figure(fig,output/'01_response',180);plt.close(fig)
    print('RESPONSE',len(pairs),'pairs;',len(usable),'identifiable;',summary['physical_median_cosine'],flush=True)


def temporal(first,later,output):
    """Matched conditions over time, unchanged RAW; no online normalization."""
    output.mkdir();a=read(first/'INDEX.json');b=read(later/'INDEX.json')
    require(a['baseline']==b['baseline'] and a['layout']==b['layout'],'time comparison contracts differ')
    layout=a['layout'];area=np.array([r['pixels'] for r in layout['regions']])
    names={v['patch']['sha256']:k for k,v in read(ROOT/'ASSETS_02/PLAN.json')['assets'].items()}
    def prepare(idx):
        z=checked(idx['features'],True)['mean_sums']/area;im=checked(idx['means'],True)['images'];g=defaultdict(list)
        for i,r in enumerate(idx['rows']):g[(r['input_id'],r.get('mask_name') or names[r['mask']['sha256']])].append(i)
        return z,im,g
    za,ia,ga=prepare(a);zb,ib,gb=prepare(b);residual=[]
    for z,g in ((za,ga),(zb,gb)):
        for ids in g.values():
            residual.extend((z[ids]-z[ids].mean(0))*np.sqrt(len(ids)/(len(ids)-1)))
    cov,white=covariance(residual);conditions=[];deltas={}
    for key in ga.keys()&gb.keys():
        aa,bb=ga[key],gb[key];d=zb[bb].mean(0)-za[aa].mean(0);deltas[key]=d
        require({a['rows'][i]['mask']['sha256'] for i in aa}=={b['rows'][i]['mask']['sha256'] for i in bb},
                'time comparison must use identical mask bytes, not merely identical names')
        ma,mb=ia[aa].mean(0),ib[bb].mean(0);border=np.ones(ma.shape,bool);border[32:-32,32:-32]=False
        background_delta=float(mb[border].mean()-ma[border].mean())
        # Fixed geometric outer border, diagnostic only; neither saved RAW nor
        # exported control input is altered by these comparisons.
        noise=np.sqrt(np.trace(cov)*(1/len(aa)+1/len(bb)))
        conditions.append(dict(input_id=key[0],mask=key[1],
            time_gap_seconds=float((np.mean([b['rows'][i]['mean_host_ns'] for i in bb])-
                                    np.mean([a['rows'][i]['mean_host_ns'] for i in aa]))/1e9),
            across_time_region_RMS_ADU=float(np.sqrt(np.mean(d*d))),
            drift_to_within_visit_noise=float(np.linalg.norm(d)/max(noise,1e-12)),
            border_background_delta_ADU=background_delta,
            diagnostic_after_background_RMS_ADU=float(np.sqrt(np.mean((d-background_delta)**2)))))
    pairs=[]
    for input_id in dict.fromkeys(r['input_id'] for r in a['rows']):
        for axis in ('X4','Y4','X8','Y8'):
            p=(input_id,axis+'P');m=(input_id,axis+'M')
            if not all(k in ga and k in gb for k in (p,m)):continue
            da=za[ga[p]].mean(0)-za[ga[m]].mean(0);db=zb[gb[p]].mean(0)-zb[gb[m]].mean(0)
            wa,wb=white@da,white@db
            pairs.append(dict(input_id=input_id,axis=axis,
                raw_cosine=float(da@db/max(np.linalg.norm(da)*np.linalg.norm(db),1e-12)),
                whitened_cosine=float(wa@wb/max(np.linalg.norm(wa)*np.linalg.norm(wb),1e-12)),
                whitened_relative_repeat_error=float(np.linalg.norm(wb-wa)/max(np.linalg.norm(wa),1e-12)),
                first_response_RMS_ADU=float(np.sqrt(np.mean(da*da))),
                later_response_RMS_ADU=float(np.sqrt(np.mean(db*db)))))
    write(output/'RESULT.json',dict(first=ref(first/'INDEX.json'),later=ref(later/'INDEX.json'),source=ref(__file__),
        matched_conditions=conditions,response_repeats=pairs,no_preprocessing_applied=True,
        new_mask_causal_effect_not_separated_from_time_automatically=True, deployment_passed=False))
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(2,1,figsize=(11,6))
    labels=[r['input_id']+'/'+r['mask'] for r in conditions]
    axes[0].bar(np.arange(len(conditions)),[r['across_time_region_RMS_ADU'] for r in conditions],color='#0072B2')
    axes[0].set_xticks(np.arange(len(conditions)),labels,rotation=80,fontsize=6);axes[0].set_ylabel('Across-time difference / ADU')
    axes[1].plot([r['whitened_cosine'] for r in pairs],'o',color='#D55E00');axes[1].axhline(0,color='gray',lw=.6)
    axes[1].set_ylim(-1.05,1.05);axes[1].set_xticks(np.arange(len(pairs)),[r['input_id']+'/'+r['axis'] for r in pairs],rotation=60,fontsize=7)
    axes[1].set_ylabel('Early vs late response cosine');fig.tight_layout();save_figure(fig,output/'01_time_consistency',180);plt.close(fig)
    print('TIME',len(conditions),'matched conditions,',len(pairs),'response pairs',flush=True)


def hybrid_audit(model,output):
    output.mkdir();plan=read(model/'PLAN.json');result=read(model/'RESULT.json');cache=checked(plan['cache'])
    idx=checked(cache['dataset']);layout=idx['layout'];area=np.array([r['pixels'] for r in layout['regions']])
    prediction=checked(result['predictions'],True);raw=checked(idx['features'],True)['mean_sums']/area
    residual=[];train=set(plan['train']);rows=cache['rows']
    for i in train:
        rr=raw[rows[i]['visits']];residual.extend((rr-rr.mean(0))*np.sqrt(len(rr)/(len(rr)-1)))
    cov,white=covariance(residual)
    vectors={k:np.stack([integrate(image,layout)/area for image in prediction[k]]) for k in ('truth','physics','predicted')}
    lookup={(r['input_id'],r['mask']):i for i,r in enumerate(rows)}
    require(len(lookup)==len(rows),'multiple amplitudes need distinct mask identifiers')
    pairs=[]
    from residual_fit import signed_pairs
    for p,m in signed_pairs(rows):
            inp=rows[p]['input_id'];axis=rows[p]['mask']+' vs '+rows[m]['mask']
            d=vectors['truth'][p]-vectors['truth'][m];wa=white@d
            noise=np.sqrt(np.trace(cov)*(1/len(rows[p]['visits'])+1/len(rows[m]['visits'])))
            snr=float(np.linalg.norm(d)/max(noise,1e-12))
            row=dict(input_id=inp,axis=axis,both_conditions_in_TRAIN=p in train and m in train,
                SNR=snr,identifiable=snr>=3.)
            for name in ('physics','predicted'):
                dh=vectors[name][p]-vectors[name][m];wb=white@dh
                row[name]=dict(cosine=float(wa@wb/max(np.linalg.norm(wa)*np.linalg.norm(wb),1e-12)),
                    amplitude_ratio=float(np.linalg.norm(wb)/max(np.linalg.norm(wa),1e-12)),
                    relative_error=float(np.linalg.norm(wb-wa)/max(np.linalg.norm(wa),1e-12)))
            pairs.append(row)
    summary=[]
    for scope in ('ALL','TRAIN','SELECT'):
        allpairs=[p for p in pairs if scope=='ALL' or p['both_conditions_in_TRAIN']==(scope=='TRAIN')]
        pp=[p for p in allpairs if p['identifiable']];s=dict(scope=scope,total_pairs=len(allpairs),identifiable=len(pp))
        for name in ('physics','predicted'):
            s[name]=None if not pp else dict(median_cosine=float(np.median([p[name]['cosine'] for p in pp])),
                direction_fraction=float(np.mean([p[name]['cosine']>0 for p in pp])),
                amplitude_fraction_05_20=float(np.mean([.5<=p[name]['amplitude_ratio']<=2 for p in pp])),
                median_relative_error=float(np.median([p[name]['relative_error'] for p in pp])))
        summary.append(s)
    write(output/'RESULT.json',dict(model=ref(model/'RESULT.json'),plan=ref(model/'PLAN.json'),source=ref(__file__),
        pairs=pairs,summary=summary,TRAIN_noise_only=True,formal_TEST=False,
        physical_calibration_saw_some_pilot_SELECT_masks=not any(r.get('input_split')=='TRAIN' for r in rows),
        SLM_training_allowed=False))
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for name,color in (('physics','#D55E00'),('predicted','#0072B2')):
        axes[0].plot([r[name]['cosine'] for r in pairs],'o-',lw=.6,label=name,color=color)
        axes[1].plot([r[name]['relative_error'] for r in pairs],'o-',lw=.6,label=name,color=color)
    axes[0].axhline(.7,color='gray',ls='--');axes[0].set_ylim(-1.05,1.05);axes[0].set_ylabel('TRAIN-whitened cosine');axes[0].legend()
    axes[1].axhline(.75,color='gray',ls='--');axes[1].set_ylabel('Relative response error')
    axes[1].set_xticks(np.arange(len(pairs)),[r['input_id']+'/'+r['axis'] for r in pairs],rotation=80,fontsize=6)
    fig.tight_layout();save_figure(fig,output/'01_hybrid_response',180);plt.close(fig)
    print('HYBRID RESPONSE',summary,flush=True)


def finite_response_modes(matrix,coefficients):
    """Finite-difference geometry, NOT a calibrated optical differential."""
    j=np.asarray(matrix,np.float64);c=np.asarray(coefficients,np.float64)
    require(j.ndim==2 and j.shape[1]==4 and j.shape[0]>=4 and c.shape==(4,),
            'four finite response columns required')
    require(np.isfinite(j).all() and np.isfinite(c).all(),'finite response data required')
    _,s,v=np.linalg.svd(j,full_matrices=False);norm=np.linalg.norm(j,axis=0)
    gram=j.T@j/np.maximum(norm[:,None]*norm[None,:],1e-12)
    independent_energy=np.sqrt(np.sum((c*norm)**2))
    return dict(singular_values=s.tolist(),right_vectors=v.tolist(),column_cosines=gram.tolist(),
        mixture_over_orthogonal_energy=float(np.linalg.norm(j@c)/max(independent_energy,1e-12)))


def basis_diagnostic(model,score,output):
    """Post-TEST diagnosis: TRAIN-only finite response modes; no gate/refit."""
    from optical_lab.reporting import plotting,save_figure
    output.mkdir();plan=read(model/'PLAN.json');cache=checked(plan['cache']);data=checked(cache['dataset'])
    rows=cache['rows'];train=set(plan['train']);layout=cache['layout']
    area=np.array([r['pixels'] for r in layout['regions']]);visits=checked(data['features'],True)['mean_sums']/area
    noise=[]
    for i in train:
        v=visits[rows[i]['visits']];noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    _,white=covariance(noise);z=np.stack([visits[r['visits']].mean(0) for r in rows])
    lookup={(r['input_id'],r['mask']):i for i,r in enumerate(rows)}
    axes=('X4','Y4','X8','Y8');matrices={};skipped=[]
    for inp in dict.fromkeys(r['input_id'] for r in rows):
        keys=[(inp,a+s) for a in axes for s in ('P','M')]
        if not all(k in lookup and lookup[k] in train for k in keys):skipped.append(inp);continue
        columns=[]
        for k,a in enumerate(axes):
            p,m=lookup[(inp,a+'P')],lookup[(inp,a+'M')];expected=np.eye(4)[k]*24
            require(np.array_equal(rows[p]['coefficients_raw_gray'],expected) and
                    np.array_equal(rows[m]['coefficients_raw_gray'],-expected),'finite probe coefficients differ')
            columns.append((z[p]-z[m])/48.)
        matrices[inp]=np.stack(columns,axis=1)
    require(matrices,'no complete TRAIN axis responses')
    stacked=np.concatenate([white@j for j in matrices.values()])/np.sqrt(len(matrices))
    modes=finite_response_modes(stacked,np.zeros(4));design=read(ROOT/'DESIGN_STRONG_N3_01/PLAN.json')
    mixing=[]
    for name in ('MIX04P','MIX05P','MIX06P','MIX07P'):
        c=np.array(design['assets'][name]['coefficients']);rr=[]
        for inp,j in matrices.items():
            r=finite_response_modes(white@j,c);rr.append(dict(input=inp,ratio=r['mixture_over_orthogonal_energy']))
        mixing.append(dict(mask=name,coefficients=c.tolist(),rows=rr,
            median_ratio=float(np.median([r['ratio'] for r in rr]))))
    scored=read(score/'RESULT.json');test=checked(scored['dataset'])
    require(test['layout']==layout and test['baseline']==data['baseline'],'diagnostic platform differs')
    tv=checked(test['features'],True)['mean_sums']/area;g=defaultdict(list)
    for i,r in enumerate(test['rows']):g[(r['input_id'],r['mask_name'])].append(i)
    comparisons=[]
    for p in scored['response_pairs']:
        inp=p['input_id'];plus,minus=p['mask_pair']
        if inp not in matrices or not plus.startswith('MIX'):continue
        ai,bi=g[(inp,plus)],g[(inp,minus)]
        ca=np.array(test['rows'][ai[0]]['coefficients']);cb=np.array(test['rows'][bi[0]]['coefficients'])
        real=white@(tv[ai].mean(0)-tv[bi].mean(0));est=white@(matrices[inp]@(ca-cb))
        comparisons.append(dict(input_id=inp,masks=[plus,minus],SNR=p['SNR'],identifiable=p['identifiable'],
            cosine=float(real@est/max(np.linalg.norm(real)*np.linalg.norm(est),1e-12)),
            relative_error=float(np.linalg.norm(real-est)/max(np.linalg.norm(real),1e-12)),
            CNN_cosine=p['predicted']['cosine']))
    write(output/'RESULT.json',dict(model=ref(model/'RESULT.json'),TRAIN_data=cache['dataset'],
        source=ref(__file__),score=ref(score/'RESULT.json'),complete_TRAIN_inputs=list(matrices),skipped_inputs=skipped,
        stacked_whitened_finite_response=modes,mixed_direction_diagnostics=mixing,
        same_known_input_empirical_response_comparisons=comparisons,
        gray_step=24,finite_difference_not_derivative=True,gray_clipping_and_time_confounding_not_removed=True,
        diagnosis_after_TEST_NOT_new_holdout=True,no_new_basis_applied=True,no_model_or_gate_change=True,
        input_specific_lookup_NOT_an_online_controller=True))
    plt=plotting();fig,aa=plt.subplots(1,3,figsize=(12,3.8))
    s=np.array(modes['singular_values']);aa[0].bar(range(1,5),s/s[0],color='#0072B2')
    aa[0].set_xlabel('Response mode');aa[0].set_ylabel('Singular value / largest')
    im=aa[1].imshow(modes['column_cosines'],vmin=-1,vmax=1,cmap='RdBu_r')
    aa[1].set_xticks(range(4),axes);aa[1].set_yticks(range(4),axes);fig.colorbar(im,ax=aa[1],fraction=.046)
    aa[1].set_title('TRAIN response column cosine',fontsize=9)
    for i,r in enumerate(mixing):
        yy=np.array([x['ratio'] for x in r['rows']]);aa[2].scatter(np.full(len(yy),i),yy,s=9,color='#0072B2',alpha=.6)
        aa[2].plot(i,np.median(yy),'_',color='#D55E00',ms=14)
    aa[2].axhline(1,color='.5',ls='--');aa[2].set_xticks(range(4),['MIX04','MIX05','MIX06','MIX07'])
    aa[2].set_ylabel('Combined / orthogonal-energy prediction')
    fig.suptitle('Post-TEST diagnostic | TRAIN-only finite response modes; no optical parameter changes',fontsize=9)
    fig.tight_layout();save_figure(fig,output/'01_response_modes',180);plt.close(fig)
    print('FINITE RESPONSE MODES',len(matrices),'inputs; normalized singular values',s/s[0],
          'mix ratios',[(r['mask'],r['median_ratio']) for r in mixing],flush=True)
    print('SAME-INPUT MIX',len(comparisons),'pairs; median cosine',np.median([r['cosine'] for r in comparisons]),flush=True)


def compare_models(models,output):
    """Evidence table and fixed-scale images, no re-fitting or model selection."""
    import csv
    output.mkdir();items=[];reference=None;cache=None
    for model in models:
        plan=read(model/'PLAN.json');result=read(model/'RESULT.json')
        if reference is None:reference=plan['cache'];cache=checked(reference)
        require(plan['cache']==reference,'comparison requires identical raw/physical cache')
        idx=checked(cache['dataset']);areas=np.array([r['pixels'] for r in cache['layout']['regions']])
        zz=checked(idx['features'],True)['mean_sums']/areas;noise=[]
        for i in plan['train']:
            values=zz[cache['rows'][i]['visits']]
            noise.extend((values-values.mean(0))*np.sqrt(len(values)/(len(values)-1)))
        cov,white=covariance(noise);pred=checked(result['predictions'],True)
        vectors={name:np.stack([integrate(im,cache['layout'])/areas for im in pred[name]]) for name in ('truth','physics','predicted')}
        from residual_fit import signed_pairs
        train=set(plan['train']);pairs=[]
        for p,m in signed_pairs(cache['rows']):
            if p in train or m in train:continue
            delta=vectors['truth'][p]-vectors['truth'][m]
            snr=np.linalg.norm(delta)/np.sqrt(np.trace(cov)*(1/len(cache['rows'][p]['visits'])+1/len(cache['rows'][m]['visits'])))
            if snr<3:continue
            a=white@delta;b=white@(vectors['predicted'][p]-vectors['predicted'][m])
            norm=max(np.linalg.norm(a),1e-12)
            pairs.append(dict(cosine=float(a@b/max(norm*np.linalg.norm(b),1e-12)),
                relative_error=float(np.linalg.norm(a-b)/norm),amplitude=float(np.linalg.norm(b)/norm)))
        require(pairs,'identifiable SELECT response pairs required')
        for metric in result['metrics']:
            items.append(dict(model=model.name,architecture=plan['architecture'],seed=plan['seed'],
                parameters=result['model_parameters'],split=metric['split'],conditions=metric['conditions'],
                scaled_RMSE=metric['hybrid']['TRAIN_scaled_region_RMSE'],
                physics_scaled_RMSE=metric['physics']['TRAIN_scaled_region_RMSE'],
                region_RMSE_ADU=metric['hybrid']['region_RMSE_ADU'],
                SELECT_response_median_cosine=float(np.median([p['cosine'] for p in pairs])),
                SELECT_response_median_relative_error=float(np.median([p['relative_error'] for p in pairs])),
                SELECT_response_identifiable_pairs=len(pairs),model_reference=ref(model/'RESULT.json')))
    write(output/'RESULT.json',dict(cache=reference,models=[ref(p/'RESULT.json') for p in models],
        source=ref(__file__),metrics=items,formal_TEST=False,
        no_model_selection_or_training=True,physical_calibration_may_include_pilot_SELECT=True))
    with (output/'metrics.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();labels=[p.name for p in models];roles=list(dict.fromkeys(r['split'] for r in items))
    fig,axes=plt.subplots(1,3,figsize=(13,4.4));xx=np.arange(len(labels));width=.7/len(roles)
    for j,role in enumerate(roles):
        rows=[r for r in items if r['split']==role]
        axes[0].bar(xx+(j-(len(roles)-1)/2)*width,[r['scaled_RMSE'] for r in rows],width,label=role)
    unique=[next(r for r in items if r['model']==label) for label in labels]
    axes[1].bar(xx,[r['SELECT_response_median_cosine'] for r in unique],color='#0072B2')
    axes[1].axhline(.7,color='#999999',ls='--');axes[1].set_ylim(-1,1)
    axes[2].bar(xx,[r['SELECT_response_median_relative_error'] for r in unique],color='#0072B2')
    axes[2].axhline(.75,color='#999999',ls='--')
    for ax in axes:ax.set_xticks(xx,labels,rotation=65,ha='right',fontsize=6)
    axes[0].set_ylabel('Region RMSE / TRAIN state scale');axes[0].legend(fontsize=6)
    axes[1].set_ylabel('SELECT response cosine');axes[2].set_ylabel('SELECT relative response error')
    fig.suptitle('Development comparison | same data; NOT independent TEST')
    fig.tight_layout();save_figure(fig,output/'01_mean_and_response',180);plt.close(fig)
    # First supplied model is an explicit visualization choice, not a winner.
    model=models[0];pred=checked(read(model/'RESULT.json')['predictions'],True)
    ids=[i for i,r in enumerate(cache['rows']) if r['mask']=='BASE' and r['input_id'] in ('CENTER','X_POS12','Y_POS12','VY_POS')]
    require(ids,'explicit diagnostic inputs absent')
    fig,axes=plt.subplots(len(ids),4,figsize=(10,2.5*len(ids)),squeeze=False)
    for row,i in enumerate(ids):
        for col,(name,title) in enumerate((('truth','Measured mean'),('physics','Physics'),('predicted','Hybrid'))):
            axes[row,col].imshow(pred[name][i],cmap='gray',vmin=10,vmax=20)
            axes[row,col].set_title(cache['rows'][i]['input_id']+' | '+title,fontsize=8)
        axes[row,3].imshow(pred['predicted'][i]-pred['truth'][i],cmap='RdBu_r',vmin=-2,vmax=2)
        axes[row,3].set_title('Hybrid - measured, [-2, 2] ADU',fontsize=8)
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(model.name+' | common display [10,20] ADU; no per-image normalization',fontsize=9)
    fig.tight_layout();save_figure(fig,output/'02_fixed_scale_CCD',180);plt.close(fig)
    print('MODEL COMPARISON',len(models),'candidates; no TEST or selection',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');p.add_argument('--output',required=True)
    p.add_argument('--hybrid');p.add_argument('--compare',nargs='+')
    p.add_argument('--basis-model');p.add_argument('--score')
    p.add_argument('--physical');p.add_argument('--later');a=p.parse_args()
    require(all(Path(n).name==n for n in (a.output,) if n is not None),'explicit direct folders required')
    if a.basis_model:
        require(a.score,'explicit frozen TEST score required')
        basis_diagnostic(ROOT/a.basis_model,ROOT/a.score,ROOT/a.output)
    elif a.compare:compare_models([ROOT/name for name in a.compare],ROOT/a.output)
    elif a.hybrid:hybrid_audit(ROOT/a.hybrid,ROOT/a.output)
    elif a.later:temporal(ROOT/a.dataset,ROOT/a.later,ROOT/a.output)
    else:audit(ROOT/a.dataset,ROOT/a.output,None if a.physical is None else ROOT/a.physical)
