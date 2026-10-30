"""File-only evidence pack. Explicit datasets/models, no SDK or GUI imports."""
import argparse
import csv
from pathlib import Path
import numpy as np
from workflow import ROOT,read,write,ref,require
from dataset import checked,integrate


def freeze_static(forecast, output):
    """TRAIN-only constant residual, sealed before new measured labels exist."""
    import time
    output.mkdir();sealed=read(forecast/'FORECAST.json');plan=checked(sealed['plan']);model=checked(sealed['model'])
    arrays=checked(model['predictions'],True);train=plan['train']
    correction=(arrays['truth'][train]-arrays['physics'][train]).mean(0)
    with (output/'STATIC_CORRECTION.npy').open('xb') as f:np.save(f,correction,allow_pickle=False)
    write(output/'FREEZE.json',dict(source=ref(__file__),forecast=ref(forecast/'FORECAST.json'),
        model=sealed['model'],plan=sealed['plan'],correction=ref(output/'STATIC_CORRECTION.npy'),
        created_unix_ns=time.time_ns(),TRAIN_conditions=len(train),
        definition='clip(physical_prediction + mean_TRAIN(measured - physical), 0, 255)',
        no_new_input_or_mask_dependence=True,no_TEST_read=True,no_model_selection_change=True))


def static_evaluation(frozen, score, output):
    """Compare a frozen static baseline without changing candidate or gates."""
    from fit import groups
    from optical_lab.reporting import plotting,save_figure
    output.mkdir();f=read(frozen/'FREEZE.json');result=read(score/'RESULT.json')
    require(result['forecast']==f['forecast'],'static and hybrid predictions use different commitment')
    forecast=checked(f['forecast']);data=checked(result['dataset']);plan=checked(f['plan'])
    require(all(r['host_times_ns'][0]>f['created_unix_ns'] for r in data['rows']),'data captured before static baseline freeze')
    arr=np.load(score/'PREDICTIONS.npz',allow_pickle=False);correction=checked(f['correction'],True)
    images=dict(truth=arr['truth'],physics=arr['physics'],static=(arr['physics']+correction).clip(0,255),hybrid=arr['predicted'])
    area=np.array([r['pixels'] for r in data['layout']['regions']]);scale=np.array(plan['state_scales_ADU'])
    z={k:np.stack([integrate(im,data['layout'])/area for im in v]) for k,v in images.items()}
    bykey={(r['input_id'],r['dmd_reference']['sha256'],r['mask_reference']['sha256']):r['role'] for r in forecast['rows']}
    roles=[bykey[(s['input_id'],s['label']['dmd']['sha256'],s['mask']['sha256'])] for s in groups(data)]
    metrics=[]
    for role in dict.fromkeys(roles):
        ids=[i for i,r in enumerate(roles) if r==role];row=dict(role=role,conditions=len(ids))
        for k in ('physics','static','hybrid'):
            e=z[k][ids]-z['truth'][ids]
            row[k]=dict(region_RMS_ADU=float(np.sqrt(np.mean(e**2))),TRAIN_scaled_RMSE=float(np.sqrt(np.mean((e/scale)**2))))
        metrics.append(row)
    plt=plotting();fig,ax=plt.subplots(figsize=(8,4))
    for j,k,color in [(0,'physics','#D55E00'),(1,'static','#999999'),(2,'hybrid','#009E73')]:
        ax.bar(np.arange(len(metrics))+(j-1)*.23,[m[k]['TRAIN_scaled_RMSE'] for m in metrics],.23,color=color,label=k)
    ax.set_xticks(range(len(metrics)),[m['role'] for m in metrics]);ax.set_ylabel('RMSE / frozen TRAIN scale');ax.legend()
    ax.set_title('Frozen TEST: fixed TRAIN residual vs input/mask-conditioned hybrid',fontsize=10)
    fig.tight_layout();save_figure(fig,output/'01_static_baseline',180);plt.close(fig)
    write(output/'RESULT.json',dict(freeze=ref(frozen/'FREEZE.json'),score=ref(score/'RESULT.json'),source=ref(__file__),
        metrics=metrics,no_refit=True,no_gate_change=True,no_model_selection=True,not_a_control_result=True))


def superposition(dataset,physical,output):
    """Diagnostic intensity-additivity check, including measured OFF leakage."""
    import torch
    from fit import groups
    from residual_fit import relay_from_config
    from optical_lab.reporting import plotting,save_figure
    output.mkdir();idx=read(dataset/'INDEX.json');cfg=read(physical/'MODEL.json')
    require(idx['baseline']==cfg['baseline'],'different platform in superposition check')
    samples=groups(idx);layout=idx['layout'];area=np.array([r['pixels'] for r in layout['regions']])
    features=checked(idx['features'],True)['mean_sums']/area
    torch.set_num_threads(2);relay=relay_from_config(cfg,'cpu');rows=[];plt=plotting()
    fig,axes=plt.subplots(2,2,figsize=(12,7));archive={}
    for column,parent in enumerate(('CENTER','VY_POS')):
        ss={s['label']['diagnostic_subset']:s for s in samples if s['label']['diagnostic_parent']==parent}
        require(set(ss)=={'ALL','S0','S1','S2','S3','OFF'},'complete declared subsets required')
        real={k:s['image'] for k,s in ss.items()};pred={}
        with torch.no_grad():
            for k,s in ss.items():
                pred[k]=relay(s['input_id'],s['dmd'],torch.tensor(s['gray'],dtype=torch.float32))[0].numpy()
        curves={};record=dict(input=parent)
        for name,images in [('measured',real),('physics',pred)]:
            signal=images['ALL']-images['OFF']
            additive=sum(images['S%d'%k] for k in range(4))-4*images['OFF']
            cross=signal-additive;z=integrate(signal,layout)/area;q=integrate(cross,layout)/area
            curves[name]=(z,q);archive[parent+'_'+name+'_cross']=cross
            record[name]=dict(net_region_RMS_ADU=float(np.sqrt(np.mean(z*z))),
                nonadditive_region_RMS_ADU=float(np.sqrt(np.mean(q*q))),
                relative_nonadditive_region_L2=float(np.linalg.norm(q)/max(np.linalg.norm(z),1e-12)),
                relative_nonadditive_pixel_L2=float(np.linalg.norm(cross)/max(np.linalg.norm(signal),1e-12)))
        variance=np.zeros(len(area))
        for k,s in ss.items():
            v=features[s['visits']];require(len(v)>=3,'three independent visits required')
            variance+=(9 if k=='OFF' else 1)*v.var(0,ddof=1)/len(v)
        noise=float(np.sqrt(np.mean(variance)));record['estimated_independent_repeat_uncertainty_RMS_ADU']=noise
        record['nonadditive_over_repeat_uncertainty']=record['measured']['nonadditive_region_RMS_ADU']/max(noise,1e-12)
        rows.append(record)
        for name,color in [('measured','#222222'),('physics','#D55E00')]:
            axes[0,column].plot(range(len(area)),curves[name][0],color=color,label=name)
            axes[1,column].plot(range(len(area)),curves[name][1],color=color,label=name)
        axes[0,column].set_title(parent+' | all - OFF');axes[0,column].set_ylabel('Net region mean / ADU')
        axes[1,column].set_title('all - sum(single) + 3 OFF');axes[1,column].set_ylabel('Nonadditive residual / ADU')
        axes[1,column].set_xlabel('Original region index');axes[1,column].axhline(0,color='.7',lw=.5)
    axes[0,0].legend();fig.suptitle('Fixed original SLM; diagnostic only, not model selection or phase identification')
    fig.tight_layout();save_figure(fig,output/'01_intensity_additivity',180);plt.close(fig)
    with (output/'RESIDUALS.npz').open('xb') as f:np.savez_compressed(f,**archive)
    write(output/'RESULT.json',dict(dataset=ref(dataset/'INDEX.json'),physical=ref(physical/'MODEL.json'),source=ref(__file__),
        metrics=rows,DMD_OFF_is_not_physical_dark=True,static_drift_and_background_can_also_break_additivity=True,
        no_coherence_cause_identified=True,no_refit=True,no_controller=True,not_independent_final_TEST=True))
    print('SUPERPOSITION DIAGNOSTIC',rows,flush=True)


def inventory(datasets,output):
    """Deduplicate paths, not image content: identical pixels may be new frames."""
    output.mkdir();source=output/'research_report.py'
    with source.open('xb') as f:f.write(Path(__file__).read_bytes())
    seen={};queries={};conditions=set();summaries=[]
    for folder in datasets:
        idx=read(folder/'INDEX.json');local=set()
        for row in idx['rows']:
            query=row['archive'];previous=queries.setdefault(query['path'],query['sha256'])
            require(previous==query['sha256'],'same query path changed bytes')
            conditions.add((row['label']['dmd']['sha256'],row['mask']['sha256']))
            require(len(row['frames'])==row['n'],'RAW budget differs')
            for frame in row['frames']:
                raw=frame['raw'];require(frame['valid'],'invalid frame in audited index')
                old=seen.setdefault(raw['path'],raw['sha256']);require(old==raw['sha256'],'RAW path changed SHA')
                local.add(raw['path'])
        summaries.append(dict(dataset=ref(folder/'INDEX.json'),queries=len(idx['rows']),RAW=len(local),
            peak_ADU=idx['peak_ADU'],clipped_pixel_observations=idx['clipped_pixel_observations'],
            roles=idx.get('dataset_roles',list(dict.fromkeys(r.get('role','UNSPECIFIED_IN_OLD_INDEX') for r in idx['rows'])))))
    with (output/'RAW_INDEX.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['path','sha256'])
        writer.writerows(sorted(seen.items()))
    write(output/'RESULT.json',dict(source=ref(source),datasets=summaries,unique_RAW=len(seen),
        unique_effective_queries=len(queries),unique_DMD_mask_pixel_pairs=len(conditions),
        raw_manifest=ref(output/'RAW_INDEX.csv'),RAW_not_copied=True,failed_batches_NOT_counted=True,
        hash_verified_by_dataset_audit_not_reread_here=True,frames_NOT_independent_states=True))


def sealed_detail(folder, output, prefix):
    """Fixed-rule diagnostic plots and input-cluster intervals; no gate changes."""
    from fit import groups
    from optical_lab.reporting import plotting,save_figure
    result=read(folder/'RESULT.json');forecast=checked(result['forecast'])
    data=checked(result['dataset']);samples=groups(data)
    arrays=np.load(folder/'PREDICTIONS.npz',allow_pickle=False)
    area=np.array([r['pixels'] for r in data['layout']['regions']])
    z={k:np.stack([integrate(im,data['layout'])/area for im in arrays[k]])
       for k in ('truth','physics','predicted')}
    lookup={(s['input_id'],s['mask_name']):i for i,s in enumerate(samples)}
    require(len(lookup)==len(samples),'duplicate sealed condition in report')
    with (output/(prefix+'_all_regions.csv')).open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['input','mask','region','measured_3xN20_ADU','physics_ADU','hybrid_ADU'])
        for i,s in enumerate(samples):
            for j in range(len(area)):
                w.writerow([s['input_id'],s['mask_name'],j,*[float(z[k][i,j]) for k in ('truth','physics','predicted')]])
    plt=plotting();pairs=result['response_pairs'];intervals=[]
    roles=list(dict.fromkeys(p['role'] for p in pairs));rng=np.random.default_rng(105)
    for role in roles:
        pp=[p for p in pairs if p['role']==role and p['identifiable']]
        ids=list(dict.fromkeys(p['input_id'] for p in pp));values=[]
        if len(ids)>=2:
            for _ in range(1000):
                drawn=rng.choice(ids,size=len(ids),replace=True)
                sample=[p['predicted']['cosine'] for i in drawn for p in pp if p['input_id']==i]
                values.append(float(np.median(sample)))
        intervals.append(dict(role=role,input_clusters=len(ids),pairs=len(pp),
            cosine_median_input_cluster_percentile95=(np.quantile(values,[.025,.975]).tolist() if values else None),
            descriptive_only=True,input_clusters_NOT_necessarily_independent_trajectories=True,
            small_cluster_count_limits_interval_reliability=True,gate_unchanged=True))
    usable=sorted([p for p in pairs if p['identifiable']],key=lambda p:p['predicted']['cosine'])
    if usable:
        chosen=[usable[k] for k in (0,len(usable)//2,len(usable)-1)]
        fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
        for ax,p in zip(axes,chosen):
            a,b=[lookup[(p['input_id'],m)] for m in p['mask_pair']]
            for k,color,label in [('truth','#222222','Measured: 3 x N20'),('physics','#D55E00','Physics'),('predicted','#009E73','Sealed hybrid')]:
                ax.plot(range(len(area)),z[k][a]-z[k][b],'.-',lw=.7,ms=3,color=color,label=label)
            ax.axhline(0,color='.75',lw=.5);ax.set_ylabel('Signed change / ADU')
            ax.set_title(p['role']+' | '+p['input_id']+' | '+' / '.join(p['mask_pair'])+
                         ' | cosine %.3f'%p['predicted']['cosine'],fontsize=9)
        axes[0].legend(fontsize=7,ncol=3);axes[-1].set_xlabel('Original fixed 64-region index')
        fig.suptitle(Path(forecast['model']['path']).parent.name+' | identifiable pairs only: worst / median / best',fontsize=10)
        fig.tight_layout();save_figure(fig,output/(prefix+'_response'),180);plt.close(fig)
    if pairs:
        fig,ax=plt.subplots(figsize=(12,4.8))
        ax.bar(range(len(pairs)),[p['SNR'] for p in pairs],
               color=['#0072B2' if p['identifiable'] else '#D55E00' for p in pairs])
        ax.axhline(3,color='.4',ls='--',label='Predeclared SNR = 3')
        ax.set_xticks(range(len(pairs)),[p['input_id']+'/'+p['mask_pair'][0] for p in pairs],rotation=80,fontsize=6)
        ax.set_ylabel('Measured response / frozen TRAIN repeat noise');ax.legend(fontsize=8)
        ax.set_title('All pairs retained: orange = below identification threshold; not deleted',fontsize=10)
        fig.tight_layout();save_figure(fig,output/(prefix+'_all_pair_SNR'),180);plt.close(fig)
    roles_by_key={(r['input_id'],r['mask']):r.get('role','TIME_HOLDOUT') for r in forecast['rows']}
    candidates=[i for i,s in enumerate(samples) if roles_by_key[(s['input_id'],s['mask_name'])]=='TEST_BOTH']
    if not candidates:candidates=list(range(len(samples)))
    errors=np.sqrt(np.mean((z['predicted'][candidates]-z['truth'][candidates])**2,1))
    order=np.argsort(errors);chosen=[candidates[order[k]] for k in (0,len(order)//2,len(order)-1)]
    fig,axes=plt.subplots(3,4,figsize=(11,8));ccd_examples=[]
    for rank,(row,i) in zip(('best','median','worst'),enumerate(chosen)):
        s=samples[i];ccd_examples.append(dict(input=s['input_id'],mask=s['mask_name'],rank=rank))
        for col,name in enumerate(('truth','physics','predicted')):
            axes[row,col].imshow(arrays[name][i],cmap='gray',vmin=10,vmax=20)
            axes[row,col].set_title(s['input_id']+'/'+s['mask_name']+' | '+name,fontsize=8)
        axes[row,3].imshow(arrays['predicted'][i]-arrays['truth'][i],cmap='RdBu_r',vmin=-2,vmax=2)
        axes[row,3].set_title('Hybrid - measured | +/-2 ADU',fontsize=8)
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Sealed forecast | best / median / worst mean error | common [10,20] ADU; no alignment',fontsize=9)
    fig.tight_layout();save_figure(fig,output/(prefix+'_CCD_fixed_scale'),180);plt.close(fig)
    write(output/(prefix+'_intervals.json'),dict(result=ref(folder/'RESULT.json'),intervals=intervals,
        CCD_examples=ccd_examples,
        resampling_unit='input cluster, retaining all mask pairs within input; never RAW frame',
        bootstrap_repetitions=1000,seed=105,no_model_or_threshold_change=True))


def figures(selection,output,sealed_scores):
    from optical_lab.reporting import plotting,save_figure
    output.mkdir();source=output/'research_report.py'
    with source.open('xb') as f:f.write(Path(__file__).read_bytes())
    review=read(selection/'RESULT.json');winner=review['selected_model'];model=ROOT/winner
    plan=read(model/'PLAN.json');result=read(model/'RESULT.json');cache=checked(plan['cache']);data=checked(cache['dataset'])
    arrays=checked(result['predictions'],True);layout=cache['layout'];area=np.array([r['pixels'] for r in layout['regions']])
    z={k:np.stack([integrate(im,layout)/area for im in a]) for k,a in arrays.items() if k in ('truth','physics','predicted')}
    v=checked(data['features'],True)['mean_sums']/area;noise=[]
    for i in plan['train']:
        vv=v[cache['rows'][i]['visits']];noise.extend((vv-vv.mean(0))*np.sqrt(len(vv)/(len(vv)-1)))
    noise=np.asarray(noise);noise_rms=float(np.sqrt(np.mean(noise**2)))
    plt=plotting();roles=('SELECT_INPUT','SELECT_MASK','SELECT_BOTH');fig,axes=plt.subplots(1,3,figsize=(12,3.8));summary=[]
    for j,role in enumerate(roles):
        ids=[i for i,s in enumerate(plan['split']) if s==role];require(ids,'missing SELECT role')
        errors={k:float(np.sqrt(np.mean((z[k][ids]-z['truth'][ids])**2))) for k in ('physics','predicted')}
        summary.append(dict(role=role,conditions=len(ids),**errors,TRAIN_repeat_noise_ADU=noise_rms))
        axes[0].bar(j-.15,errors['physics'],.3,color='#D55E00',label='Physics' if j==0 else None)
        axes[0].bar(j+.15,errors['predicted'],.3,color='#0072B2',label='Hybrid' if j==0 else None)
    axes[0].axhline(noise_rms,color='#222222',ls=':',label='TRAIN repeat noise');axes[0].set_ylabel('Region mean RMS / ADU');axes[0].legend(fontsize=7)
    selected=next(c for c in review['candidates'] if c['model']==winner)
    for axis,key,threshold in ((axes[1],'median_cosine',.7),(axes[2],'median_relative_error',.75)):
        rr=[next(g for g in selected['groups'] if g['role']==r) for r in roles]
        axis.bar(range(3),[g['response'][key] for g in rr],color='#0072B2');axis.axhline(threshold,color='#999999',ls='--')
        axis.set_ylabel('Whitened response cosine' if key=='median_cosine' else 'Relative response error')
    for ax in axes:ax.set_xticks(range(3),['New input','New mask','Both new'])
    fig.suptitle(winner+' | development SELECT, NOT independent TEST',fontsize=10);fig.tight_layout();save_figure(fig,output/'01_mapping_and_response',180);plt.close(fig)
    # Display the worst/median/best SELECT-input mean errors, selected by a
    # declared numerical rule; do not choose hand-picked attractive images.
    ids=[i for i,s in enumerate(plan['split']) if s=='SELECT_BOTH'];err=np.sqrt(np.mean((z['predicted'][ids]-z['truth'][ids])**2,1))
    order=np.argsort(err);chosen=[ids[order[k]] for k in (0,len(order)//2,len(order)-1)]
    fig,axes=plt.subplots(3,4,figsize=(11,8));chosen_rows=[]
    for k,i in enumerate(chosen):
        row=cache['rows'][i];chosen_rows.append(dict(index=i,input=row['input_id'],mask=row['mask'],rank=['best','median','worst'][k]))
        for j,name in enumerate(('truth','physics','predicted')):
            axes[k,j].imshow(arrays[name][i],cmap='gray',vmin=10,vmax=20)
            label={'truth':'Measured: 3 x N20','physics':'Physics mean','predicted':'Hybrid mean'}[name]
            axes[k,j].set_title(row['input_id']+'/'+row['mask']+' | '+label,fontsize=8)
        axes[k,3].imshow(arrays['predicted'][i]-arrays['truth'][i],cmap='RdBu_r',vmin=-2,vmax=2)
        axes[k,3].set_title('Hybrid - measured | ±2 ADU',fontsize=8)
        for ax in axes[k]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('SELECT_BOTH best / median / worst | same [10,20] ADU, no image-wise alignment',fontsize=9)
    fig.tight_layout();save_figure(fig,output/'02_CCD_fixed_scale',180);plt.close(fig)
    # Show all 64 regions and signs, not only a scalar NCC.
    audit=read(ROOT/(winner+'_AUDIT')/'RESULT.json');lookup={(r['input_id'],r['mask']):i for i,r in enumerate(cache['rows'])}
    pp=[p for p in audit['pairs'] if not p['both_conditions_in_TRAIN'] and p['identifiable']]
    pp=sorted(pp,key=lambda p:p['predicted']['cosine']);pick=[pp[i] for i in (0,len(pp)//2,len(pp)-1)]
    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True);response_rows=[]
    for ax,p in zip(axes,pick):
        plus,minus=p['axis'].split(' vs ');a,b=lookup[(p['input_id'],plus)],lookup[(p['input_id'],minus)]
        for name,color in [('truth','#222222'),('physics','#D55E00'),('predicted','#0072B2')]:
            label={'truth':'Measured: 3 x N20','physics':'Physics mean','predicted':'Hybrid mean'}[name]
            ax.plot(range(64),z[name][a]-z[name][b],'.-',lw=.7,ms=3,color=color,label=label)
        ax.axhline(0,color='.75',lw=.5);ax.set_ylabel('Signed change / ADU')
        ax.set_title(p['input_id']+' | '+p['axis']+' | whitened cosine %.3f'%p['predicted']['cosine'],fontsize=9);response_rows.append(p)
    axes[0].legend(fontsize=7,ncol=3);axes[-1].set_xlabel('Original fixed 64-region index');fig.tight_layout();save_figure(fig,output/'03_signed_64_region_response',180);plt.close(fig)
    tests=[]
    for folder in sealed_scores:
        item=read(folder/'RESULT.json');tests.append(dict(reference=ref(folder/'RESULT.json'),
            model=checked(item['forecast'])['model'],metrics=item['metrics']))
    if tests:
        flat=[(Path(t['model']['path']).parent.name,m) for t in tests for m in t['metrics']]
        fig,axes=plt.subplots(1,4,figsize=(15,4.5));names=[m['role']+'\n'+('PASS' if m['passed'] else 'FAIL') for name,m in flat]
        for j,(_,m) in enumerate(flat):
            for offset,name,color in ((-.15,'physics','#D55E00'),(.15,'predicted','#009E73')):
                axes[0].bar(j+offset,m[name]['TRAIN_scaled_RMSE'],.3,color=color,label=name if j==0 else None)
            if m['response']:
                axes[1].bar(j,m['response']['median_cosine'],color='#009E73');axes[2].bar(j,m['response']['median_relative_error'],color='#009E73')
            axes[3].bar(j,m['identifiable_fraction'],color='#009E73' if m['identifiable_fraction']>=.8 else '#D55E00')
        axes[0].set_ylabel('RMSE / frozen TRAIN scale');axes[0].legend(fontsize=7)
        axes[1].set_ylabel('Whitened response cosine');axes[1].axhline(.7,ls='--',color='.6')
        axes[2].set_ylabel('Relative response error');axes[2].axhline(.75,ls='--',color='.6')
        axes[3].set_ylabel('Identifiable fraction (ALL pairs)');axes[3].axhline(.8,ls='--',color='.6');axes[3].set_ylim(0,1.05)
        for ax in axes:ax.set_xticks(range(len(flat)),names,rotation=15,ha='right',fontsize=6)
        fig.suptitle('Pre-capture sealed forecasts | no refit / no post-capture alignment');fig.tight_layout();save_figure(fig,output/'04_frozen_evaluation',180);plt.close(fig)
        for k,folder in enumerate(sealed_scores):sealed_detail(folder,output,'05_sealed_%02d'%k)
    write(output/'RESULT.json',dict(selection=ref(selection/'RESULT.json'),source=ref(source),selected_model=ref(model/'RESULT.json'),
        mean_error_summary=summary,CCD_examples=chosen_rows,response_examples=response_rows,sealed_evaluations=tests,
        empirical_CCD_target='mean of three independent N20 effective frames, not noiseless physical truth',
        no_new_fitting=True,no_hardware=True,SLM_training_or_control_NOT_claimed=True))
    with (output/'mapping_metrics.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+')
    p.add_argument('--selection');p.add_argument('--sealed-scores',nargs='*',default=[])
    p.add_argument('--freeze-static');p.add_argument('--static-evaluation');p.add_argument('--score')
    p.add_argument('--superposition');p.add_argument('--physical');a=p.parse_args()
    require(Path(a.output).name==a.output and sum(map(bool,[a.datasets,a.selection,a.freeze_static,a.static_evaluation,a.superposition]))==1,'explicit single report operation')
    if a.superposition:
        require(a.physical,'explicit frozen physics required');superposition(ROOT/a.superposition,ROOT/a.physical,ROOT/a.output)
    elif a.freeze_static:freeze_static(ROOT/a.freeze_static,ROOT/a.output)
    elif a.static_evaluation:
        require(a.score,'explicit sealed score required');static_evaluation(ROOT/a.static_evaluation,ROOT/a.score,ROOT/a.output)
    elif a.datasets:inventory([ROOT/n for n in a.datasets],ROOT/a.output)
    else:figures(ROOT/a.selection,ROOT/a.output,[ROOT/n for n in a.sealed_scores])
