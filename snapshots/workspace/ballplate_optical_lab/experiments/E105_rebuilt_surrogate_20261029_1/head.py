"""Frozen-mask 64->2 baseline: file-only fitting, no optical surrogate online."""
import argparse
from pathlib import Path
import numpy as np
from workflow import ROOT,read,write,ref,require
from dataset import checked


def records(dataset):
    idx=read(dataset/'INDEX.json');allz=checked(idx['features'],True)['mean_sums']
    require(len(idx['layout']['regions'])==64,'frozen 64-region layout required')
    areas=np.array([r['pixels'] for r in idx['layout']['regions']]);rows=[];values=[]
    baseline=checked(idx['baseline']);base_sha=baseline['files']['slm']['sha256']
    for i,r in enumerate(idx['rows']):
        if r['mask']['sha256']!=base_sha:continue
        require(r['n']==20,'head requires exactly independent N20 RAW means')
        label=r['label'];pos=np.asarray(label['position_xy_m']);vel=np.asarray(label['velocity_xy_m_s'])
        target=np.array([6*pos[1]+2.5*vel[1],-6*pos[0]-2.5*vel[0]])
        require(np.allclose(target,label['teacher_action_rad'],rtol=0,atol=1e-12),'frozen teacher differs')
        rows.append(dict(input_id=r['input_id'],trajectory=label['trajectory'],
            split=label.get('new_platform_split','PILOT'),target=target.tolist(),source_index=i))
        values.append(allz[i]/areas)
    require(len(rows)>0,'BASE-mask RAW required')
    return idx,np.array(values),rows,areas


def solve(x,y,names,ridge,eta,background_null=False):
    names=np.asarray(names);groups=list(dict.fromkeys(names));means=np.stack([x[names==g].mean(0) for g in groups])
    targets=np.stack([y[names==g].mean(0) for g in groups]);residual=[]
    for g,mean in zip(groups,means):
        v=x[names==g];require(len(v)>=2,'repeated TRAIN observations required')
        residual.extend((v-mean)*np.sqrt(len(v)/(len(v)-1)))
    residual=np.array(residual);mu=means.mean(0);sd=np.maximum(means.std(0),np.maximum(residual.std(0),.001))
    ym=targets.mean(0);z=(means-mu)/sd;nr=residual/sd
    q=np.eye(64)
    if background_null:
        _,_,vh=np.linalg.svd((1/sd)[None],full_matrices=True);q=vh[1:].T
    zz=z@q;rr=nr@q
    a=zz.T@zz/len(z)+ridge*np.eye(q.shape[1])+eta*(rr.T@rr/len(rr))
    coef=q@np.linalg.solve(a,zz.T@(targets-ym)/len(z));w=coef/sd[:,None];b=ym-mu@w
    require(np.isfinite(w).all() and np.isfinite(b).all(),'nonfinite head')
    return dict(weights_ADU=w,bias=b,mean=mu,scale=sd)


def score(pred,y,rows):
    names=np.array([r['input_id'] for r in rows]);groups=list(dict.fromkeys(names));errors=[];noise=[]
    for name in groups:
        ids=names==name;p=pred[ids];t=y[ids]
        errors.append((p.mean(0)-t.mean(0)))
        if len(p)>1:noise.extend((p-p.mean(0))*np.sqrt(len(p)/(len(p)-1)))
    errors=np.array(errors)
    return dict(inputs=len(groups),MAE_rad=float(np.abs(errors).mean()),
        RMS_L2_rad=float(np.sqrt(np.mean(np.sum(errors**2,axis=1)))),
        p95_L2_rad=float(np.quantile(np.linalg.norm(errors,axis=1),.95)),
        max_L2_rad=float(np.linalg.norm(errors,axis=1).max()),
        repeat_RMS_L2_rad=float(np.sqrt(np.mean(np.sum(np.array(noise)**2,axis=1)))) if noise else None)


def frozen_predict(head,mean_ADU,areas):
    w=np.asarray(head['weights_sum'],dtype=np.float64);b=np.asarray(head['bias'],dtype=np.float64)
    areas=np.asarray(areas,dtype=np.float64);x=np.asarray(mean_ADU,dtype=np.float64)
    require(head['parameters']==130 and w.shape==(64,2) and b.shape==(2,),'fixed affine budget differs')
    require(areas.shape==(64,) and (areas>0).all() and x.ndim==2 and x.shape[1]==64,'readout dimensions differ')
    require(np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(b).all(),'nonfinite frozen head')
    return (x*areas)@w+b


def fit(dataset,output):
    output.mkdir();idx,x,rows,areas=records(dataset)
    require(not any(r.get('role','').startswith('TEST') or r['label'].get('new_platform_split')=='TEST'
                    for r in idx['rows']),'TEST dataset cannot enter head fitting')
    require(not any(r['split']=='TEST' for r in rows),'TEST forbidden in head fitting/selection')
    y=np.array([r['target'] for r in rows]);names=np.array([r['input_id'] for r in rows]);traj=np.array([r['trajectory'] for r in rows])
    tr=np.array([r['split'] in ('TRAIN','PILOT') for r in rows]);se=np.array([r['split']=='SELECT' for r in rows])
    require(len(set(names[tr]))>=16,'joint TRAIN coverage needed')
    require(not(set(traj[tr])&set(traj[se])),'trajectory leakage')
    grid=[(ridge,eta,bg) for ridge in (1e-4,.001,.01,.1,1.,10.) for eta in (0.,4.,16.) for bg in (False,True)]
    write(output/'PLAN.json',dict(dataset=ref(dataset/'INDEX.json'),source=ref(__file__),grid=grid,
        head_parameters=130,input='64 original region means of N20 RAW; equivalent scaled sums',
        no_gamma=True,no_black_subtraction=True,teacher='u=[6*y+2.5*vy, -6*x-2.5*vx], unclipped training labels',
        control_limits_NOT_applied_in_mapping_scores=True,selector='SELECT input-equal MSE if present, else TRAIN leave-trajectory-out',
        no_optical_surrogate_in_inference=True,control_qualified=False))
    (output/'SOURCE').mkdir()
    with (output/'SOURCE/head.py').open('xb') as f:f.write(Path(__file__).read_bytes())
    candidates=[];validation=[]
    for ridge,eta,bg in grid:
        if se.any():
            m=solve(x[tr],y[tr],names[tr],ridge,eta,bg);p=x[se]@m['weights_ADU']+m['bias'];rr=[r for r,k in zip(rows,se) if k]
            metrics=score(p,y[se],rr)
        else:
            predictions=np.full_like(y,np.nan)
            for group in dict.fromkeys(traj[tr]):
                held=tr&(traj==group);keep=tr&~held
                m=solve(x[keep],y[keep],names[keep],ridge,eta,bg)
                predictions[held]=x[held]@m['weights_ADU']+m['bias']
            p=predictions[tr];metrics=score(p,y[tr],[r for r,k in zip(rows,tr) if k])
        candidates.append(dict(ridge=ridge,noise_weight=eta,background_null=bg,validation=metrics))
        validation.append(p)
    best=min(range(len(candidates)),key=lambda i:candidates[i]['validation']['RMS_L2_rad'])
    c=candidates[best];m=solve(x[tr],y[tr],names[tr],c['ridge'],c['noise_weight'],c['background_null'])
    w=m['weights_ADU'];b=m['bias'];pred=x@w+b
    write(output/'HEAD.json',dict(weights_sum=(w/areas[:,None]).tolist(),bias=b.tolist(),parameters=130,
        source=ref(__file__),dataset=ref(dataset/'INDEX.json'),baseline=idx['baseline'],layout=idx['layout'],
        sample_N=20,filter='independent mean',preprocess='RAW, no ADU subtraction, no gamma',
        selection=c,qualification='offline mapping only; no control qualification',formal_TEST_evaluated=False))
    write(output/'RESULT.json',dict(choice=c,candidates=candidates,head=ref(output/'HEAD.json'),
        train=score(pred[tr],y[tr],[r for r,k in zip(rows,tr) if k]),
        selection_scope='SELECT' if se.any() else 'TRAIN cross-validation; not independent validation',
        background_001ADU_action_L2_rad=float(np.linalg.norm(.01*np.ones(64)@w)),
        raw_gain_1pct_diagnostic_L2_rad=float(np.sqrt(np.mean(np.sum((.01*x[tr]@w)**2,axis=1)))),
        gain_stress_NOT_claimed_to_equal_laser_power_change=True,formal_TEST=False,control_qualified=False))
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,all_fit_prediction=pred,target=y,
        validation_prediction=validation[best],validation_target=y[se] if se.any() else y[tr])
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(8,4));t=y[se] if se.any() else y[tr];p=validation[best]
    for j,ax in enumerate(axes):
        ax.scatter(t[:,j],p[:,j],s=8,c='#0072B2');ax.plot([-1,1],[-1,1],color='#999999',ls='--')
        ax.set_xlabel('Teacher command / rad');ax.set_ylabel('Head command / rad');ax.set_title(('Roll','Pitch')[j])
    fig.suptitle('BASE real RAW head | '+('SELECT' if se.any() else 'TRAIN trajectory CV, not TEST'))
    fig.tight_layout();save_figure(fig,output/'01_head_mapping',180);plt.close(fig)
    print('REAL RAW BASE HEAD',c,flush=True)


def evaluate(model,dataset,output):
    """Unchanged head on trajectory-held TEST. Labels only score predictions."""
    output.mkdir();result=read(model/'RESULT.json');head=checked(result['head'])
    from fit import archive_sources
    archive_sources(output)
    idx,x,rows,areas=records(dataset)
    require(idx['baseline']==head['baseline'] and idx['layout']==head['layout'],'TEST hardware/readout contract changed')
    require(all(r['split']=='TEST' for r in rows),'frozen evaluation requires TEST inputs only')
    train=checked(head['dataset']);train_names={r['input_id'] for r in train['rows']}
    train_trajectories={r['label']['trajectory'] for r in train['rows']}
    require(not ({r['input_id'] for r in rows}&train_names),'TEST input was seen during fit/selection')
    require(not ({r['trajectory'] for r in rows}&train_trajectories),'TEST trajectory was seen during fit/selection')
    require(head['sample_N']==20 and head['filter']=='independent mean' and
            head['preprocess']=='RAW, no ADU subtraction, no gamma','unsupported frozen preprocessing')
    p=frozen_predict(head,x,areas);y=np.array([r['target'] for r in rows])
    write(output/'RESULT.json',dict(model=result['head'],dataset=ref(dataset/'INDEX.json'),
        source=ref(__file__),formal_TEST=True,model_changed=False,
        metrics=score(p,y,rows),inputs=rows,head_parameters=130,
        commands_unclipped=True,closed_loop_NOT_tested=True,control_qualified=False))
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,prediction=p,target=y)
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(8,4))
    for j,ax in enumerate(axes):
        ax.scatter(y[:,j],p[:,j],s=10,c='#0072B2')
        ax.plot([-1,1],[-1,1],color='#999999',ls='--')
        ax.set_xlabel('Teacher command / rad');ax.set_ylabel('Frozen head command / rad')
        ax.set_title(('Roll','Pitch')[j])
    fig.suptitle('Original SLM + 130-parameter head | unseen TEST trajectories')
    fig.tight_layout();save_figure(fig,output/'01_frozen_head_TEST',180);plt.close(fig)
    print('FROZEN REAL HEAD TEST',score(p,y,rows),flush=True)


def diagnostic(dataset,cache,output):
    """TRAIN-only distinguish repeat noise from affine mapping/generalization."""
    from dataset import integrate
    output.mkdir();idx,x,rows,areas=records(dataset);ci=read(cache/'INDEX.json')
    from fit import archive_sources
    archive_sources(output)
    require(ci['dataset']==ref(dataset/'INDEX.json'),'counterfactual inputs differ')
    require(all(r['split'] in ('TRAIN','PILOT') for r in rows),'TRAIN-only diagnostic required')
    physical=checked(ci['cache'],True)['physics'];mapping={}
    for i,row in enumerate(ci['rows']):
        if row['mask']=='BASE':
            v=integrate(physical[i],idx['layout'])/areas
            for visit in row['visits']:mapping[visit]=v
    xp=np.stack([mapping[r['source_index']] for r in rows]);xm=x.copy()
    y=np.array([r['target'] for r in rows]);names=np.array([r['input_id'] for r in rows]);traj=np.array([r['trajectory'] for r in rows])
    for name in dict.fromkeys(names):xm[names==name]=x[names==name].mean(0)
    results=[];predictions={}
    grid=[(ridge,eta,bg) for ridge in (1e-4,.001,.01,.1,1.,10.) for eta in (0.,4.,16.) for bg in (False,True)]
    write(output/'PLAN.json',dict(dataset=ref(dataset/'INDEX.json'),cache=ref(cache/'INDEX.json'),source=ref(__file__),grid=grid,
        labels_USED_FOR_OFFLINE_CV_ONLY=True,no_head_export=True,no_TEST=True,
        calibrated_physics_NOT_ideal_truth=True,
        empirical_input_mean_uses_repeated_held_measurements=True,
        empirical_mean_ablation_removes_noise_covariance_regularization=True,
        not_a_performance_upper_bound=True,not_an_online_filter=True))
    for kind,features in [('measured_effective',x),('measured_input_mean',xm),('calibrated_physics',xp)]:
        candidates=[];best=None
        for ridge,eta,bg in grid:
            pred=np.full_like(y,np.nan)
            for group in dict.fromkeys(traj):
                held=traj==group;keep=~held
                m=solve(features[keep],y[keep],names[keep],ridge,eta,bg)
                pred[held]=features[held]@m['weights_ADU']+m['bias']
            metrics=score(pred,y,rows);item=dict(ridge=ridge,noise_weight=eta,background_null=bg,metrics=metrics)
            candidates.append(item)
            if best is None or metrics['RMS_L2_rad']<best['metrics']['RMS_L2_rad']:
                best=item;predictions[kind]=pred.copy()
        results.append(dict(kind=kind,best=best,candidates=candidates))
    write(output/'RESULT.json',dict(dataset=ref(dataset/'INDEX.json'),results=results,
        TRAIN_trajectory_CV_with_hyperparameter_selection=True,independent_validation=False,
        no_head_export=True,no_online_algorithm_change=True))
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,target=y,**predictions)
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(8,3.5));xx=np.arange(len(results))
    for ax,key,title in zip(axes,('RMS_L2_rad','repeat_RMS_L2_rad'),('Input-mean command error','Repeat command fluctuation')):
        ax.bar(xx,[r['best']['metrics'][key] for r in results],color=['#222222','#009E73','#D55E00'])
        ax.set_xticks(xx,[r['kind'] for r in results],rotation=20,ha='right',fontsize=7)
        ax.set_ylabel('Two-axis RMS / rad');ax.set_title(title)
    fig.suptitle('TRAIN trajectory CV | diagnostic, not deployment qualification')
    fig.tight_layout();save_figure(fig,output/'01_noise_vs_mapping',180);plt.close(fig)
    print('COUNTERFACTUAL HEAD DIAGNOSTIC',[(r['kind'],r['best']) for r in results],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--evaluate');p.add_argument('--diagnostic-cache');a=p.parse_args()
    require(Path(a.output).name==a.output and Path(a.dataset).name==a.dataset,'explicit direct folders required')
    require(not(a.evaluate and a.diagnostic_cache),'choose one file-only mode')
    if a.diagnostic_cache:
        require(Path(a.diagnostic_cache).name==a.diagnostic_cache,'explicit cache required')
        diagnostic(ROOT/a.dataset,ROOT/a.diagnostic_cache,ROOT/a.output)
    elif a.evaluate:
        require(Path(a.evaluate).name==a.evaluate,'explicit model folder required')
        evaluate(ROOT/a.evaluate,ROOT/a.dataset,ROOT/a.output)
    else:fit(ROOT/a.dataset,ROOT/a.output)
