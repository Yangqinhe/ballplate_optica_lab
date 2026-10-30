"""Effective N20 readout noise, separate from conditional mean errors.

Three nonnegative variance coefficients + one TRAIN correlation matrix. This
model is offline only. It never adds an electronic image controller or changes
the already declared primary response-whitening acceptance metric.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from workflow import ROOT,read,write,ref,require
from dataset import checked,integrate
from residual_fit import pilot_split
from response import covariance


def variance_design(mean,areas,background):
    signal=np.maximum(mean-background,0.)
    return np.stack((np.ones_like(mean),np.broadcast_to(1/areas,mean.shape),signal**2),-1)


def evaluate(model,score,output):
    """Fixed TRAIN noise parameters on fresh queries; never hide mean bias."""
    from fit import groups
    output.mkdir();cfg=read(model/'MODEL.json');result=read(score/'RESULT.json');forecast=checked(result['forecast'])
    data=checked(result['dataset']);samples=groups(data);train=checked(cfg['cache'])
    require(data['baseline']==checked(train['dataset'])['baseline'] and data['layout']==train['layout'],'noise platform changed')
    frozen_ns=read(model/'FREEZE.json')['created_unix_ns']
    require(all(r['host_times_ns'][0]>frozen_ns for r in data['rows']),'noise model not frozen before capture')
    noise=checked(cfg['TRAIN_correlation'],True);areas=np.array([r['pixels'] for r in data['layout']['regions']])
    zz=checked(data['features'],True)['mean_sums']/areas;arrays=np.load(score/'PREDICTIONS.npz',allow_pickle=False)
    means=np.stack([integrate(im,data['layout'])/areas for im in arrays['predicted']])
    variance=(variance_design(means,areas,cfg['background_ADU'])@noise['variance_coefficients']).clip(min=1e-10)
    variance*=noise['TRAIN_standardized_region_SD'][None,:]**2
    mapping={(r['input_id'],r['dmd_reference']['sha256'],r['mask_reference']['sha256']):r.get('role','TIME_HOLDOUT') for r in forecast['rows']}
    details=[]
    for i,s in enumerate(samples):
        v=zz[s['visits']];r=(v-v.mean(0))*np.sqrt(len(v)/(len(v)-1))
        role=mapping[(s['input_id'],s['label']['dmd']['sha256'],s['mask']['sha256'])]
        for name in ('stationary','conditional_proxy'):
            c=(noise['stationary_covariance'] if name=='stationary' else
               noise['TRAIN_correlation']*np.sqrt(variance[i,:,None]*variance[i,None,:]))+np.eye(len(areas))*1e-10
            sign,logdet=np.linalg.slogdet(c);require(sign>0,'fixed noise covariance not positive definite')
            q=float(np.mean(np.sum((r@np.linalg.inv(c))*r,axis=1))/len(areas))
            details.append(dict(input_id=s['input_id'],mask=s['mask_name'],role=role,model=name,
                centered_NLL_per_region=.5*(logdet/len(areas)+q+np.log(2*np.pi)),whitened_energy_per_region=q,
                measured_over_predicted_RMS=float(np.sqrt(np.mean(r*r))/np.sqrt(np.trace(c)/len(areas)))))
    summary=[]
    for role in dict.fromkeys(d['role'] for d in details):
        for name in ('stationary','conditional_proxy'):
            selected=[d for d in details if d['role']==role and d['model']==name]
            summary.append(dict(role=role,model=name,conditions=len(selected),
                mean_centered_NLL_per_region=float(np.mean([d['centered_NLL_per_region'] for d in selected])),
                mean_whitened_energy_per_region=float(np.mean([d['whitened_energy_per_region'] for d in selected])),
                median_measured_over_predicted_RMS=float(np.median([d['measured_over_predicted_RMS'] for d in selected]))))
    write(output/'RESULT.json',dict(model=ref(model/'MODEL.json'),freeze=ref(model/'FREEZE.json'),score=ref(score/'RESULT.json'),
        source=ref(__file__),summary=summary,details=details,TRAIN_parameters_unchanged=True,
        deterministic_mean_error_NOT_included_in_noise=True,between_condition_time_drift_NOT_estimated=True,
        acceptance_gates_unchanged=True,Gaussian_assumption_NOT_verified=True,control_qualified=False))
    print('FROZEN NOISE EVALUATION',summary,flush=True)


def fit(cache,output,model=None):
    output.mkdir();index=read(cache/'INDEX.json');rows=index['rows'];labels=pilot_split(rows)
    train=np.array([s=='TRAIN' for s in labels]);require(train.any() and (~train).any(),'TRAIN/SELECT required; TEST forbidden')
    data=checked(index['dataset']);areas=np.array([r['pixels'] for r in index['layout']['regions']]);zz=checked(data['features'],True)['mean_sums']/areas
    cfg=checked(index['physical_model']);background=cfg['photometry'][1]+3*np.tanh(cfg['q'][12])
    if cfg.get('extra') is not None:background+=.1*np.tanh(cfg['extra'][9])
    means=[];residuals=[];condition_variance=[]
    for row in rows:
        v=zz[row['visits']];require(len(v)>=3,'three independent effective frames required')
        r=(v-v.mean(0))*np.sqrt(len(v)/(len(v)-1))
        means.append(v.mean(0));residuals.append(r);condition_variance.append((r*r).mean(0))
    means=np.array(means);target=np.array(condition_variance);noise=np.concatenate([v for v,t in zip(residuals,train) if t])
    stationary,_=covariance(noise);diag=np.diag(stationary).clip(min=1e-10)
    features=variance_design(means,areas,background)
    x=(features[train]/diag[None,:,None]).reshape(-1,3);y=(target[train]/diag).ravel()
    norm=np.linalg.norm(x,axis=0).clip(min=1e-12);x=x/norm;weights=np.ones(len(y))
    for _ in range(5):
        co,_=nnls(x*np.sqrt(weights[:,None]),y*np.sqrt(weights));err=x@co-y
        weights=np.minimum(1.,1.5/np.maximum(abs(err),1e-12))
    coef=co/norm;variance=(features@coef).clip(min=1e-10)
    standardized=np.concatenate([r/np.sqrt(v) for r,v,t in zip(residuals,variance,train) if t])
    corr,_=covariance(standardized);scale=np.sqrt(np.diag(corr));corr=corr/scale[:,None]/scale[None,:]
    # Fixed correlation is estimated only from TRAIN; its diagonal is exactly1.
    predmeans=None
    if model is not None:
        plan=read(model/'PLAN.json');require(plan['cache']==ref(cache/'INDEX.json'),'proxy/noise cache differs')
        pred=checked(read(model/'RESULT.json')['predictions'],True)['predicted']
        predmeans=np.stack([integrate(a,index['layout'])/areas for a in pred])
    predictions={'stationary':np.tile(diag,(len(rows),1)),'conditional_measured_mean_diagnostic':variance,
        'conditional_TRAIN_calibrated_measured_mean':variance*scale[None,:]**2}
    if predmeans is not None:
        pv=(variance_design(predmeans,areas,background)@coef).clip(min=1e-10)
        predictions['conditional_proxy_mean']=pv
        predictions['conditional_TRAIN_calibrated_proxy_mean']=pv*scale[None,:]**2
    summary=[];detail=[]
    for name,var in predictions.items():
        per=[]
        for i,(r,v) in enumerate(zip(residuals,var)):
            c=stationary if name=='stationary' else corr*np.sqrt(v[:,None]*v[None,:])
            c=c+np.eye(64)*1e-10;sign,logdet=np.linalg.slogdet(c);require(sign>0,'noise covariance not positive definite')
            q=float(np.mean(np.sum((r@np.linalg.inv(c))*r,axis=1))/64)
            nll=.5*(logdet/64+q+np.log(2*np.pi))
            per.append(dict(input_id=rows[i]['input_id'],mask=rows[i]['mask'],split=labels[i],
                centered_residual_Gaussian_NLL_per_region=float(nll),whitened_energy_per_region=q,
                measured_region_RMS_ADU=float(np.sqrt(np.mean(target[i]))),predicted_region_RMS_ADU=float(np.sqrt(np.mean(v)))))
        for role in dict.fromkeys(labels):
            rr=[r for r in per if r['split']==role]
            summary.append(dict(model=name,split=role,conditions=len(rr),
                centered_residual_Gaussian_NLL_per_region=float(np.mean([r['centered_residual_Gaussian_NLL_per_region'] for r in rr])),
                mean_whitened_energy_per_region=float(np.mean([r['whitened_energy_per_region'] for r in rr])),
                median_measured_over_predicted_RMS=float(np.median([r['measured_region_RMS_ADU']/r['predicted_region_RMS_ADU'] for r in rr]))))
        detail.append(dict(model=name,conditions=per))
    with (output/'NOISE.npz').open('xb') as f:np.savez_compressed(f,stationary_covariance=stationary,
        TRAIN_correlation=corr,TRAIN_standardized_region_SD=scale,variance_coefficients=coef,condition_mean_ADU=means,condition_variance_ADU2=target)
    write(output/'MODEL.json',dict(cache=ref(cache/'INDEX.json'),source=ref(__file__),variance_coefficients=coef.tolist(),
        equation='variance = c0 + c1 / region_pixel_area + c2 * max(region_mean_ADU - background_ADU, 0)^2',
        background_ADU=float(background),filter='independent N20 RAW mean',TRAIN_correlation=ref(output/'NOISE.npz'),
        conditional_calibrated_variant_uses_TRAIN_only_64_region_variance_scale=scale.tolist(),
        fast_and_between_visit_changes_not_uniquely_separated=True,slow_time_holdout_pending=True,
        conditional_measured_mean_is_diagnostic_NOT_a_deployable_prediction=True,
        primary_response_whitener_unchanged=True,no_online_network=True,control_qualified=False))
    write(output/'RESULT.json',dict(model=ref(output/'MODEL.json'),summary=summary,details=detail,
        source=ref(__file__),formal_TEST=False,mean_mapping_residuals_NOT_used_as_noise=True,
        effective_frames_not_independent_states=True,Gaussian_diagnostic_NOT_proof_of_Gaussian_noise=True))
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,ax=plt.subplots(figsize=(9,4));roles=list(dict.fromkeys(labels));names=list(predictions)
    for j,name in enumerate(names):
        rr=[next(r for r in summary if r['model']==name and r['split']==s) for s in roles]
        ax.bar(np.arange(len(roles))+(j-(len(names)-1)/2)*.22,[r['mean_whitened_energy_per_region'] for r in rr],.22,label=name)
    ax.axhline(1,color='#999999',ls='--');ax.set_xticks(np.arange(len(roles)),roles);ax.set_ylabel('Centered residual whitened energy / region')
    ax.legend(fontsize=7);fig.tight_layout();save_figure(fig,output/'01_noise_calibration',180);plt.close(fig)
    print('NOISE MODEL',coef.tolist(),summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache');p.add_argument('--output',required=True);p.add_argument('--model')
    p.add_argument('--evaluate');p.add_argument('--score');a=p.parse_args()
    require(Path(a.output).name==a.output,'fresh explicit output required')
    if a.evaluate:
        require(a.score,'explicit frozen score required');evaluate(ROOT/a.evaluate,ROOT/a.score,ROOT/a.output)
    else:
        require(a.cache,'explicit TRAIN/SELECT cache required');fit(ROOT/a.cache,ROOT/a.output,None if a.model is None else ROOT/a.model)
