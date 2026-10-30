"""Training-only shared 64-readout kernel residual over calibrated physics.

Diagnostic alternative to a full CCD residual image. Inputs are actual DMD
pixels and local SLM coefficients, NEVER state labels, camera values or time.
There is no online image-network path and no hardware import/operation.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from workflow import ROOT,read,write,ref,require
from dataset import checked,integrate


def image_features(dmd):
    """24 fixed image moments from the four exact 300-square DMD tiles.

    The last tile is only a fixed coordinate reference, not a supplied state.
    Pixel mass and centered second moments retain binary rendering differences.
    This encoding restricts the surrogate to this four-ball input family.
    """
    require(dmd.shape==(600,600) and set(np.unique(dmd))<={0,255},'actual binary four-tile input required')
    yy,xx=np.mgrid[:300,:300];features=[]
    for y in (0,300):
        for x in (0,300):
            a=dmd[y:y+300,x:x+300].astype(float)/255;mass=a.sum()
            require(mass>0,'empty tile outside moment surrogate domain')
            cx=(a*xx).sum()/mass;cy=(a*yy).sum()/mass
            features.append([cx/300,cy/300,np.log(mass),
                (a*(xx-cx)**2).sum()/mass/300**2,(a*(xx-cx)*(yy-cy)).sum()/mass/300**2,
                (a*(yy-cy)**2).sum()/mass/300**2])
    f=np.asarray(features);centers=f[:,:2].copy();f[:3,:2]-=centers[3]
    return f.ravel()


def kernel(x,c,y,d,length,degree):
    delta=((x[:,None]-y[None])**2).mean(-1)
    return np.exp(-.5*delta/length**2)*(1+c@d.T)**degree


def response_metrics(ztrue,zpred,pairs,rows,white,cov,split):
    result=[]
    for p,m in pairs:
        require(split[p]==split[m],'opposite mask partitions differ')
        delta=ztrue[p]-ztrue[m];a=white@delta;b=white@(zpred[p]-zpred[m])
        snr=np.linalg.norm(delta)/np.sqrt(np.trace(cov)*(1/len(rows[p]['visits'])+1/len(rows[m]['visits'])))
        result.append(dict(input_id=rows[p]['input_id'],role=split[p],masks=[rows[p]['mask'],rows[m]['mask']],
            SNR=float(snr),cosine=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12)),
            amplitude_ratio=float(np.linalg.norm(b)/max(np.linalg.norm(a),1e-12)),
            relative_error=float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-12))))
    return result


def train(cache,output,feature_mode='moments'):
    from residual_fit import pilot_split,signed_pairs
    from response import covariance
    output.mkdir();start=time.monotonic();index=read(cache/'INDEX.json');data=checked(index['dataset'])
    rows=index['rows'];split=pilot_split(rows);train=np.array([i for i,s in enumerate(split) if s=='TRAIN']);select=np.array([i for i,s in enumerate(split) if s!='TRAIN'])
    require({split[i] for i in select}=={'SELECT_INPUT','SELECT_MASK','SELECT_BOTH'},'three SELECT roles required')
    source=output/'SOURCE.py'
    with source.open('xb') as f:f.write(Path(__file__).read_bytes())
    areas=np.array([r['pixels'] for r in index['layout']['regions']]);arrays=checked(index['cache'],True)
    true=np.stack([integrate(im,index['layout'])/areas for im in arrays['truth']])
    physical=np.stack([integrate(im,index['layout'])/areas for im in arrays['physics']])
    require(feature_mode in ('centers','center_mass','moments'),'declared pixel-feature subset required')
    selected_features=[6*i+j for i in range(4) for j in (range(6) if feature_mode=='moments' else range(3) if feature_mode=='center_mass' else range(2))]
    xf=np.stack([image_features(checked(r['dmd_reference'],True))[selected_features] for r in rows]);coef=np.array([r['coefficients_raw_gray'] for r in rows])/24
    require(np.max(np.abs(coef).sum(1))<=1.000001,'local gray envelope exceeded')
    # Unique TRAIN DMDs, rather than repeated masks, determine feature scales.
    unique={rows[i]['dmd_reference']['sha256']:i for i in train};ii=list(unique.values())
    mu=xf[ii].mean(0);sd=xf[ii].std(0).clip(min=1e-6);x=(xf-mu)/sd
    baseline=[i for i in train if rows[i]['mask']=='BASE'];scales=true[baseline].std(0,ddof=1).clip(min=.01)
    measured=checked(data['features'],True)['mean_sums']/areas;noise=[]
    for i in train:
        v=measured[rows[i]['visits']];noise.extend((v-v.mean(0))*np.sqrt(len(v)/(len(v)-1)))
    cov,white=covariance(noise);pairs=signed_pairs(rows);tset=set(train);sset=set(select)
    trainpairs=[(p,m) for p,m in pairs if p in tset and m in tset]
    selectpairs=[(p,m) for p,m in pairs if p in sset and m in sset]
    target=(true[train]-physical[train])/scales
    # Same-input response emphasis acts only on TRAIN row differences.
    G=np.eye(len(train));lookup={int(j):k for k,j in enumerate(train)}
    for p,m in trainpairs:
        u,v=lookup[p],lookup[m];G[u,u]+=1;G[v,v]+=1;G[u,v]-=1;G[v,u]-=1
    plan=dict(cache=ref(cache/'INDEX.json'),source=ref(source),train=train.tolist(),select=select.tolist(),split=split,
        image_features='fixed DMD-pixel moments; no MuJoCo state input',feature_mode=feature_mode,feature_indices=selected_features,
        feature_mu=mu.tolist(),feature_sd=sd.tolist(),
        state_scales_ADU=scales.tolist(),degrees=[1,2],lengths=[.25,.5,1.,2.,4.,8.],ridges=[.0001,.001,.01,.1,1.],
        selection='SELECT scaled mean MSE + .5 TRAIN-whitened relative response MSE',
        response_pair_weight=1.,online_electronic_network_added=False,CCD_image_prediction_NOT_provided=True,
        input_family='same fixed four-ball binary rendering only',formal_TEST=False)
    write(output/'PLAN.json',plan);results=[];best=float('inf');winner=None;best_prediction=None
    for degree in plan['degrees']:
        for length in plan['lengths']:
            K=kernel(x[train],coef[train],x[train],coef[train],length,degree)
            values,vectors=np.linalg.eigh(K);phi=vectors*np.sqrt(values.clip(min=0))[None]
            lhs=phi.T@G@phi;rhs=phi.T@G@target
            for ridge in plan['ridges']:
                beta=np.linalg.solve(lhs+ridge*np.eye(len(train)),rhs)
                alpha=vectors@((vectors.T@(phi@beta))/np.maximum(values,1e-10)[:,None])
                pred=physical+kernel(x,coef,x[train],coef[train],length,degree)@alpha*scales
                mean=float(np.mean(((pred[select]-true[select])/scales)**2));errors=[]
                for p,m in selectpairs:
                    a=white@(true[p]-true[m]);b=white@(pred[p]-pred[m]-true[p]+true[m])
                    floor=64*(1/len(rows[p]['visits'])+1/len(rows[m]['visits']))
                    errors.append(float(b@b/max(a@a,floor)))
                score=mean+.5*np.mean(errors)
                item=dict(degree=degree,length=length,ridge=ridge,SELECT_mean_MSE=mean,SELECT_response_MSE=float(np.mean(errors)),score=float(score))
                results.append(item)
                if score<best:
                    best=score;winner=dict(item,alpha=alpha.copy());best_prediction=pred.copy()
            print('READOUT KERNEL',degree,length,'best SELECT',round(best,6),flush=True)
    with (output/'MODEL.npz').open('xb') as f:np.savez_compressed(f,alpha=winner.pop('alpha'),train_features=x[train],train_coefficients=coef[train],
        feature_mu=mu,feature_sd=sd,state_scales=scales,TRAIN_covariance=cov,TRAIN_whitener=white)
    pp=response_metrics(true,best_prediction,pairs,rows,white,cov,split);metrics=[]
    for role in dict.fromkeys(split):
        ids=[i for i,s in enumerate(split) if s==role];rr=[p for p in pp if p['role']==role];good=[p for p in rr if p['SNR']>=3]
        rawerr=best_prediction[ids]-true[ids];hybrid=float(np.sqrt(np.mean((rawerr/scales)**2)))
        physics=float(np.sqrt(np.mean(((physical[ids]-true[ids])/scales)**2)))
        summary=None if not good else dict(coverage=len(good)/len(rr),median_cosine=float(np.median([p['cosine'] for p in good])),
            direction_fraction=float(np.mean([p['cosine']>0 for p in good])),amplitude_fraction_05_20=float(np.mean([.5<=p['amplitude_ratio']<=2 for p in good])),
            median_relative_error=float(np.median([p['relative_error'] for p in good])))
        passed=bool(summary and summary['coverage']>=.8 and summary['median_cosine']>=.7 and summary['direction_fraction']>=.8 and
            summary['amplitude_fraction_05_20']>=.8 and summary['median_relative_error']<=.75 and 1-hybrid/physics>=.2)
        metrics.append(dict(role=role,conditions=len(ids),region_RMS_ADU=float(np.sqrt(np.mean(rawerr**2))),
            TRAIN_scaled_RMSE=hybrid,physics_TRAIN_scaled_RMSE=physics,improvement_fraction=1-hybrid/physics,response=summary,passed=passed))
    with (output/'PREDICTIONS.npz').open('xb') as f:np.savez_compressed(f,predicted=best_prediction,truth=true,physics=physical)
    write(output/'GRID.json',results)
    write(output/'RESULT.json',dict(selected=winner,metrics=metrics,response_pairs=pp,model=ref(output/'MODEL.npz'),
        predictions=ref(output/'PREDICTIONS.npz'),source=ref(source),seconds=time.monotonic()-start,
        SELECT_gate_passed=all(m['passed'] for m in metrics if m['role']!='TRAIN'),
        dual_residual_coefficients=len(train)*64,full_CCD_image_NOT_predicted=True,
        SLM_training_allowed=False,formal_TEST_NOT_done=True,time_holdout_NOT_done=True,no_hardware=True))
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(1,3,figsize=(11,3.8));xx=np.arange(len(metrics))
    axes[0].bar(xx-.18,[m['physics_TRAIN_scaled_RMSE'] for m in metrics],.36,label='physics',color='#D55E00')
    axes[0].bar(xx+.18,[m['TRAIN_scaled_RMSE'] for m in metrics],.36,label='readout residual',color='#0072B2');axes[0].legend(fontsize=7)
    axes[1].bar(xx,[m['response']['median_cosine'] if m['response'] else np.nan for m in metrics],color='#0072B2');axes[1].axhline(.7,color='gray',ls='--')
    axes[2].bar(xx,[m['response']['median_relative_error'] if m['response'] else np.nan for m in metrics],color='#0072B2');axes[2].axhline(.75,color='gray',ls='--')
    for ax,title in zip(axes,('TRAIN-scaled region RMSE','Response cosine','Relative response error')):
        ax.set_ylabel(title);ax.set_xticks(xx,[m['role'] for m in metrics],rotation=25,ha='right',fontsize=7)
    fig.suptitle('Offline 64-readout surrogate | no online image network; development only')
    fig.tight_layout();save_figure(fig,output/'01_readout_model',180);plt.close(fig)
    print('READOUT KERNEL RESULT',winner,metrics,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--features',choices=('centers','center_mass','moments'),default='moments');a=p.parse_args()
    require(Path(a.output).name==a.output,'fresh explicit output required');train(ROOT/a.cache,ROOT/a.output,a.features)
