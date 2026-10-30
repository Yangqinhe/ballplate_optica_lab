"""Offline separate 130-parameter heads; same TRAIN/SELECT, no new RAW."""
import argparse
import sys
import numpy as np
import core as c


def run(out):
    out.mkdir()
    sys.path.insert(0,str(c.MODEL))
    old=c.load(c.MODEL/'head.py','e107_head_reference')
    idx,x,rows,areas=old.records(c.MODEL/'BASE_HEAD_SELECT_AUDIT_01')
    manifest=c.read(c.PREVIEW/'MODEL_01/MODEL.json')
    predictor=c.load(c.PREVIEW/'model.py','e107_frozen_mean')
    engine=predictor.Engine(manifest)
    baseline=c.read(c.checked(manifest['baseline'])); mask=np.load(c.checked(baseline['files']['slm']))
    context=c.context_fixture(manifest); by_input={}; predictions={}
    for r in rows:
        name=r['input_id']
        if name in by_input: continue
        source=idx['rows'][r['source_index']]['label']['dmd']
        images,meta=engine.predict(np.load(c.checked(source)),mask,context)
        by_input[name]=c.integrate(images['corrected'],idx['layout'])/areas
        predictions[name]=dict(source=source,metadata=meta)
        print('MODEL',name,flush=True)
    xm=np.stack([by_input[r['input_id']] for r in rows])
    y=np.array([r['target'] for r in rows]); names=np.array([r['input_id'] for r in rows])
    tr=np.array([r['split'] in ('TRAIN','PILOT') for r in rows]); se=np.array([r['split']=='SELECT' for r in rows])
    if not tr.any() or not se.any(): raise ValueError('TRAIN and SELECT required')
    if {r['trajectory'] for r,k in zip(rows,tr) if k} & {r['trajectory'] for r,k in zip(rows,se) if k}:
        raise ValueError('trajectory leakage')
    report={}; refs={}
    for name,features in (('simulation',xm),('real',x)):
        candidates=[]
        for ridge in (1e-4,.001,.01,.1,1.,10.):
            m=old.solve(features[tr],y[tr],names[tr],ridge,0.,False)
            pred=features[se]@m['weights_ADU']+m['bias']
            metrics=old.score(pred,y[se],[r for r,k in zip(rows,se) if k])
            candidates.append((metrics['RMS_L2_rad'],ridge,m,metrics))
        _,ridge,m,metrics=min(candidates,key=lambda a:a[0])
        head=dict(parameters=130,weights_sum=(m['weights_ADU']/areas[:,None]).tolist(),bias=m['bias'].tolist(),
            layout=idx['layout'],sample_N=20,filter='mean',readout='float RAW sums; black=0 gamma=1 software_gain=1',
            fit='ordinary ridge; no extra robustness terms',ridge=ridge,domain=name,
            dataset=c.ref(c.MODEL/'BASE_HEAD_SELECT_AUDIT_01/INDEX.json'),mask=baseline['files']['slm'],
            qualification='diagnostic only; NO closed-loop qualification')
        # Keep an already fitted, same-budget head if ordinary ridge degrades
        # the existing SELECT comparison. This is reuse, not a new holdout.
        reuse=None
        if name=='real':
            prior_path=c.MODEL/'BASE_HEAD_SELECT_01/HEAD.json';prior=c.read(prior_path)
            if prior['dataset']!=head['dataset'] or prior['layout']!=head['layout']:
                raise ValueError('prior head uses a different dataset/layout')
            prior_prediction=old.frozen_predict(prior,features,areas)
            prior_score=old.score(prior_prediction[se],y[se],[r for r,k in zip(rows,se) if k])
            if prior_score['RMS_L2_rad']<metrics['RMS_L2_rad']:
                reuse=dict(reference=c.ref(prior_path),select=prior_score,
                    note='reuse already SELECT-chosen fixed 130-parameter head; no new online layer')
                head=prior;head['domain']='real';head['mask']=baseline['files']['slm'];head['reused_from']=reuse['reference']
        path=out/(name+'_HEAD.json'); c.write(path,head); refs[name]=c.ref(path)
        report[name]=dict(select=metrics,train=old.score(features[tr]@m['weights_ADU']+m['bias'],y[tr],
            [r for r,k in zip(rows,tr) if k]),ridge=ridge,candidates=[dict(ridge=a[1],select=a[3]) for a in candidates],
            chosen_prior=reuse,exported_select=reuse['select'] if reuse else metrics)
    with (out/'FEATURES.npz').open('xb') as f: np.savez(f,real=x,simulation=xm,target=y,train=tr,select=se)
    files=[c.ROOT/'core.py',c.ROOT/'prepare.py',c.MODEL/'head.py',c.PREVIEW/'model.py',
        c.LAB.parent/'ballplate_fourframe96_deploy/outputs/single_frame_adaptation_20261020_v2/MODEL.xml',
        c.LAB.parent/'ballplate_optical_diffraction_ppo/env/ball_camera.py',
        c.EXP/'E080_ball_input_20261027_1/catalog.py',c.EXP/'E098_model_closedloop_20261029_1/evaluate.py']
    c.write(out/'PACKAGE.json',dict(heads=refs,model=c.ref(c.PREVIEW/'MODEL_01/MODEL.json'),context=context,
        mask=baseline['files']['slm'],layout=idx['layout'],radius=2,dt=.02,
        sources=[c.ref(p) for p in files],hardware_actions=0,diagnostic_only=True))
    c.write(out/'RESULT.json',dict(metrics=report,rows=rows,model_inputs=predictions,
        data_reused=True,new_hardware_actions=0,formal_TEST=False,
        model_SELECT_inputs_previously_seen_by_surrogate=True,
        model_mean_repeats_are_deterministic_NOT_noise_stability=True,
        fairness='same fixed mask, labels, states, splits, 130 affine parameters; heads fitted separately',
        closed_loop_validated=False))
    print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=__import__('pathlib').Path,required=True)
    run(p.parse_args().output)
