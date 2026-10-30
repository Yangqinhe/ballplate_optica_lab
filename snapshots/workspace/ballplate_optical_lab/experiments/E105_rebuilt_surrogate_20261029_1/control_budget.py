"""Offline truth-feedback error budget. NOT an optical controller or RL run."""
import argparse
import concurrent.futures
import math
from pathlib import Path
import time
import numpy as np
from workflow import ROOT,LAB,read,write,ref,require,load
from dataset import checked
from head import records,frozen_predict

ENGINE=LAB/'experiments/E087_mujoco_tolerance_20261028_1/evaluate.py'
ENGINE_SHA='8aba96c7a4d7efee882a37d1b041754f54ccaad7dfcba17a0b966fcaa9c8e8c3'


def sequence(config,steps,case_index):
    if config['kind']=='bias':
        return np.tile(config['amplitude']*np.array([math.cos(config['angle']),math.sin(config['angle'])]),(steps,1))
    cov=np.asarray(config['covariance_rad2'],float)*config['multiplier']**2
    eigen,vectors=np.linalg.eigh(cov);root=vectors@np.diag(np.sqrt(np.maximum(eigen,0)))
    rng=np.random.default_rng(config['seed']+1009*case_index)
    innovations=rng.normal(size=(steps,2))@root.T
    rho=math.exp(-.02/config['tau_sim_s']) if config['tau_sim_s'] else 0.
    out=np.empty_like(innovations);out[0]=innovations[0]
    for i in range(1,steps):out[i]=rho*out[i-1]+math.sqrt(1-rho*rho)*innovations[i]
    return out


def worker(args):
    config,folder=args;folder=Path(folder)
    require(ref(ENGINE)['sha256']==ENGINE_SHA,'frozen physics sensitivity implementation changed')
    engine=load(ENGINE,'e105_frozen_budget_engine')
    engine.error_sequence=sequence  # Isolated offline callback; original file is unchanged.
    teach=engine.teacher()
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(teach.MODEL))
    require(abs(model.opt.timestep-.002)<1e-12,'physics timestep changed')
    traces={};rows=[]
    for i,case in enumerate(engine.cases()):
        trace,metric=engine.episode(model,mujoco,teach,case,config,i)
        traces[case['id']]=trace;rows.append(dict(case=case,**metric))
    path=folder/'traces'/(config['id']+'.npz')
    with path.open('xb') as f:np.savez_compressed(f,**traces)
    result=dict(condition=config,cases=rows,traces=ref(path),fallen=sum(r['fallen'] for r in rows),
        settled=sum(r['settled'] for r in rows),
        worst_tail_RMS_m=max((r['tail_RMS_radius_m'] or 0.) for r in rows),
        incomplete_cases=sum(r['simulated_seconds']<10 for r in rows))
    write(folder/'conditions'/(config['id']+'.json'),result)
    return result


def run(model,dataset,output):
    output.mkdir();(output/'traces').mkdir();(output/'conditions').mkdir();(output/'SOURCE').mkdir()
    require(ref(ENGINE)['sha256']==ENGINE_SHA,'sensitivity engine changed')
    engine=load(ENGINE,'e105_budget_protocol');teach=engine.teacher()
    for source in (Path(__file__),ENGINE,engine.TP,Path(teach.MODEL)):
        with (output/'SOURCE'/source.name).open('xb') as f:f.write(source.read_bytes())
    idx,x,rows,areas=records(dataset);head=checked(read(model/'RESULT.json')['head'])
    require(head['dataset']==ref(dataset/'INDEX.json'),'noise data/head fit source differs')
    require(all(r['split'] in ('TRAIN','PILOT') for r in rows),'TRAIN-only noise estimate required')
    prediction=frozen_predict(head,x,areas);names=np.array([r['input_id'] for r in rows]);residual=[]
    for name in dict.fromkeys(names):
        p=prediction[names==name];residual.extend((p-p.mean(0))*np.sqrt(len(p)/(len(p)-1)))
    residual=np.asarray(residual);cov=residual.T@residual/len(residual)
    conditions=[dict(id='BASELINE',kind='bias',amplitude=0.,angle=0.)]
    for amplitude in (.02,.04,.08,.12,.16):
        for k in range(8):conditions.append(dict(id='BIAS_%g_%d'%(amplitude,k),kind='bias',amplitude=amplitude,angle=k*math.pi/4))
    for multiplier in (1.,2.):
        for tau in (0.,.2,1.):
            for seed in (17,83,211):
                conditions.append(dict(id='MEASURED_%g_%g_%d'%(multiplier,tau,seed),kind='measured_covariance',
                    multiplier=multiplier,tau_sim_s=tau,seed=seed,covariance_rad2=cov.tolist()))
    protocol=dict(dataset=ref(dataset/'INDEX.json'),head=ref(model/'HEAD.json'),source=ref(__file__),
        engine=ref(ENGINE),teacher=ref(engine.TP),MJCF=ref(teach.MODEL),
        covariance_rad2=cov.tolist(),measured_repeat_RMS_L2_rad=float(np.sqrt(np.trace(cov))),
        conditions=conditions,scenarios=engine.cases(),parameters=engine.PARAMETERS,
        DT_sim_s=.02,episode_sim_s=10,control_clip_rad=.8,slew_rad_per_sim_step=.08,actuator_alpha=teach.ALPHA,
        static_mapping_error_NOT_included_in_noise=True,
        time_constants_are_stress_assumptions_NOT_measured_wall_time=True,
        truth_feedback_FOR_DIAGNOSTIC_ONLY=True,optical_controller=False,hardware_actions=0,
        no_optical_surrogate_or_SLM_training=True,release_approved=False,
        criterion='no fall; 10s complete; last3s fraction(r<.02m AND speed<.05m/s)>=.9',
        columns=list(teach.COLUMNS)+['injected_roll_error_rad','injected_pitch_error_rad'])
    write(output/'PROTOCOL.json',protocol);start=time.monotonic();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(worker,[(c,str(output)) for c in conditions]):
            results.append(result)
            print('OFFLINE ERROR BUDGET',result['condition']['id'],result['settled'],'/25 settled',result['fallen'],'falls',flush=True)
    write(output/'RESULT.json',dict(protocol=ref(output/'PROTOCOL.json'),results=results,
        episodes=len(results)*25,wall_seconds=time.monotonic()-start,
        actual_optical_closed_loop=False,head_qualified=False,SLM_qualified=False))
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(9,4))
    amps=(.02,.04,.08,.12,.16)
    selected=[[r for r in results if r['condition']['kind']=='bias' and r['condition']['amplitude']==a] for a in amps]
    axes[0].plot(amps,[sum(r['settled'] for r in group)/(25*len(group)) for group in selected],'o-',color='#D55E00')
    axes[0].set(xlabel='Constant bias vector norm / rad',ylabel='Settled case fraction',ylim=(-.02,1.02),title='8 bias directions × 25 scenarios')
    for scale,color in ((1.,'#0072B2'),(2.,'#009E73')):
        values=[]
        for tau in (0.,.2,1.):
            group=[r for r in results if r['condition']['kind']=='measured_covariance' and
                   r['condition']['multiplier']==scale and r['condition']['tau_sim_s']==tau]
            values.append(sum(r['settled'] for r in group)/(25*len(group)))
        axes[1].plot((0.,.2,1.),values,'o-',label='%g × measured SD'%scale,color=color)
    axes[1].set(xlabel='Assumed correlation time / simulation s',ylabel='Settled case fraction',ylim=(-.02,1.02))
    axes[1].legend();fig.suptitle('Truth-feedback sensitivity ONLY | not real CCD control')
    fig.tight_layout();save_figure(fig,output/'01_control_error_budget',180);plt.close(fig)
    require(ref(ENGINE)['sha256']==ENGINE_SHA,'frozen dependency changed during simulation')


def sampling_profile(model,dataset,output):
    """Reuse archived per-RAW integrals; disjoint windows, no new measurement."""
    output.mkdir();idx,x,rows,areas=records(dataset);head=checked(read(model/'RESULT.json')['head'])
    require(head['dataset']==ref(dataset/'INDEX.json'),'sampling/head dataset differs')
    features=checked(idx['features'],True);offsets=features['offsets'];single=features['single_sums']/areas
    selected=[r for r in rows if r['split'] in ('TRAIN','PILOT')];results=[]
    durations=[];frame_steps=[]
    for row in selected:
        item=idx['rows'][row['source_index']];durations.append((item['host_times_ns'][-1]-item['host_times_ns'][0])/1e9)
        frame_steps.append((item['last_frame']-item['first_frame'])/19)
    for n in (1,2,5,10,20):
        groups={};short=[];short_actions=[]
        for row in selected:
            i=row['source_index'];v=single[offsets[i]:offsets[i+1]];require(v.shape==(20,64),'exact N20 source expected')
            means=v.reshape(20//n,n,64).mean(1)
            action=frozen_predict(head,means,areas)
            if len(means)>1:
                factor=np.sqrt(len(means)/(len(means)-1))
                short.extend((means-means.mean(0))*factor);short_actions.extend((action-action.mean(0))*factor)
            for slot,(mean,u) in enumerate(zip(means,action)):
                groups.setdefault((row['input_id'],slot),[]).append((mean,u))
        long=[];long_actions=[]
        for values in groups.values():
            v=np.array([x[0] for x in values]);u=np.array([x[1] for x in values]);require(len(v)>=2,'independent repeat visits required')
            factor=np.sqrt(len(v)/(len(v)-1));long.extend((v-v.mean(0))*factor);long_actions.extend((u-u.mean(0))*factor)
        results.append(dict(N=n,within_query_region_RMS_ADU=None if not short else float(np.sqrt(np.mean(np.square(short)))),
            within_query_action_RMS_L2_rad=None if not short_actions else float(np.sqrt(np.mean(np.sum(np.square(short_actions),axis=1)))),
            repeated_visit_region_RMS_ADU=float(np.sqrt(np.mean(np.square(long)))),
            repeated_visit_action_RMS_L2_rad=float(np.sqrt(np.mean(np.sum(np.square(long_actions),axis=1)))),
            per_region_repeat_SD_ADU=np.sqrt(np.mean(np.square(long),axis=0)).tolist()))
    write(output/'RESULT.json',dict(dataset=ref(dataset/'INDEX.json'),head=ref(model/'HEAD.json'),source=ref(__file__),
        TRAIN_effective_queries=len(selected),TRAIN_unique_inputs=len(set(r['input_id'] for r in selected)),results=results,
        N20_capture_span_s=dict(median=float(np.median(durations)),minimum=float(np.min(durations)),maximum=float(np.max(durations))),
        native_FrameID_mean_step=dict(median=float(np.median(frame_steps)),maximum=float(np.max(frame_steps))),
        subwindows_do_NOT_increase_independent_state_count=True,subwindows_may_be_temporally_correlated=True,
        N20_source_means_are_not_ground_truth=True,no_new_RAW_collected=True,no_filter_or_head_changed=True,
        wallclock_NOT_20ms_realtime=True,control_qualified=False))
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(9,3.8));nn=[r['N'] for r in results]
    for ax,key,label in [(axes[0],'repeated_visit_region_RMS_ADU','Repeat region RMS / ADU'),
                         (axes[1],'repeated_visit_action_RMS_L2_rad','Repeat action vector RMS / rad')]:
        ax.plot(nn,[r[key] for r in results],'o-',color='#0072B2');ax.set(xlabel='Archived RAW samples per disjoint mean',ylabel=label)
    fig.suptitle('TRAIN-only sampling diagnostic | fixed BASE mask and head');fig.tight_layout();save_figure(fig,output/'01_sampling_noise',180);plt.close(fig)
    print('ARCHIVED SAMPLING PROFILE',results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-offline',action='store_true');p.add_argument('--head',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--sampling-profile',action='store_true');a=p.parse_args()
    require(a.run_offline and all(Path(n).name==n for n in (a.head,a.dataset,a.output)),'explicit offline fresh paths required')
    if a.sampling_profile:sampling_profile(ROOT/a.head,ROOT/a.dataset,ROOT/a.output)
    else:run(ROOT/a.head,ROOT/a.dataset,ROOT/a.output)
