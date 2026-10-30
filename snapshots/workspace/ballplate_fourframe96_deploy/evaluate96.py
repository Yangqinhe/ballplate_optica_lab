"""RAW128 -> deterministic RAW96 FIFO -> fresh optical inference each step."""
import runtime96 as rt
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from env.ballplate_env import BallPlateEnv
from fourframe_training import Budget
from fourframe_evaluate import cases
from model96 import FIFO96,load


@torch.no_grad()
def rollout(model,xml,label,seed,xy,push,budget):
    env=BallPlateEnv(seed=seed,xml=xml,duration=15.);fifo=FIFO96();device=next(model.parameters()).device
    positions=[];actions=[];stable=[];distance=[];rewards=[];terminated=interrupted=False;started=time.monotonic()
    try:
        raw=fifo.push(env.reset(xy))
        for step in range(env.max_steps):
            if budget():interrupted=True;break
            action=model(torch.from_numpy(raw)[None].to(device))[0].cpu().numpy().clip(-.8,.8)
            if push and step==250:
                import mujoco
                dof=int(env.model.jnt_dofadr[env.model.joint("ball_free").id])
                env.data.qvel[dof:dof+2]+=np.array([push,0.]) if seed%2==0 else np.array([0.,-push])
                mujoco.mj_forward(env.model,env.data)
            frame,reward,terminated,truncated,info=env.step(action);raw=fifo.push(frame)
            positions.append(env.position()[:2].tolist());actions.append(action.tolist())
            stable.append(info["stable"]);distance.append(info["distance"]);rewards.append(reward)
            if (step+1)%250==0:print("HEARTBEAT",seed,step+1,flush=True)
            if terminated or truncated:break
        complete=env.steps==750 and not terminated and not interrupted
        tail=float(np.mean(stable[-250:])) if complete else None
        success=complete and env.first_settle is not None and env.first_settle<=5. and tail>=.8
        row=dict(case=label,seed=seed,initial_xy_m=xy,push_speed_m_s=push,steps=env.steps,
            complete=complete,fallen=bool(terminated),interrupted=interrupted,task_success=bool(success),
            first_settle_s=env.first_settle,tail_stable_fraction=tail,
            mean_distance_m=float(np.mean(distance)) if distance else None,
            return_sum=float(sum(rewards)),position_trace=positions,action_trace=actions,
            wall_seconds=time.monotonic()-started,actor_parameters=18,CCD_history=0,hardware_enabled=False)
        print("EVAL",json.dumps({k:v for k,v in row.items() if not k.endswith("trace")}),flush=True);return row
    finally:env.close()


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);rt.setup(args.device);budget=Budget(args.output,args.seconds)
    model,payload=load(args.checkpoint,args.device,args.simulation_size);model.eval().requires_grad_(False)
    if args.flat:
        with torch.no_grad():model.optical.phase_raw.fill_(-30)
    requested=cases(2,args.cases.split(','));rows=[]
    report=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=rt.digest(args.checkpoint),
        geometry=model.geometry(),simulation_size=model.cfg.simulation_size,
        development_selection_NOT_independent_test=True,phase_ablation="zero_command_same_head" if args.flat else "none",
        success_definition="15s no fall; first25 stable steps <=5s; last5s stable >=80%",requested_cases=len(requested),
        registered_grid_surrogate_not_measured_current_optics=True,hardware_enabled=False)
    for label,seed,xy,push in requested:
        if budget():break
        rows.append(rollout(model,payload['metadata']['model_xml'],label,seed,xy,push,budget))
        report.update(results=rows,complete=len(rows)==len(requested) and not any(r['interrupted'] for r in rows))
        rt.json_save(args.output/'evaluation.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--simulation-size',type=int,default=1200)
    p.add_argument('--cases',default='centre,push');p.add_argument('--seconds',type=float,default=900)
    p.add_argument('--flat',action='store_true');run(p.parse_args())
