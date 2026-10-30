"""Closed-loop MuJoCo validation with RAW-input FIFO only, never a CCD cache."""
import fourframe_runtime as rt
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from env.ballplate_env import BallPlateEnv
from fourframe_model import FourFramePolicy, RawFrameFIFO
from fourframe_checkpoint import load
from fourframe_training import Budget, initialize


@torch.no_grad()
def rollout(model, xml, seed, xy, push_speed=0., budget=lambda:False):
    device = next(model.parameters()).device
    env = BallPlateEnv(seed=seed, xml=xml, initial_range=.15, duration=15.)
    fifo = RawFrameFIFO()
    positions, distances, stable, rewards, actions = [], [], [], [], []
    interrupted = terminated = False
    started = time.monotonic()
    try:
        raw = fifo.push(env.reset(np.asarray(xy)))
        for step in range(env.max_steps):
            if budget(): interrupted = True; break
            # Exactly one propagation per action; no memoized CCD/feature values.
            action = model(torch.from_numpy(raw)[None].to(device))[0].cpu().numpy().clip(-.8, .8)
            if push_speed and step == 250:
                import mujoco
                dof = int(env.model.jnt_dofadr[env.model.joint("ball_free").id])
                env.data.qvel[dof:dof+2] += np.array([push_speed, 0.]) if seed%2 == 0 else np.array([0., -push_speed])
                mujoco.mj_forward(env.model, env.data)
            frame, reward, terminated, truncated, info = env.step(action)
            raw = fifo.push(frame)
            positions.append(env.position()[:2].tolist()); distances.append(info["distance"])
            stable.append(info["stable"]); rewards.append(reward); actions.append(action.tolist())
            if (step+1)%200 == 0: print("EVAL_HEARTBEAT", seed, step+1, flush=True)
            if terminated or truncated: break
        complete = env.steps == env.max_steps and not terminated and not interrupted
        tail = float(np.mean(stable[-250:])) if complete else 0.
        settled = env.first_settle is not None and env.first_settle <= 5.
        recovery = None
        if push_speed:
            after = stable[250:]
            for i in range(24, len(after)):
                if all(after[i-24:i+1]): recovery = (i+1)*env.control_dt; break
        return dict(seed=seed, initial_xy_m=xy, push_speed_m_s=push_speed, steps=env.steps,
                    complete=complete, interrupted=interrupted, fallen=bool(terminated),
                    task_success=bool(complete and settled and tail >= .8),
                    mean_distance_m=float(np.mean(distances)) if distances else None,
                    tail_mean_distance_m=float(np.mean(distances[-250:])) if complete else None,
                    tail_stable_fraction=tail, first_settle_s=env.first_settle, recovery_s=recovery,
                    return_sum=float(sum(rewards)), wall_seconds=time.monotonic()-started,
                    position_trace=positions, action_trace=actions,
                    optical_acquisitions=0 if getattr(model, "centroid_baseline", False) else env.steps,
                    ccd_feature_cache_entries=0,
                    actor_inputs_are_current_ccd_only=not getattr(model, "centroid_baseline", False),
                    full_image_centroid_baseline=bool(getattr(model, "centroid_baseline", False)),
                    hardware_enabled=False)
    finally: env.close()


def cases(episodes=2, include=("centre", "far", "corner", "push")):
    rng = np.random.default_rng(92001)
    result = [("centre", 92010+i, xy.tolist(), 0.) for i,xy in enumerate(rng.uniform(-.05, .05, (episodes, 2)))]
    result += [("far", 92030+i, xy, 0.) for i,xy in enumerate(([.12,0.],[-.12,0.],[0.,.12],[0.,-.12]))]
    result += [("corner", 92130+i, xy, 0.) for i,xy in enumerate(([-.12,-.12],[-.12,.12],[.12,-.12],[.12,.12]))]
    result += [("push", 92050+i, xy.tolist(), .35) for i,xy in enumerate(rng.uniform(-.05, .05, (2,2)))]
    return [row for row in result if row[0] in include]


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    rt.setup(args.device); budget = Budget(args.output, args.seconds)
    source, payload = load(args.checkpoint, "cpu")
    cfg = source.encoder.cfg
    if args.native:
        cfg = replace(cfg, propagation="asm", grid_size=4608, samples_per_mirror=3)
    model = FourFramePolicy(cfg, **source.options).to(args.device)
    model.load_state_dict(source.state_dict()); del source
    if args.phase_ablation == "flat": initialize(model, "flat")
    model.eval().requires_grad_(False)
    requested = cases(args.episodes, args.cases.split(","))
    rows = []
    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=rt.digest(args.checkpoint),
                  optical_config=cfg.to_dict(), phase_ablation=args.phase_ablation,
                  source_metadata=payload["metadata"], hardware_enabled=False,
                  development_not_final_acceptance=True, requested_cases=len(requested),
                  success_definition="15s without fall; settle <=5s; last5s stable fraction >=0.8",
                  source_sha256=rt.digest(__file__))
    try:
        for label, seed, xy, push in requested:
            if budget(): break
            row = rollout(model, payload["metadata"]["model_xml"], seed, xy, push, budget)
            row["case"] = label; rows.append(row)
            print("EVAL", json.dumps({k:v for k,v in row.items() if not k.endswith("trace")}), flush=True)
            report.update(results=rows, complete=len(rows)==len(requested) and not any(r["interrupted"] for r in rows),
                          wall_seconds=budget.elapsed)
            rt.json_save(args.output/"evaluation.json", report)
    except Exception as error:
        rt.json_save(args.output/"failure.json", dict(error=repr(error), wall_seconds=budget.elapsed)); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--phase-ablation", choices=("none", "flat"), default="none")
    parser.add_argument("--cases", default="centre,far,corner,push")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=1800)
    run(parser.parse_args())
