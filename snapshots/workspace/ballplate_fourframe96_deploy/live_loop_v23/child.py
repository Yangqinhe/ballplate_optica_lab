"""Pure MuJoCo child. No Qt, camera, optical predictor or real actuators."""
import argparse
import json
import os
from pathlib import Path
import select
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def emit(data): print(json.dumps(data, allow_nan=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head', type=Path, required=True); p.add_argument('--sha', required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--seconds', type=float, default=3.)
    p.add_argument('--initial-xy', type=float, nargs=2, default=[.025, -.025])
    p.add_argument('--seed', type=int, default=1036000)
    a = p.parse_args()
    import numpy as np
    if (not np.isfinite(a.seconds) or not .02 <= a.seconds <= 10
            or not np.isfinite(a.initial_xy).all() or max(map(abs, a.initial_xy)) > .07):
        raise ValueError('bounded MuJoCo-only experiment required')
    import mujoco
    import runtime96
    from model96 import resize_binary
    from env.ballplate_env import BallPlateEnv
    from live_deploy_v20.protocol import pack_dmd
    from live_loop_v23.core import load_head, limited_action, validate_push
    from live_adapt_v22.core import digest, write_new
    from tools.sparse_pilot_child_v151 import SparseFIFO
    head, bundle = load_head(a.head, a.sha)
    a.output.mkdir(parents=True, exist_ok=False)
    env = BallPlateEnv(seed=a.seed, xml=Path(bundle.data['model_xml']).read_text(), duration=a.seconds)
    fifo = SparseFIFO(3); previous = np.zeros(2); trace = []; seen = set(); start = time.monotonic()
    report = dict(completed=False, error=None, source='real_optical_single_frame', CCD_history=0,
        input_frame_stride=3, RAW_input_ages_s=[.18, .12, .06, 0.], raw_head_parameters=18,
        real_actuators_enabled=False, simulation_paused_during_optical_query=True,
        realtime_20ms_qualified=False, seed=a.seed, initial_xy_m=a.initial_xy,
        head_sha256=a.sha, release_approved=False)
    fallen = False
    try:
        raw = fifo.push96(resize_binary(env.reset(a.initial_xy)))
        for step in range(env.max_steps):
            dmd = a.output/f'DMD_{step:04d}.npy'; np.save(dmd, pack_dmd(raw), allow_pickle=False)
            rgb = a.output/f'MuJoCo_{step:04d}.npy'; np.save(rgb, env.camera.last_rgb, allow_pickle=False)
            emit(dict(kind='optical_query', sequence=step, dmd=str(dmd.resolve()), dmd_sha256=digest(dmd),
                rgb=str(rgb.resolve()), rgb_sha256=digest(rgb), created_unix_ns=time.time_ns(),
                simulation_time_s=env.steps*env.control_dt, position_m=env.position().tolist()))
            if not select.select([sys.stdin], [], [], 30)[0]: raise TimeoutError('fresh optical reply absent')
            line = sys.stdin.readline()
            if not line: raise InterruptedError('parent disconnected')
            reply = json.loads(line)
            if reply.get('error') or reply.get('sequence') != step or reply.get('dmd_sha256') != digest(dmd):
                raise ValueError('invalid optical reply association')
            values = np.asarray(reply['raw_integrals8'], float)
            raw_action, action = limited_action(head, values, previous); previous = action.copy()
            applied = []
            for event in reply.get('push_events', []):
                v = validate_push(event)
                if event['id'] in seen: raise ValueError('repeated disturbance id')
                seen.add(event['id']); applied.append(event)
                dof = int(env.model.jnt_dofadr[env.model.joint('ball_free').id])
                env.data.qvel[dof:dof+2] += v
            if applied: mujoco.mj_forward(env.model, env.data)
            image, reward, fallen, truncated, info = env.step(action)
            raw = fifo.push96(resize_binary(image))
            dof = int(env.model.jnt_dofadr[env.model.joint('ball_free').id])
            rgb_after = a.output/f'MuJoCo_after_{step:04d}.npy'
            np.save(rgb_after, env.camera.last_rgb, allow_pickle=False)
            row = dict(step=step, simulation_time_s=env.steps*env.control_dt, position_m=env.position().tolist(),
                velocity_world_m_s=env.data.qvel[dof:dof+2].tolist(), reward=float(reward), stable=bool(info['stable']),
                action_head_rad=raw_action.tolist(), action_limited_rad=action.tolist(),
                raw_integrals8=values.tolist(), push_events_applied=applied,
                query_latency_s=float(reply['query_latency_s']), fallen=bool(fallen),
                rgb=str(rgb_after.resolve()), rgb_sha256=digest(rgb_after))
            trace.append(row); write_new(a.output/f'step_{step:04d}.json', row)
            emit(dict(kind='mujoco_step', **row))
            if fallen or truncated: break
        report.update(completed=env.steps == env.max_steps and not fallen,
            fallen=bool(fallen), simulated_seconds=env.steps*env.control_dt,
            final_distance_m=float(np.linalg.norm(env.position()[:2])), first_settle_s=env.first_settle,
            convergence_NOT_established_by_short_interactive_run=True)
    except BaseException as error: report['error'] = repr(error)
    finally:
        env.close(); report.update(trace=trace, wall_seconds=time.monotonic()-start)
        write_new(a.output/'pilot.json', report)
        emit(dict(kind='pilot_complete', report={k: v for k, v in report.items() if k != 'trace'}))
    if report['error']: raise SystemExit(1)


if __name__ == '__main__':
    os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '1')
    main()
