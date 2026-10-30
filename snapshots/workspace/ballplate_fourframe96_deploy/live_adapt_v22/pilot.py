"""MuJoCo-only child. Optical features arrive over a bounded parent pipe.

The simulator PAUSES while the real optical query settles. This first pilot
is a slow hardware-in-the-loop functional test, NOT a realtime controller.
No SDK, Qt, network, actuator interface or optical propagation is imported.
"""
import argparse
import json
import os
from pathlib import Path
import select
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--head-sha256', required=True)
    p.add_argument('--seconds', type=float, default=3.)
    p.add_argument('--mock-for-tests', action='store_true')
    a = p.parse_args()
    if not 0 < a.seconds <= 3:
        p.error('maximum three simulated seconds per pilot')
    import numpy as np
    import runtime96
    from model96 import FIFO96
    from env.ballplate_env import BallPlateEnv
    from live_deploy_v20.protocol import pack_dmd
    from live_adapt_v22.core import digest, predict, write_new, Bundle
    if digest(a.head) != a.head_sha256:
        raise ValueError('explicit frozen head digest required')
    head = json.loads(a.head.read_text()); bundle = Bundle(head['bundle'])
    if not a.mock_for_tests and head.get('real_hardware_features') is not True:
        raise ValueError('synthetic feature head cannot be called a real optical pilot')
    if not head['short_MuJoCo_pilot_eligible'] and not a.mock_for_tests:
        raise ValueError('held-out single-frame criterion failed; do not run a real optical pilot')
    if (head['bundle_sha256'] != bundle.sha
            or head['phase_sha256'] != digest(bundle.data['SLM_RAW_GRAY_NO_LUT'])
            or head['native_weights_sha256'] != digest(bundle.data['native_weights'])):
        raise ValueError('head/calibration/mask mismatch')
    a.output.mkdir(parents=True, exist_ok=False)
    env = BallPlateEnv(seed=102023, xml=Path(bundle.data['model_xml']).read_text(), duration=a.seconds)
    fifo = FIFO96(); trace = []; previous = np.zeros(2); started = time.monotonic()
    report = dict(completed=False, error=None, real_actuators_enabled=False,
                  source='mock' if a.mock_for_tests else 'real_optical_single_frame',
                  frozen_head_sha256=a.head_sha256, physical_dt_s=env.control_dt,
                  simulation_paused_during_optical_query=True, realtime_20ms_qualified=False,
                  CCD_history=0, raw_head_parameters=18, release_approved=False)
    try:
        raw = fifo.push(env.reset([.025, -.025]))
        for step in range(env.max_steps):
            if time.monotonic()-started > 600:
                raise TimeoutError('MuJoCo pilot wall clock bound exceeded')
            path = a.output/f'DMD_{step:04d}.npy'
            np.save(path, pack_dmd(raw), allow_pickle=False)
            request = dict(kind='optical_query', sequence=step, dmd=str(path.resolve()),
                           dmd_sha256=digest(path), created_unix_ns=time.time_ns())
            print(json.dumps(request), flush=True)
            if not select.select([sys.stdin], [], [], 12)[0]:
                raise TimeoutError('no fresh optical reply; no stale feature reuse')
            line = sys.stdin.readline()
            if not line:
                raise InterruptedError('parent disconnected')
            reply = json.loads(line)
            if reply.get('sequence') != step or reply.get('error'):
                raise RuntimeError('optical reply rejected: '+str(reply))
            if reply['dmd_sha256'] != request['dmd_sha256']:
                raise ValueError('wrong optical input association')
            value = np.asarray(reply['raw_integrals8'], np.float64)
            action_raw = predict(head, value)
            action = np.clip(action_raw, -.8, .8)
            # Explicit pilot-only slew limiter; report separately from head.
            action = np.clip(action, previous-.08, previous+.08); previous = action.copy()
            image, reward, fallen, truncated, info = env.step(action)
            raw = fifo.push(image)
            row = dict(step=step, simulation_time_s=env.steps*env.control_dt,
                       position_m=env.position().tolist(), action_head_rad=action_raw.tolist(),
                       action_limited_rad=action.tolist(), raw_integrals8=value.tolist(),
                       query_latency_s=float(reply['query_latency_s']),
                       stable=bool(info['stable']), reward=float(reward), fallen=bool(fallen))
            trace.append(row)
            write_new(a.output/f'step_{step:04d}.json', row)
            print(json.dumps(dict(kind='mujoco_step', **row)), flush=True)
            if fallen or truncated:
                break
        report.update(completed=env.steps == env.max_steps and not fallen, fallen=bool(fallen),
                      simulated_seconds=env.steps*env.control_dt,
                      final_distance_m=float(np.linalg.norm(env.position()[:2])),
                      first_settle_s=env.first_settle,
                      convergence_NOT_established_by_three_second_pilot=True)
    except BaseException as error:
        report['error'] = str(error)
    finally:
        env.close()
        report.update(wall_seconds=time.monotonic()-started, trace=trace)
        write_new(a.output/'pilot.json', report)
        print(json.dumps(dict(kind='pilot_complete', report=report)), flush=True)
    if report['error']:
        raise SystemExit(1)


if __name__ == '__main__':
    # Never use desktop GL in this child; leave desktop/UI GPU unmodified.
    os.environ.setdefault('MUJOCO_GL', 'egl')
    main()
