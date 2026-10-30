"""Frozen-mask/head refinement on static AND fresh moving four-frame inputs."""
import argparse
import gc
from pathlib import Path
import time

import numpy as np
import torch
from model import load, FIFO96, rt
import mujoco
from env.ballplate_env import BallPlateEnv
from fourframe_training import Budget


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    rt.setup(a.device)
    budget = Budget(a.output, a.seconds)
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    points = [(x, y) for x in (-.12, 0., .12) for y in (-.12, 0., .12)]
    protocol = dict(seed=102291, points=points, moving_speed_m_s=.30, steps_per_history=3,
        samples=a.samples, tolerance_rad=.002, checkpoint_sha256=rt.digest(a.checkpoint),
        cohort=a.cohort, selected_fixture_indices=list(range(18)) if a.cohort == 'full18' else [0, 1, 8, 9, 16, 17],
        protocol_frozen_before_rendering=True, hardware_enabled=False, release_approved=False)
    rt.json_save(a.output/'protocol.json', protocol)
    env = BallPlateEnv(xml=payload['metadata']['model_xml'], seed=protocol['seed'])
    fixtures, labels = [], []
    try:
        dof = int(env.model.jnt_dofadr[env.model.joint('ball_free').id])
        for index, xy in enumerate(points):
            fifo = FIFO96()
            raw = fifo.push(env.reset(xy))
            fixtures.append(raw.copy())
            labels.append(dict(kind='static', xy=xy))
            angle = 2*np.pi*index/len(points)
            env.data.qvel[dof:dof+2] = .30*np.array([np.cos(angle), np.sin(angle)])
            mujoco.mj_forward(env.model, env.data)
            for _ in range(3):
                image, _, fallen, _, _ = env.step(np.zeros(2))
                if fallen:
                    raise RuntimeError('audit history unexpectedly fell')
                raw = fifo.push(image)
            if np.array_equal(raw[0], raw[-1]):
                raise ValueError('moving history did not change its binary image')
            fixtures.append(raw.copy())
            labels.append(dict(kind='moving', xy=xy, velocity_angle_rad=float(angle)))
    finally:
        env.close()
    selected = protocol['selected_fixture_indices']
    fixtures, labels = [fixtures[index] for index in selected], [labels[index] for index in selected]
    raw = torch.from_numpy(np.stack(fixtures))
    np.savez_compressed(a.output/'fixtures.npz', raw_four=raw.numpy())
    report = dict(protocol, fixture_sha256=rt.digest(a.output/'fixtures.npz'), fixtures=labels,
        grids=[], comparisons=[], completed=False, passed=False,
        generated_numeric_cohort_NOT_task_success_validation=True,
        all_higher_sampling_and_real_hardware_not_proven=True,
        sources={str(Path(__file__).resolve()): rt.digest(__file__)})
    reference_phase = None
    try:
        for samples in a.samples:
            model = None
            row = dict(samples=samples, completed=False)
            begin = time.monotonic()
            try:
                model, _ = load(a.checkpoint, a.device, samples)
                model.eval().requires_grad_(False)
                phase = model.optical.phase_uint8().cpu()
                if reference_phase is None:
                    reference_phase = phase
                if not torch.equal(reference_phase, phase):
                    raise ValueError('physical mask changed across integration grids')
                features, actions = [], []
                with torch.no_grad():
                    for image in raw.split(1):
                        if budget():
                            raise TimeoutError('bounded integration audit stopped')
                        q = model.encode_frames(image.to(a.device))
                        features.append(q.cpu())
                        actions.append(model.actor_head(q).cpu())
                row.update(completed=True, quadrature_pair=model.geometry()['quadrature_pair'],
                    features=torch.cat(features).tolist(), actions=torch.cat(actions).tolist())
            except Exception as error:
                row['error'] = repr(error)
                raise
            finally:
                row['seconds'] = time.monotonic()-begin
                report['grids'].append(row)
                del model
                gc.collect()
                if a.device == 'cuda':
                    torch.cuda.empty_cache()
                rt.json_save(a.output/'refinement.json', report)
            print('GRID', samples, row['seconds'], flush=True)
        for lo, hi in zip(report['grids'], report['grids'][1:]):
            delta = (torch.tensor(lo['actions'])-torch.tensor(hi['actions'])).abs().amax(1)
            comparison = dict(pairs=[lo['quadrature_pair'], hi['quadrature_pair']],
                max_action_error_rad=float(delta.max()), SLM_raw_gray_bitwise_equal=True,
                static_max_rad=float(delta[0::2].max()), moving_max_rad=float(delta[1::2].max()),
                passed=bool(delta.max() <= .002))
            report['comparisons'].append(comparison)
        report.update(completed=True, passed=all(row['passed'] for row in report['comparisons']))
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report['seconds'] = budget.elapsed
        rt.json_save(a.output/'refinement.json', report)
    print('REFINEMENT', report['passed'], report['comparisons'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, nargs='+', default=[3, 4, 6])
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--cohort', choices=('full18', 'deep6'), default='full18')
    p.add_argument('--seconds', type=float, default=600)
    a = p.parse_args()
    if not 60 <= a.seconds <= 1800 or len(a.samples) < 2 or a.samples != sorted(set(a.samples)) or not set(a.samples) <= {2, 3, 4, 6}:
        p.error('bounded increasing integration ladder required')
    run(a)
