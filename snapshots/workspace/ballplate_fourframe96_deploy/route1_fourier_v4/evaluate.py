"""MuJoCo-only route-1 development rollout; no real optics or actuators."""
import argparse
from pathlib import Path
import torch
from model import load, rt
from evaluate96 import rollout
from fourframe_training import Budget
from fourframe_evaluate import cases


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    rt.setup('cuda')
    budget = Budget(a.output, a.seconds)
    model, payload = load(a.checkpoint, 'cuda')
    model.eval().requires_grad_(False)
    if a.flat:
        with torch.no_grad():
            model.optical.phase_raw.fill_(-30.)
    requested = cases(a.episodes, a.cases.split(','))
    report = dict(checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=rt.digest(a.checkpoint),
                  geometry=model.geometry(), requested_cases=len(requested), results=[],
                  development_NOT_final_acceptance=True,
                  phase_ablation='flat_raw_gray_same_head' if a.flat else 'none',
                  hardware_enabled=False, real_actuators_enabled=False, release_approved=False)
    for label, seed, xy, push in requested:
        if budget():
            break
        row = rollout(model, payload['metadata']['model_xml'], label, seed, xy, push, budget)
        report['results'].append(row)
        report['complete'] = len(report['results']) == len(requested) and not any(r['interrupted'] for r in report['results'])
        report['strict_successes'] = sum(r['task_success'] for r in report['results'])
        rt.json_save(a.output/'evaluation.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', default='centre,push')
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--seconds', type=float, default=600)
    p.add_argument('--flat', action='store_true')
    a = p.parse_args()
    if not (0 < a.seconds <= 3600 and 1 <= a.episodes <= 8):
        p.error('bounded evaluation required')
    run(a)
