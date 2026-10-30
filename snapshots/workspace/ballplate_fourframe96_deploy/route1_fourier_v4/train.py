"""Higher-order ideal-4f warmstart with one physical mask; NOT PPO. No hardware."""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from model import Config, Policy, migrate_prior, load, save, resize_binary, rt
from fourframe_training import Budget, fit_head


@torch.no_grad()
def initialize(model, mode, checkpoint):
    if mode == 'resume_route':
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload['architecture'] == 'route1_RAW96x4_150_100_RAWGRAY_CCD8_LINEAR18_v1':
            return migrate_prior(model, checkpoint, copy_head=False)
        donor, payload = load(checkpoint, 'cpu')
        if donor.optical.phase_raw.shape != model.optical.phase_raw.shape:
            raise ValueError('cannot silently resize phase controls')
        model.optical.phase_raw.copy_(donor.optical.phase_raw.to(model.optical.phase_raw))
        # Only phase parameters are continued. New head is independently fitted;
        # the refined propagators are built from the new Config, not restored.
        return dict(mode=mode, source_checkpoint=str(Path(checkpoint).resolve()),
                    source_checkpoint_sha256=rt.digest(checkpoint), old_head_copied=False,
                    old_propagation_buffers_copied=False, new_focal_lengths_mm=[150, 100],
                    samples_per_canvas_pixel=model.cfg.simulation_size//600)
    if mode == 'split_xy':
        cfg = model.cfg
        magnification = cfg.f2_m/cfg.f1_m
        shift_x = cfg.canvas*cfg.registered_pitch_m*magnification/8
        shift_y = cfg.canvas*cfg.registered_pitch_m*magnification/4
        periods = [cfg.wavelength_m*cfg.f2_m/shift for shift in (shift_x, shift_y)]
        pitch = cfg.slm_block*cfg.registered_pitch_m
        if min(periods) < 2*pitch:
            raise ValueError('split-XY phase is underresolved on this SLM macro-pixel')
        axis = (np.arange(600//cfg.slm_block)+.5-300/cfg.slm_block)*pitch
        curve = model.optical.phase_command_response.cpu().numpy()
        depth = float(curve.max())
        amplitude = min(1.2, depth/5)
        target = depth/2 + amplitude*(np.sin(2*np.pi*axis[None, :]/periods[0])
                                     + np.sin(2*np.pi*axis[:, None]/periods[1]))
        raw = np.abs(target[:, :, None]-curve[None, None, :]).argmin(-1)
        model.optical.phase_raw.copy_(torch.logit(torch.as_tensor(raw/255., dtype=torch.float32)
            .clamp(1e-6, 1-1e-6)).to(model.optical.phase_raw))
        return dict(mode=mode, periods_m=periods, shifts_CCD_m=[shift_x, shift_y],
                    phase_amplitude_per_axis_rad=amplitude, Nyquist_control_pitch_m=pitch,
                    historical_response_prior=True, old_propagation_buffers_copied=False)
    if mode == 'flat':
        model.optical.phase_raw.fill_(-30.)
        return dict(mode=mode, old_propagation_buffers_copied=False)
    raise ValueError('unsupported Fourier initialization')


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    device = rt.setup('cuda')
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    device['allocator_fraction'] = a.memory_fraction
    budget = Budget(a.output, a.seconds)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    source = a.output/'source'
    source.mkdir()
    for p in Path(__file__).resolve().parent.glob('*.py'):
        shutil.copy2(p, source/p.name)
    data = torch.load(rt.FIXTURES, map_location='cpu', weights_only=False)
    frames = torch.from_numpy(resize_binary(data['frames'].numpy()))
    targets = data['actions'].clip(-.8, .8)
    fit_ids = torch.nonzero(data['train_mask']).flatten().numpy()
    rng.shuffle(fit_ids)
    selection = fit_ids[-a.selection:]
    train = fit_ids[:-a.selection]
    head_ids = train[np.linspace(0, len(train)-1, a.head_samples).astype(int)]
    model = Policy(Config(simulation_size=600*a.samples, refinement_samples={2: 3, 3: 4}[a.samples], gain_jitter=a.gain_jitter,
        phase_depth_jitter=a.phase_depth_jitter, read_noise_ADU=a.read_noise_adu,
        detector_shift_pixels=a.detector_shift_pixels)).to('cuda')
    migration = initialize(model, a.initialization, a.checkpoint)
    initial_gray = model.optical.phase_uint8().cpu().clone()
    model.critic.requires_grad_(False)
    model.log_std.requires_grad_(False)
    meta = dict(role='teacher_warmstart_and_optical_shaping_NOT_PPO', device=device,
                model_xml=data['model_xml'], model_config=asdict(model.cfg), geometry=model.geometry(),
                initialization=migration, fixture_sha256=rt.digest(rt.FIXTURES), seed=a.seed,
                train_indices=train.tolist(), FIT_selection_indices=selection.tolist(),
                frozen_validation_indices=torch.nonzero(~data['train_mask']).flatten().tolist(),
                frozen_validation_used_for_selection=False, teacher_targets_clipped_rad=[-.8, .8],
                updates=0, selected_update=0, hardware_enabled=False, release_approved=False,
                numerical_refinement_passed=False, domain_randomization_is_assumed_not_measured=True,
                source_training_sha256=rt.digest(__file__))
    rt.json_save(a.output/'config.json', meta)

    @torch.no_grad()
    def encode(ids, policy=model):
        policy.eval()
        q = []
        for i in ids:
            if budget():
                raise TimeoutError('bounded stop during encoding')
            q.append(policy.encode_frames(frames[int(i):int(i)+1].to('cuda')).cpu())
        return torch.cat(q)

    @torch.no_grad()
    def refit():
        q = encode(head_ids)
        selected = encode(selection)
        features = torch.cat((q, selected))
        labels = torch.cat((targets[head_ids], targets[selection]))
        mask = torch.arange(len(features)) < len(head_ids)
        # A small FIT-only second-grid cohort reduces head sensitivity to
        # discretization. This is training, not independent refinement proof.
        if a.refit_second_grid:
            twin = Policy(replace(model.cfg, simulation_size=1800)).to('cuda')
            twin.optical.phase_raw.copy_(model.optical.phase_raw)
            ids = head_ids[:a.refit_second_grid]
            q3 = encode(ids, twin)
            del twin
            gc.collect()
            torch.cuda.empty_cache()
            features = torch.cat((features, q3))
            labels = torch.cat((labels, targets[ids]))
            mask = torch.cat((mask, torch.ones(len(ids), dtype=torch.bool)))
        return fit_head(model, features, labels, mask, a.ridge)

    @torch.no_grad()
    def assess():
        error = model.actor_head(encode(selection).to('cuda')).cpu()-targets[selection]
        return dict(FIT_selection_mse=float(error.square().mean()), FIT_selection_mae=float(error.abs().mean()))

    optimizer = torch.optim.Adam([model.optical.phase_raw], lr=a.lr)
    try:
        meta['initial_head_fit'] = refit()
        meta['metrics'] = assess()
        best = meta['metrics']['FIT_selection_mse']
        save(a.output/'initial.pt', model, meta)
        save(a.output/'best.pt', model, meta)
        print('INITIAL', json.dumps(meta['metrics']), flush=True)
        model.actor_head.requires_grad_(False)
        for step in range(1, a.updates+1):
            if budget():
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for i in rng.choice(train, a.batch, replace=False):
                prediction = model(frames[int(i):int(i)+1].to('cuda'))
                mse = (prediction-targets[int(i):int(i)+1].to('cuda')).square().mean()
                phase = model.optical.phase_raw
                smooth = (phase[1:]-phase[:-1]).square().mean()+(phase[:, 1:]-phase[:, :-1]).square().mean()
                loss = (mse/.05**2 + 1e-4*smooth)/a.batch
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                loss.backward()
                loss_sum += float(mse.detach())/a.batch
            gradient = torch.nn.utils.clip_grad_norm_([model.optical.phase_raw], 1.)
            if not torch.isfinite(gradient):
                raise FloatingPointError('nonfinite phase gradient')
            optimizer.step()
            meta['updates'] = step
            row = dict(update=step, batch_mse=loss_sum, gradient_norm=float(gradient), seconds=budget.elapsed)
            if step % a.validate_every == 0:
                if step % a.refit_every == 0:
                    row['head_refit'] = refit()
                row.update(assess())
                row['changed_raw_gray_fraction'] = float((model.optical.phase_uint8().cpu() != initial_gray).float().mean())
                meta['changed_raw_gray_fraction'] = row['changed_raw_gray_fraction']
                meta['metrics'] = {k: row[k] for k in ('FIT_selection_mse', 'FIT_selection_mae')}
                if row['FIT_selection_mse'] < best:
                    best = row['FIT_selection_mse']
                    meta['selected_update'] = step
                    save(a.output/'best.pt', model, meta)
            with (a.output/'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if step % 10 == 0:
                print('TRAIN', json.dumps(row), flush=True)
        meta.update(stop_reason='budget_or_request' if budget() else 'max_updates', seconds=budget.elapsed)
        meta['changed_raw_gray_fraction'] = float((model.optical.phase_uint8().cpu() != initial_gray).float().mean())
        save(a.output/'last.pt', model, meta)
        rt.json_save(a.output/'status.json', meta)
    except TimeoutError as error:
        meta.update(stop_reason='bounded_stop_during_encoding', error=str(error), seconds=budget.elapsed)
        save(a.output/'interrupted.pt', model, meta)
        rt.json_save(a.output/'status.json', meta)
    except Exception as error:
        rt.json_save(a.output/'failure.json', dict(error=repr(error), seconds=budget.elapsed))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--initialization', choices=('migrate', 'flat', 'resume_route', 'split_xy'), default='resume_route')
    p.add_argument('--samples', type=int, choices=(2, 3), default=3)
    p.add_argument('--head-samples', type=int, default=384)
    p.add_argument('--selection', type=int, default=64)
    p.add_argument('--refit-second-grid', type=int, default=0)
    p.add_argument('--memory-fraction', type=float, default=.65)
    p.add_argument('--gain-jitter', type=float, default=.08)
    p.add_argument('--phase-depth-jitter', type=float, default=.04)
    p.add_argument('--read-noise-adu', type=float, default=.15)
    p.add_argument('--detector-shift-pixels', type=int, choices=(0, 1, 2), default=1)
    p.add_argument('--updates', type=int, default=400)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--ridge', type=float, default=.03)
    p.add_argument('--validate-every', type=int, default=25)
    p.add_argument('--refit-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=102001)
    p.add_argument('--seconds', type=float, default=1200)
    a = p.parse_args()
    if not (0 < a.head_samples <= 1024 and 0 < a.selection <= 512 and 0 <= a.refit_second_grid <= a.head_samples
            and a.refit_second_grid == 0 and .4 <= a.memory_fraction <= .65 and 0 < a.seconds <= 3600
            and a.refit_every % a.validate_every == 0 and a.batch > 0):
        p.error('bounded training arguments required')
    run(a)
