"""Teacher-action initialization then optical-only shaping with a fixed tiny head.

The focal FFT backend is a declared warm start, not hardware validation. Native
CCD pixel area sampling, saturation and Mono8 quantization remain in both modes.
No raw frame is propagated separately: one packed four-time image per forward.
"""
import fourframe_runtime as rt
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from fourframe_model import FourFramePolicy, default_config, geometry
from fourframe_checkpoint import save, load


class Budget:
    def __init__(self, output, seconds):
        self.output, self.seconds = Path(output), seconds
        self.started, self.requested = time.monotonic(), False
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, self.request)

    def request(self, *_): self.requested = True
    @property
    def elapsed(self): return time.monotonic()-self.started
    def __call__(self): return self.requested or self.elapsed >= self.seconds or (self.output/"stop.request").exists()


@torch.no_grad()
def initialize(model, kind, seed=97103):
    cfg, slm = model.encoder.cfg, model.encoder.slm
    if kind == "flat": slm.raw.zero_(); return
    generator = torch.Generator().manual_seed(seed)
    if kind == "random":
        slm.raw.copy_((torch.randn(slm.raw.shape, generator=generator)*.5).to(slm.raw)); return
    axis = (torch.arange(cfg.phase_controls, dtype=torch.float64)-(cfg.phase_controls-1)/2)
    axis = axis*(cfg.slm_roi_pixels*cfg.slm_pitch_m/cfg.phase_controls)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    if kind == "split_y":
        # Physical phase grating: two first-order copies shifted by half a tile.
        # ONE shared mask; this is not an independent learned kernel per time.
        desired_shift = 64*cfg.superpixel*cfg.dmd_pitch_m*cfg.magnification
        period = cfg.wavelength_m*cfg.f2_m/desired_shift
        angle = math.pi+1.84*torch.sin(2*math.pi*y/period)
    elif kind == "defocus":
        angle = (math.pi+math.pi*.15*(x*x+y*y)/(cfg.wavelength_m*cfg.f1_m**2)).remainder(2*math.pi)
    else: raise ValueError(kind)
    slm.raw.copy_(torch.logit((angle/(2*math.pi)).clamp(1/255, 254/255)).to(slm.raw))


@torch.no_grad()
def encode(model, data, take, budget):
    device = next(model.parameters()).device
    rows, inverse = data["indices"][take, -4:].unique(dim=0, return_inverse=True)
    outputs = []
    for i, row in enumerate(rows):
        if budget(): raise TimeoutError("stopped while encoding frozen-mask dataset")
        frames = data["frames"][row][None].to(device)
        outputs.append(model.encode_frames(frames).cpu())
        if (i+1) % 200 == 0: print("ENCODE", i+1, "/", len(rows), flush=True)
    return torch.cat(outputs)[inverse]


@torch.no_grad()
def fit_head(model, q, actions, train, ridge=.01):
    x, y = q.double(), actions.double()
    mean, scale = x[train].mean(0), x[train].std(0).clamp_min(1e-4)
    z = (x[train]-mean)/scale; ym = y[train].mean(0)
    weight = torch.linalg.solve(z.T@z+ridge*len(z)*torch.eye(x.shape[1], dtype=torch.float64),
                                z.T@(y[train]-ym))
    weight = (weight/scale[:, None]).T; bias = ym-weight@mean
    model.actor_head.weight.copy_(weight.to(model.actor_head.weight))
    model.actor_head.bias.copy_(bias.to(model.actor_head.bias))
    error = x@weight.T+bias-y
    return dict(train_mse=float(error[train].square().mean()),
                validation_mse=float(error[~train].square().mean()),
                validation_mae=float(error[~train].abs().mean()),
                head_weight_max=float(weight.abs().max()), train_samples=int(train.sum()),
                validation_samples=int((~train).sum()))


@torch.no_grad()
def assess(model, data, take, budget):
    q = encode(model, data, take, budget)
    prediction = model.actor_mean(q.to(next(model.parameters()).device)).cpu()
    error = prediction-data["actions"][take]
    return dict(validation_mse=float(error.square().mean()), validation_mae=float(error.abs().mean()))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    device_info = rt.setup(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    budget = Budget(args.output, args.seconds)
    data = torch.load(rt.DATA, map_location="cpu", weights_only=False)
    rt.reused.verify_sources(data["source_manifest"])
    if args.checkpoint:
        model, payload = load(args.checkpoint, args.device)
        cfg = model.encoder.cfg
        if cfg.propagation != args.backend:
            cfg = replace(cfg, propagation=args.backend, grid_size=args.grid, samples_per_mirror=args.samples)
            state = model.state_dict(); options = model.options
            del model
            model = FourFramePolicy(cfg, **options).to(args.device); model.load_state_dict(state)
    else:
        cfg = replace(default_config(), propagation=args.backend, grid_size=args.grid,
                      samples_per_mirror=args.samples, phase_controls=args.controls)
        model = FourFramePolicy(cfg, readout=args.readout).to(args.device)
        initialize(model, args.initialization, args.seed)
    phase_pitch = cfg.wavelength_m*cfg.f1_m/(cfg.grid_size*cfg.grid_pitch_m) if cfg.propagation == "focal_fft" else cfg.grid_pitch_m
    control_pitch = cfg.slm_pitch_m*cfg.slm_roi_pixels/cfg.phase_controls
    if phase_pitch > control_pitch/2:
        raise ValueError("SLM control pitch undersampled; increase grid or reduce phase controls")
    from fast_slm_v2 import install
    install(model)
    take = torch.arange(0, len(data["actions"]), args.stride)
    mask = data["train_mask"][take]
    heldout = take[~mask]
    selection = heldout[torch.linspace(0, len(heldout)-1, min(32, len(heldout))).long()]
    metadata = dict(model_xml=data["model_xml"], seed=args.seed, steps=0, updates=0, selected_update=0,
                    role="optical_shaping_fixed_linear_head" if args.updates else "frozen_phase_linear_head_baseline",
                    dataset_sha256=rt.digest(rt.DATA), teacher_actions_only=True,
                    teacher_weights_copied=False, privileged_actor_state=False,
                    no_ccd_or_feature_history_at_inference=True, geometry=geometry(cfg),
                    actor_parameters=model.actor_parameters(), actor_macs=2*model.feature_dim,
                    electrically_integrated_pixels=cfg.ccd_roi_pixels**2, camera_side_binning_enabled=False,
                    device=device_info, initialization=args.initialization,
                    phase_plane_pitch_m=phase_pitch, control_pitch_m=control_pitch,
                    control_sampling_gate_passed=True, native_slm_pixels_resolved=phase_pitch <= cfg.slm_pitch_m/2,
                    backend_status="native_ASM_candidate" if cfg.propagation == "asm" else "ideal_4f_warmstart_NOT_ASM_validated",
                    numerical_refinement_passed=False, hardware_enabled=False, calibrated=False,
                    source_training_sha256=rt.digest(__file__), fast_slm_sha256=rt.digest(rt.INNOV/"fast_slm_v2.py"),
                    warmstart_source=None if not args.checkpoint else str(args.checkpoint.resolve()))
    rt.json_save(args.output/"config.json", dict(optical_config=cfg.to_dict(), options=model.options,
                 arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, metadata=metadata))
    try:
        q = encode(model, data, take, budget)
        metrics = fit_head(model, q, data["actions"][take], mask, args.ridge)
        metadata["initial_head_fit"] = metrics
        metadata["metrics"] = assess(model, data, selection, budget)
        best = metadata["metrics"]["validation_mse"]
        save(args.output/"initial.pt", model, metadata); save(args.output/"best.pt", model, metadata)
        print("INITIAL", json.dumps(metrics), flush=True)
        original_phase = model.encoder.slm.raw.detach().clone()
        original_head = {k:v.detach().clone() for k,v in model.actor_head.state_dict().items()}
        model.actor_head.requires_grad_(False)
        model.critic.requires_grad_(False); model.log_std.requires_grad_(False)
        optimizer = torch.optim.Adam([model.encoder.slm.raw], lr=args.lr)
        training = torch.nonzero(data["train_mask"]).flatten().numpy()
        for step in range(args.updates):
            if budget(): break
            optimizer.zero_grad(set_to_none=True); batch = rng.choice(training, args.batch, replace=False)
            mse_sum = 0.
            for i in batch:
                frames = data["frames"][data["indices"][i, -4:]][None].to(args.device)
                prediction = model(frames)
                mse = (prediction-data["actions"][i:i+1].to(args.device)).square().mean()
                loss = (mse/.05**2+1e-5*model.encoder.slm.smoothness())/len(batch)
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite optical shaping loss")
                loss.backward(); mse_sum += float(mse.detach())/len(batch)
            norm = torch.nn.utils.clip_grad_norm_([model.encoder.slm.raw], 1.)
            if not torch.isfinite(norm): raise FloatingPointError("nonfinite SLM gradient")
            optimizer.step(); metadata["updates"] = step+1
            row = dict(update=step+1, batch_mse=mse_sum, phase_gradient_norm=float(norm), wall_s=budget.elapsed)
            if (step+1) % args.validate_every == 0:
                score = assess(model, data, selection, budget); row.update(score)
                metadata["metrics"] = score
                if score["validation_mse"] < best:
                    best = score["validation_mse"]; metadata["selected_update"] = step+1
                    save(args.output/"best.pt", model, metadata, optimizer)
            with (args.output/"progress.jsonl").open("a") as out: out.write(json.dumps(row)+"\n")
            if (step+1) % 10 == 0: print("PHASE", json.dumps(row), flush=True)
        metadata.update(stop_reason="time_budget_or_request" if budget() else "max_updates",
                        wall_seconds=budget.elapsed,
                        phase_max_raw_change=float((model.encoder.slm.raw-original_phase).detach().abs().max()),
                        linear_head_bitwise_unchanged_during_phase_learning=all(
                            torch.equal(model.actor_head.state_dict()[k], v) for k,v in original_head.items()))
        save(args.output/"last.pt", model, metadata, optimizer)
        rt.json_save(args.output/"status.json", metadata)
        print("FINISHED", args.output, flush=True)
    except TimeoutError as error:
        metadata.update(stop_reason="time_budget_or_request_during_encoding", error=str(error), wall_seconds=budget.elapsed)
        rt.json_save(args.output/"status.json", metadata)
        print("STOP", str(error), flush=True)
    except Exception as error:
        rt.json_save(args.output/"failure.json", dict(error=repr(error), wall_seconds=budget.elapsed)); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("asm", "focal_fft"), default="asm")
    parser.add_argument("--grid", type=int, default=4608)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--controls", type=int, default=128)
    parser.add_argument("--initialization", choices=("flat", "random", "split_y", "defocus"), default="split_y")
    parser.add_argument("--readout", choices=("tiles8", "diagnostic16", "quad4"), default="tiles8")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--ridge", type=float, default=.01)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=.005)
    parser.add_argument("--validate-every", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--seed", type=int, default=97103)
    run(parser.parse_args())
