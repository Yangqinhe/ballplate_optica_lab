"""E005 CPU references and four GLOBAL physical-parameter finite differences.

No fitting, state estimator, hardware client, SLM training, or online correction.
Default preparation validates metadata/assets only; old Torch imports occur in
an explicitly executed, CPU-only subprocess. Parameter rank is NOT measured
state/velocity observability and is never a hardware or deployment release.
"""
from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from . import simulate as sim

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "optical-lab-global-physics-sensitivity-1"
BANK_SCHEMA = "optical-lab-observability-bank-1"
DESIGN_SHA = "0df06c6040b9c98c7f303e17db64648c014a904231e216c8e1012395b4dc2cf6"
GRID = ("REF", "X_PLUS", "X_MINUS", "Y_PLUS", "Y_MINUS", "VX_PLUS", "VX_MINUS", "VY_PLUS", "VY_MINUS")
MODELS = ("M0_REFERENCE", "M1_RADIAL")
MASKS = ("learned", "lens", "flat")
PARAMETERS = ("effective_input_defocus_m", "phase_response_multiplier",
              "global_ADU_gain_multiplier", "global_latent_ADU_bias")
NOMINAL = (0., 1., 1., 0.)
STEPS = (.0005, .03, .05, .005)
UNRELEASED = dict(hardware_accessed=False, projection_executed=False,
                  parameters_fitted=False, new_SLM_trained=False,
                  control_release_approved=False, production_qualified=False)


def _digest(blob):
    return hashlib.sha256(blob).hexdigest()


def _read(path, bindings, expected=None):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("regular bound file required: " + str(path))
    blob = path.read_bytes()
    value = _digest(blob)
    if expected is not None and value != expected:
        raise ValueError("source/asset SHA differs: " + str(path))
    if str(path) in bindings and bindings[str(path)] != value:
        raise ValueError("source/asset changed during operation: " + str(path))
    bindings[str(path)] = value
    return blob


def _json(path, bindings, expected=None):
    return json.loads(_read(path, bindings, expected))


def _array(path, bindings, expected=None):
    return np.load(io.BytesIO(_read(path, bindings, expected)), allow_pickle=False)


def _write(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _check(bindings):
    for path, expected in list(bindings.items()):
        _read(path, bindings, expected)


def _parent_bindings(legacy, bindings):
    """Same explicit frozen M0/M1 families as simulate, not latest selection."""
    for relative, expected in sim.FIXED.items():
        if relative not in (sim.RAW, sim.DMD):
            _read(legacy / relative, bindings, expected)
    reference = _json(legacy / sim.REFERENCE, bindings)
    fit = _json(legacy / sim.WAVEFIT, bindings)
    if (fit["reference_path"] != str(legacy / sim.REFERENCE)
            or fit["reference_sha256"] != sim.FIXED[sim.REFERENCE]
            or fit.get("completed") is not True or fit.get("selection_frozen") is not True
            or fit.get("error") is not None or fit["wavefront_kind"] != "radial"
            or fit["physical_DMD_micromirror_pitch_m"] is not None or fit["samples"] != [2, 3]):
        raise ValueError("exact completed registered8um M0/M1 parents required")
    for table in (reference["sources"], fit["source_sha256"]):
        for path, expected in table.items():
            _read(path, bindings, expected)
    for pk, sk in (("protocol_path", "protocol_sha256"), ("selection_path", "selection_sha256")):
        _read(fit[pk], bindings, fit[sk])
    for pk, sk in (("native_readout_regions_path", "native_readout_regions_sha256"),
                   ("frozen_original_head_path", "frozen_original_head_sha256"),
                   ("historical_raw_gray_response_path", "historical_raw_gray_response_sha256")):
        _read(fit["physical_manifest"][pk], bindings, fit["physical_manifest"][sk])


def _preflight(bank_dir, expected_bank_sha256, legacy_root):
    bank_dir = Path(bank_dir).resolve(strict=True)
    bindings = {}
    manifest = _json(bank_dir / "MANIFEST.json", bindings, expected_bank_sha256)
    if manifest.get("schema") != BANK_SCHEMA:
        raise ValueError("explicit E005 observability bank required")
    actual = {str(p.relative_to(bank_dir)) for p in bank_dir.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}
    if actual != set(manifest["file_sha256"]):
        raise ValueError("bank file inventory differs")
    for name, digest in manifest["file_sha256"].items():
        p = bank_dir / name
        if Path(name).is_absolute() or ".." in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(bank_dir):
            raise ValueError("bank asset path escapes archive")
        _read(p, bindings, digest)
    bank = _json(bank_dir / "BANK.json", bindings)
    design = _json(bank_dir / "DESIGN.json", bindings, DESIGN_SHA)
    if bank.get("schema") != BANK_SCHEMA or bank["design"] != design or design["hardware_actions_enabled"] is not False:
        raise ValueError("bank/design identity differs")
    if set(bank["inputs"]) != {"R" + str(r) + "_" + g for r in (0, 2) for g in GRID} or set(bank["masks"]) != set(MASKS):
        raise ValueError("exact eighteen calibration histories and three fixed masks required")
    for key, record in bank["inputs"].items():
        rawref = record["raw96"]
        if (Path(rawref["path"]).parent != bank_dir or Path(record["path"]).parent != bank_dir
                or record["id"] != key or record["history_ages_s"] != [.18, .12, .06, 0.]):
            raise ValueError("explicit local chronological RAW/DMD identity required")
        raw = _array(rawref["path"], bindings, rawref["sha256"])
        dmd = _array(record["path"], bindings, record["sha256"])
        if raw.shape != (4, 1, 96, 96) or raw.dtype != np.uint8 or not np.isin(raw, (0, 255)).all():
            raise ValueError("exact four binary RAW96 histories required")
        packed = np.pad(np.block([[raw[0, 0], raw[1, 0]], [raw[2, 0], raw[3, 0]]]).repeat(3, 0).repeat(3, 1), 12)
        if dmd.shape != (600, 600) or dmd.dtype != np.uint8 or not np.array_equal(dmd, packed):
            raise ValueError("DMD is not exact chronological pack/repeat3/pad12")
    for key in MASKS:
        rec = bank["masks"][key]
        if Path(rec["path"]).parent != bank_dir or rec["sha256"] != design["masks"][key]["sha256"]:
            raise ValueError("fixed mask label/source identity differs")
        mask = _array(rec["path"], bindings, rec["sha256"])
        if mask.shape != (600, 600) or mask.dtype != np.uint8:
            raise ValueError("native600 uint8 command, no resize/LUT/flip, required")
    legacy = (Path(legacy_root) if legacy_root else ROOT.parent / "ballplate_fourframe96_deploy").resolve(strict=True)
    _parent_bindings(legacy, bindings)
    for p in (Path(__file__), Path(sim.__file__)):
        _read(p, bindings)
    return dict(schema=SCHEMA, bank_directory=str(bank_dir), bank_sha256=expected_bank_sha256,
                bank_json_sha256=manifest["file_sha256"]["BANK.json"], legacy_root=str(legacy),
                models={"M0_REFERENCE": dict(path=str(legacy / sim.REFERENCE), sha256=sim.FIXED[sim.REFERENCE]),
                        "M1_RADIAL": dict(path=str(legacy / sim.WAVEFIT), sha256=sim.FIXED[sim.WAVEFIT])},
                inputs=bank["inputs"], masks=bank["masks"], samples=[2, 3], mode="mean",
                native_shape=[554, 552], readout_path=str(legacy / sim.REGIONS),
                readout_sha256=sim.FIXED[sim.REGIONS], parameter_names=list(PARAMETERS),
                parameter_nominal=list(NOMINAL), central_difference_steps=list(STEPS),
                sensitivity_inputs=[k for k in bank["inputs"] if k.startswith("R2_")],
                sensitivity_parent="M0_REFERENCE_only", source_and_asset_sha256=bindings,
                sensitivity_is_NOT_state_velocity_observability=True,
                parameter_steps_are_research_perturbations_NOT_measured_confidence_bounds=True,
                unqualified=UNRELEASED)


def summarize_jacobian(jacobian):
    """Column scaling uses the declared FD steps; no data-fitted whitening."""
    j = np.asarray(jacobian, np.float64)
    if j.ndim != 3 or j.shape[1:] != (8, 4) or not np.isfinite(j).all():
        raise ValueError("finite cells x eight sums x four physical parameters required")
    scaled = (j * np.asarray(STEPS)[None, None, :]).reshape(-1, 4)
    _, singular, right = np.linalg.svd(scaled, full_matrices=False)
    norm = np.linalg.norm(scaled, axis=0)
    denom = norm[:, None] * norm[None, :]
    correlation = np.divide(scaled.T @ scaled, denom, out=np.zeros((4, 4)), where=denom > 0)
    return dict(singular_values_parameter_step_scaled_raw8=singular.tolist(),
                numerical_rank_rtol_1e_minus6=int(np.count_nonzero(singular > (singular[0]*1e-6 if singular[0] else 0))),
                condition_number=(float(singular[0]/singular[-1]) if singular[-1] > 0 else None),
                column_cosines=correlation.tolist(), weakest_right_singular_vector=right[-1].tolist(),
                parameter_names=list(PARAMETERS), finite_difference_steps=list(STEPS),
                rank_is_NOT_empirical_identifiability_or_state_observability=True,
                noise_whitening_used=False, confidence_interval_claimed=False)


def _engines(plan):
    """CPU imports only; no original hardware modules or actuator interfaces."""
    sys.path.insert(0, plan["legacy_root"])
    import torch
    from workbench.optical_retrain.physics import ResearchPhysics
    from workbench.optical_retrain.input_wavefront import InputWavefrontResearchPhysics, load_input_wavefront
    from workbench.optical_retrain.uncertainty import UncertainResearchPhysics
    from workbench.twin.model import gaussian_code_mean
    from route1_fourier_v4.model import extrapolate
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU child must not initialize CUDA")

    class GlobalModel(InputWavefrontResearchPhysics, UncertainResearchPhysics):
        # Shared scalar SENSOR parameters act before ADC; not post-CCD edits.
        def sensor(self, intensity, mode="mean", generator=None):
            if mode != "mean":
                raise ValueError("this sensitivity study predicts means only")
            if self.global_gain_multiplier == 1. and self.global_bias == 0.:
                return super().sensor(intensity, mode, generator)
            signal = intensity * self.adu_gain * self.global_gain_multiplier
            mu = signal + self.latent_dark + self.global_bias
            sigma = torch.sqrt(signal.clamp_min(0)*self.noise["effective_signal_variance_slope_ADU"]
                               + self.noise["effective_latent_sigma_ADU"]**2)
            return gaussian_code_mean(mu, sigma)

    class Engine:
        def __init__(self, model, sensitivity=False):
            self.model, self.sensitivity = model.eval().requires_grad_(False), sensitivity
            self.last_key = self.fields = None
            self.input_key, self.native_cache = None, {}
        def manifest(self):
            return self.model.manifest()
        def predict(self, raw, mask, theta=NOMINAL):
            if not self.sensitivity and tuple(theta) != NOMINAL:
                raise ValueError("frozen nominal parent cannot be changed")
            z, q, gain, bias = map(float, theta)
            input_key = _digest(raw.tobytes())
            if self.input_key != input_key:
                self.input_key, self.native_cache = input_key, {}
                self.fields = self.last_key = None
            key = (input_key, z)
            optical_key = (z, q, _digest(mask.tobytes()))
            with torch.no_grad():
                if self.sensitivity:
                    self.model.set_equivalent_defocus_m(z)
                    self.model.set_nuisance(0., q)
                    self.model.global_gain_multiplier, self.model.global_bias = gain, bias
                if optical_key not in self.native_cache:
                    if key != self.last_key:
                        self.fields = self.model.cache_fields(torch.from_numpy(raw)[None])
                        self.last_key = key
                    gray = torch.from_numpy(mask).float()
                    values = [self.model._one_grid(field, gray, stage, static)
                              for field, stage, static in zip(self.fields, self.model.stages,
                                                           (self.model.low_static_phase, self.model.high_static_phase))]
                    self.native_cache[optical_key] = self.model.projection(
                        extrapolate(values[0], values[1], *self.model.samples).clamp_min(0))
                intensity = self.native_cache[optical_key]
                ccd = self.model.sensor(intensity, "mean")[0]
                features = self.model.readout.features(ccd[None])[0]
            return ccd.cpu().numpy(), features.cpu().numpy()

    for label in (*MODELS, "SENSITIVITY_M0"):
        if label == "M1_RADIAL":
            p = plan["models"][label]
            model = load_input_wavefront(p["path"], p["sha256"], device="cpu", trainable=False, samples=(2, 3))
        else:
            p = plan["models"]["M0_REFERENCE"]
            cls = GlobalModel if label == "SENSITIVITY_M0" else ResearchPhysics
            model = cls(samples=(2, 3), device="cpu", reference_path=p["path"], reference_sha256=p["sha256"])
            if label == "SENSITIVITY_M0":
                model.global_gain_multiplier, model.global_bias = 1., 0.
        if (model.native_shape != (554, 552) or model.offset != (0, -10)
                or model.spec["readout_path"] != plan["readout_path"]
                or any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers()))):
            raise ValueError("frozen placement/BROAD8/native shape/CPU family differs")
        yield label, Engine(model, label == "SENSITIVITY_M0")
        del model
        gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU study unexpectedly initialized CUDA")


def _save_array(out, name, value):
    path = out / name
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return dict(path=name, sha256=_digest(path.read_bytes()))


def _compute(plan, out, engines):
    """Full body is injectable only by tests; production uses _engines above."""
    start = time.monotonic()
    bindings = dict(plan["source_and_asset_sha256"])
    raw = {k: _array(v["raw96"]["path"], bindings, v["raw96"]["sha256"]) for k, v in plan["inputs"].items()}
    masks = {k: _array(v["path"], bindings, v["sha256"]) for k, v in plan["masks"].items()}
    regions = _array(plan["readout_path"], bindings, plan["readout_sha256"]).astype(float)
    rows, sensitivities, all_jacobians, labels = [], [], [], []
    def stop():
        if time.monotonic()-start > plan["maximum_seconds"]-10:
            raise TimeoutError("CPU prediction time bound; partial is not a complete archive")
    def predict(engine, image, mask, theta=NOMINAL):
        stop(); ccd, f = engine.predict(image, mask, theta)
        ccd, f = np.asarray(ccd), np.asarray(f)
        if (ccd.shape != (554, 552) or ccd.dtype != np.float32 or not np.isfinite(ccd).all()
                or f.shape != (8,) or not np.isfinite(f).all() or ccd.min() < 0 or ccd.max() > 255
                or not np.allclose(np.einsum("khw,hw->k", regions, ccd.astype(float)), f, rtol=0, atol=1e-6)):
            raise ValueError("native mean CCD/raw8 prediction integrity differs")
        return ccd, f.astype(np.float64)
    for label, engine in engines:
        labels.append(label)
        if label not in (*MODELS, "SENSITIVITY_M0") or labels != list((*MODELS, "SENSITIVITY_M0")[:len(labels)]):
            raise ValueError("independent exact M0/M1/sensitivity model order required")
        freeze = dict(model=label, model_manifest=engine.manifest(),
                      source_sha256=sim._loaded_sources(plan["legacy_root"]),
                      frozen_before_this_model_first_input_field_propagation=True,
                      four_parameter_nominal=list(NOMINAL), four_parameter_steps=list(STEPS))
        _check(bindings)
        _write(out / (label + "_PRE_PREDICTION_FREEZE.json"), freeze)
        for ik, image in raw.items():
            if label == "SENSITIVITY_M0" and ik not in plan["sensitivity_inputs"]:
                continue
            for mk in MASKS:
                ccd, f = predict(engine, image, masks[mk])
                stem = label + "__" + ik + "__" + mk
                if label in MODELS:
                    rows.append(dict(model=label, input_id=ik, mask_id=mk,
                        files=dict(ccd=_save_array(out, stem + "_CCD.npy", ccd),
                                   native8=_save_array(out, stem + "_RAW8.npy", f)),
                        mean_CCD_max_ADU=float(ccd.max()), predicted_max_below_capture_limit240=bool(ccd.max() < 240)))
                    continue
                parent = next(r for r in rows if r["model"] == "M0_REFERENCE" and r["input_id"] == ik and r["mask_id"] == mk)
                reference = _array(out / parent["files"]["ccd"]["path"], {}, parent["files"]["ccd"]["sha256"])
                if not np.array_equal(ccd, reference):
                    raise ValueError("zero perturbation must exactly reproduce frozen M0 CCD")
                j8, ji, variants = [], [], []
                for j, step in enumerate(STEPS):
                    pair = []
                    for sign in (-1, 1):
                        theta = list(NOMINAL); theta[j] += sign*step
                        pc, pf = predict(engine, image, masks[mk], theta)
                        tag = stem + "__P" + str(j) + ("_minus" if sign < 0 else "_plus")
                        variants.append(dict(parameter=PARAMETERS[j], theta=theta,
                            files=dict(ccd=_save_array(out, tag + "_CCD.npy", pc),
                                       native8=_save_array(out, tag + "_RAW8.npy", pf))))
                        pair.append((pc, pf))
                    j8.append((pair[1][1]-pair[0][1])/(2*step))
                    ji.append((pair[1][0].astype(float)-pair[0][0].astype(float))/(2*step))
                j8 = np.stack(j8, axis=-1); ji = np.stack(ji, axis=0)
                sensitivities.append(dict(input_id=ik, mask_id=mk, parent="M0_REFERENCE",
                    zero_perturbation_exact_parent=True, variants=variants,
                    jacobian8=_save_array(out, stem + "_J8.npy", j8),
                    jacobian_CCD=_save_array(out, stem + "_JCCD.npy", ji.astype(np.float32)),
                    summary=summarize_jacobian(j8[None])))
                all_jacobians.append(j8)
    if labels != [*MODELS, "SENSITIVITY_M0"] or len(rows) != 108 or len(sensitivities) != 27:
        raise ValueError("complete108 nominal/27 sensitivity command denominators required")
    _check(bindings)
    result = dict(schema=SCHEMA, complete=True, bank_sha256=plan["bank_sha256"],
                  nominal_prediction_count=len(rows), sensitivity_command_count=len(sensitivities),
                  rows=rows, sensitivities=sensitivities,
                  global_physical_parameter_sensitivity=summarize_jacobian(np.asarray(all_jacobians)),
                  actual_hardware_frames=0, hardware_accessed=False, device="CPU", samples=[2, 3],
                  maximum_nominal_CCD_ADU=max(r["mean_CCD_max_ADU"] for r in rows),
                  all_nominal_means_below_capture_limit240=all(r["predicted_max_below_capture_limit240"] for r in rows),
                  model_freezes_precede_each_model_propagation=True,
                  no_parameters_fitted=True, no_E003_validation_outcomes_loaded=True,
                  source_and_assets_unchanged=True, wall_seconds=time.monotonic()-start,
                  limitations=["model-only finite-difference parameter sensitivity, NOT measured state observability",
                               "M0/M1 retain old unqualified response/pupil/registered scale/noise priors",
                               "no measured noise whitening or confidence interval; no rank-based release",
                               "raw command once, historical response once, model direction once; no new LUT"],
                  unqualified=UNRELEASED)
    _write(out / "WORKER_RESULT.json", result)
    return result


def _worker_main(protocol_path):
    plan = json.loads(Path(protocol_path).read_bytes())
    _check(dict(plan["source_and_asset_sha256"]))
    _compute(plan, Path(plan["output"]), _engines(plan))


def _run_worker(protocol_path, maximum_seconds):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="",
               OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    script = "from optical_lab.identification import _worker_main; import sys; _worker_main(sys.argv[1])"
    with (protocol_path.parent / "WORKER.log").open("xb") as log:
        subprocess.run([sys.executable, "-c", script, str(protocol_path)], env=env, cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, timeout=maximum_seconds, check=True)


def prepare_sensitivity(bank_dir, output, *, expected_bank_sha256, execute=False,
                        maximum_seconds=900, legacy_root=None):
    """expected_bank_sha256 is the external BANK MANIFEST SHA, not BANK.json.

    Default validates only; explicit execution writes a NEW archive. Timeout or
    any integrity failure leaves FAILED/partial files and no completed seal.
    """
    if type(execute) is not bool or not math.isfinite(maximum_seconds) or not 30 <= maximum_seconds <= 1800:
        raise ValueError("explicit execute boolean and bounded30..1800 CPU seconds required")
    plan = _preflight(bank_dir, expected_bank_sha256, legacy_root)
    if not execute:
        return dict(schema=SCHEMA, complete=False, validated=True, propagation_executed=False,
                    bank_sha256=expected_bank_sha256, planned_nominal_predictions=108,
                    planned_sensitivity_commands=27, output_created=False, hardware_accessed=False)
    out = Path(output).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError("new output directory required; no overwrite/resume")
    if out.is_relative_to(Path(plan["bank_directory"])):
        raise ValueError("output cannot modify input bank")
    out.mkdir(exist_ok=False)
    plan.update(output=str(out), maximum_seconds=float(maximum_seconds), execute_requested=True)
    try:
        source = _read(__file__, plan["source_and_asset_sha256"])
        with (out / "identification_source.py").open("xb") as stream: stream.write(source)
        _write(out / "PROSPECTIVE_PROTOCOL.json", plan)
        _run_worker(out / "PROSPECTIVE_PROTOCOL.json", maximum_seconds)
        result = json.loads((out / "WORKER_RESULT.json").read_bytes())
        if result.get("complete") is not True or result["schema"] != SCHEMA or result["bank_sha256"] != expected_bank_sha256:
            raise ValueError("complete bound CPU worker result required")
        _check(plan["source_and_asset_sha256"])
        if (out / "identification_source.py").read_bytes() != source:
            raise ValueError("frozen prediction source snapshot changed")
        result["protocol_sha256"] = _digest((out / "PROSPECTIVE_PROTOCOL.json").read_bytes())
        _write(out / "RESULT.json", result)
        _write(out / "MANIFEST.json", dict(schema=SCHEMA, complete=True,
            file_sha256={str(p.relative_to(out)): _digest(p.read_bytes()) for p in out.rglob("*") if p.is_file()},
            manifest_excludes_itself=True, hardware_accessed=False, production_qualified=False))
        return result
    except Exception as exc:
        _write(out / "FAILED.json", dict(schema=SCHEMA, complete=False, error=repr(exc),
                                        partial_files_preserved=True, hardware_accessed=False))
        raise
