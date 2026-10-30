"""Explicit CPU-only E002 references; default is identity validation, not propagation.

Only M0 frozen reference ba9d... and M1 frozen radial fit a67d... are supported.
Old Torch/model imports occur ONLY in an isolated explicit-execution child.
This is not a calibrated twin, new LUT, hardware command, fit, or release.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from .probes import verify_pilot_assets


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "optical-lab-fixed-pilot-cpu-predictions-1"
CASE_SCHEMA = "optical-lab-static-case-cpu-predictions-1"
PILOT_MANIFEST_SHA = "0e829e0dce87e4e8010b75c7baca0138886d8d228767a1cad69e9a8a81c87f3c"
RAW = "outputs/platform_twin_20261022/FRESH_DOMAIN_1/T01_S18_RAW96.npy"
DMD = "outputs/platform_twin_20261022/FRESH_DOMAIN_1/T01_S18_DMD.npy"
REFERENCE = "outputs/platform_twin_20261022/ZERO_REFERENCE_IDENTIFICATION_1/FROZEN_REFERENCE_STATE.json"
WAVEFIT = "outputs/optical_retrain_20261022/INPUT_WAVEFRONT_WIDE_8UM_CPU_1/FROZEN_INPUT_WAVEFRONT.json"
REGIONS = "outputs/readout_coverage_20261022_v34/BROAD_NATIVE8.npy"
HEAD = "outputs/live_calibration/INTERACTIVE_LOOP_V23_20261021_1930/GUI/single_frame_20261022_114123_404191/HEAD.json"
CONTRACT = "outputs/learned_real_control_20261022_v31/BUNDLE/CONTRACT.json"
FIXED = {
    RAW: "f300698b5f84ef76412c2bdcdca4595601acaf9ab58d16289c20136bb7384c6b",
    DMD: "74d0cd8a5396bd4c7a143fa6ac6c9c341ec98e81fc81c3b4fadeb463f1aa2b52",
    REFERENCE: "ba9d9f69b01301fb78a8e7365a46f4f200f8612fb9b733ad70efc572184afa52",
    WAVEFIT: "a67d37a64d52f7bfddf928ebd66e53286048f03a7176dc15ae37122413ca92a0",
    REGIONS: "0a76e770d8d9f2761a9edce492b4a846eab98ebf0e6f719271711202954601c5",
    HEAD: "7285a406adba4ad852999c577471d3e2d9d762f72b42ce0197091f7977df319e",
    CONTRACT: "59c833b91ac4b1c3395905e12a89c918fa994294fc3ad89880eeed886589b992",
    "workbench/optical_retrain/physics.py": "68526de9f1440f9c2a781c6ef23e7bbef1cb2b92e6cc5875e5a936051b8879b3",
    "workbench/optical_retrain/input_wavefront.py": "1fc508dcce8da33f864b0b4fd5be9ba5254f42a928d0b39e4204ea76adac0637",
}
MODEL_IDS = ("M0_REFERENCE", "M1_RADIAL")
MASK_IDS = ("Z", "A", "B")
RELEASE = {"hardware_accessed": False, "projection_executed": False, "captured_frames": 0,
           "new_LUT_measured": False, "parameters_fitted": False, "placement_changed": False,
           "hardware_release_approved": False, "control_release_approved": False,
           "production_qualified": False}
INPUT_PLAN_SCHEMA = "optical-lab-prescribed-input-plan-1"
INPUT_SCHEMA = "optical-lab-prescribed-input-bank-1"
INPUT_MODE = "prescribed_pose_sequence_NOT_free_dynamics"
_INPUT_ENV_SOURCES = {
    "env/ballplate_env.py": "61b882e601441500793addce02bc72eb6fcc0d4668af0eb9d4e3e1a67b79cc42",
    "env/ball_camera.py": "751e781946be72f3656027e58d138daa07c16dbc683cde3024aaf32780b0ab32",
    "env/reward.py": "d7cd0a1cca3858306e8377d9262444c6b61f2834a64104e6a9e4fe81ddbc6b70",
    "config.py": "6af104511eb1e21e5290efd926e4d7378dacd1fd6dd92860a1cc9802ad950b96",
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_INPUT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_bytes())


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _check(bindings):
    for name, expected in bindings.items():
        if _sha(name) != expected:
            raise ValueError("bound source/asset changed: " + name)


def _case_identity(directory, manifest_sha256):
    # Explicit alternative asset family; the frozen E002 pilot verifier and
    # its fixed external SHA are never relaxed or monkeypatched.
    from .study import verify_case_assets
    from . import study
    checked = verify_case_assets(directory, expected_manifest_sha256=manifest_sha256)
    manifest = checked["manifest"]
    if checked["verified"] is not True or checked["manifest_sha256"] != manifest_sha256:
        raise ValueError("explicit case asset verification failed")
    if manifest["schema"] != "optical-lab-static-case-assets-1":
        raise ValueError("explicit static case asset schema required")
    return checked, {"case_asset_manifest": {"path": str(directory), "sha256": manifest_sha256},
        "case_id": manifest["design"]["case_id"], "role": manifest["design"]["role"],
        "design": manifest["design"], "fixed_input": manifest["fixed_input"],
        "probe_spec": manifest["probe_spec"]}, Path(study.__file__).resolve()


def _preflight(pilot_dir, legacy_root, *, case_manifest_sha256=None):
    pilot = Path(pilot_dir).resolve(strict=True)
    case, case_source = None, None
    if case_manifest_sha256 is None:
        verified = verify_pilot_assets(pilot, expected_manifest_sha256=PILOT_MANIFEST_SHA)
    else:
        verified, case, case_source = _case_identity(pilot, case_manifest_sha256)
    legacy = (Path(legacy_root) if legacy_root is not None else ROOT.parent / "ballplate_fourframe96_deploy").resolve(strict=True)
    refs = {str(legacy / name): sha for name, sha in FIXED.items() if case is None or name not in (RAW, DMD)}
    if case is not None:
        for path_key, sha_key in (("raw_path", "raw_sha256"), ("dmd_path", "dmd_sha256")):
            refs[case["fixed_input"][path_key]] = case["fixed_input"][sha_key]
        if case["fixed_input"]["history_ages_s"] != [.18, .12, .06, 0.]:
            raise ValueError("same chronological sparse60 RAW history required")
        refs[case["design"]["path"]] = case["design"]["sha256"]
        refs[str(case_source)] = _sha(case_source)
    _check(refs)
    reference, fit = _json(legacy / REFERENCE), _json(legacy / WAVEFIT)
    # Frozen old records contain absolute dependency paths. Never silently
    # rewrite them on relocation, or select a different/newest model.
    if fit["reference_path"] != str(legacy / REFERENCE) or fit["reference_sha256"] != FIXED[REFERENCE]:
        raise ValueError("frozen legacy absolute reference layout differs; relocation is not automatic")
    if (fit.get("completed") is not True or fit.get("selection_frozen") is not True
            or fit.get("error") is not None or fit["wavefront_kind"] != "radial"
            or fit["physical_DMD_micromirror_pitch_m"] is not None or fit["samples"] != [2, 3]):
        raise ValueError("exact completed registered8um radial hypothesis required")
    for table in (reference["sources"], fit["source_sha256"]):
        for path, sha in table.items():
            if path in refs and refs[path] != sha:
                raise ValueError("conflicting old source identity")
            refs[path] = sha
    refs[fit["protocol_path"]] = fit["protocol_sha256"]
    refs[fit["selection_path"]] = fit["selection_sha256"]
    physical = fit["physical_manifest"]
    for path_key, sha_key in (("native_readout_regions_path", "native_readout_regions_sha256"),
                              ("frozen_original_head_path", "frozen_original_head_sha256"),
                              ("historical_raw_gray_response_path", "historical_raw_gray_response_sha256")):
        refs[physical[path_key]] = physical[sha_key]
    if (physical["native_readout_regions_path"] != str(legacy / REGIONS)
            or physical["frozen_original_head_path"] != str(legacy / HEAD)
            or physical["native_shape"] != [554, 552]):
        raise ValueError("old native ROI/BROAD8/head family required; not W7 XYXY")
    for path in (Path(__file__).resolve(), Path(__file__).with_name("probes.py").resolve()):
        refs[str(path)] = _sha(path)
    for path in pilot.iterdir():
        refs[str(path)] = _sha(path)
    _check(refs)
    result = {"schema": SCHEMA if case is None else CASE_SCHEMA, "pilot_directory": str(pilot),
            "pilot_manifest_sha256": verified["manifest_sha256"],
            "legacy_root": str(legacy), "source_and_asset_sha256": refs,
            "models": {"M0_REFERENCE": {"path": str(legacy / REFERENCE), "sha256": FIXED[REFERENCE]},
                       "M1_RADIAL": {"path": str(legacy / WAVEFIT), "sha256": FIXED[WAVEFIT]}},
            "fixed_input": {"id": "T01_S18", "raw_path": str(legacy / RAW), "raw_sha256": FIXED[RAW],
                "dmd_path": str(legacy / DMD), "dmd_sha256": FIXED[DMD],
                "selection_rule": "first_nonzero_input_in_E001_declared_cell_order_not_by_performance",
                "history_ages_s": [.18, .12, .06, 0.]},
            "samples": [2, 3], "sensor_mode": "mean", "native_shape": [554, 552],
            "mask_use": "absolute_canonical600_raw_gray_NOT_old_mask_plus_residual_no_export_LUT_or_flip",
            "readout_family": "ORIGINAL_BROAD_NATIVE8_and_original_diagnostic_head_NOT_W7",
            "model_choice_fixed_before_predictions": True, "parameters_refitted_after_real_capture": False,
            "G2_model_qualified": False, "unreleased": dict(RELEASE)}
    if case is not None:
        result.update(case)
        del result["pilot_manifest_sha256"]
        result["model_choice_fixed_before_predictions"] = True
    return result


def _loaded_sources(legacy):
    """Project Python implementation identities only, not synthetic Torch names."""
    roots = (Path(legacy).parent.resolve(), ROOT.resolve())
    result = {}
    for module in list(sys.modules.values()):
        value = getattr(module, "__dict__", {}).get("__file__")
        if not isinstance(value, str) or not Path(value).is_absolute():
            continue
        path = Path(value).resolve()
        if path.suffix == ".py" and any(path.is_relative_to(root) for root in roots):
            if not path.is_file():
                raise ValueError("loaded project source missing: " + str(path))
            result[str(path)] = _sha(path)
    return result


def _legacy_predictors(plan, raw):
    """The only old/Torch import site. Called only inside explicit CPU child."""
    sys.path.insert(0, plan["legacy_root"])
    import torch
    from workbench.optical_retrain.physics import ResearchPhysics
    from workbench.optical_retrain.input_wavefront import load_input_wavefront
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only child unexpectedly initialized CUDA")

    class Predictor:
        def __init__(self, model):
            self.model, self.fields = model.eval().requires_grad_(False), None
        def manifest(self):
            return self.model.manifest()
        def predict(self, gray):
            with torch.no_grad():
                if self.fields is None:
                    self.fields = self.model.cache_fields(torch.from_numpy(raw)[None])
                prediction = self.model.predict_from_fields(self.fields, torch.from_numpy(gray).float(), mode="mean")
            return prediction.ccd[0].detach().cpu().numpy(), prediction.features[0].detach().cpu().numpy()

    for label in MODEL_IDS:
        item = plan["models"][label]
        if label == "M0_REFERENCE":
            model = ResearchPhysics(samples=(2, 3), device="cpu", reference_path=item["path"], reference_sha256=item["sha256"])
        else:
            model = load_input_wavefront(item["path"], item["sha256"], device="cpu", trainable=False, samples=(2, 3))
        if any(value.device.type != "cpu" for value in [*model.parameters(), *model.buffers()]):
            raise ValueError("all optical model tensors must stay on CPU")
        if (model.native_shape != (554, 552) or model.offset != (0, -10)
                or model.spec["readout_path"] != str(Path(plan["legacy_root"]) / REGIONS)
                or model.spec["head_path"] != str(Path(plan["legacy_root"]) / HEAD)):
            raise ValueError("frozen placement/native/BROAD8/head differs")
        predictor = Predictor(model)
        yield label, predictor
        del predictor, model
        gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only child initialized CUDA")


def _worker_main(protocol_path):
    """Fixed real body also exercised by NumPy toy predictors in unit tests."""
    protocol_path = Path(protocol_path).resolve(strict=True)
    plan = _json(protocol_path)
    if plan["schema"] not in (SCHEMA, CASE_SCHEMA) or plan["execution_backend"] != "isolated_CPU_threads2_no_CUDA":
        raise ValueError("explicit fixed CPU execution protocol required")
    if plan["schema"] == CASE_SCHEMA:
        _, current, _ = _case_identity(Path(plan["case_asset_manifest"]["path"]), plan["case_asset_manifest"]["sha256"])
        if _canonical(current) != _canonical({k: plan[k] for k in current}):
            raise ValueError("frozen case identity differs before propagation")
    _check(plan["source_and_asset_sha256"])
    out, legacy = Path(plan["output_directory"]), Path(plan["legacy_root"])
    raw = np.load(plan["fixed_input"]["raw_path"], allow_pickle=False)
    dmd = np.load(plan["fixed_input"]["dmd_path"], allow_pickle=False)
    if raw.shape != (4, 1, 96, 96) or raw.dtype != np.uint8 or not np.isin(raw, [0, 255]).all():
        raise ValueError("fixed RAW96 history must be exact binary uint8")
    mosaic = np.block([[raw[0, 0], raw[1, 0]], [raw[2, 0], raw[3, 0]]])
    if not np.array_equal(dmd, np.pad(mosaic.repeat(3, 0).repeat(3, 1), ((12, 12), (12, 12)))):
        raise ValueError("fixed DMD does not match chronological RAW packing")
    regions = np.load(legacy / REGIONS, allow_pickle=False)
    if regions.shape != (8, 554, 552) or not np.isin(regions, [0, 1]).all() or not np.all(regions.sum(0) == 1):
        raise ValueError("original BROAD8 must be a full native binary partition")
    rows, freezes, seen = [], {}, []
    for label, predictor in _legacy_predictors(plan, raw):
        if label != MODEL_IDS[len(seen)]:
            raise ValueError("fixed model order differs")
        seen.append(label)
        before = predictor.manifest()
        loaded = _loaded_sources(legacy)
        _check(loaded)
        freeze_file = f"{label}_PRE_PREDICTION_FREEZE.json"
        _write(out / freeze_file, {"model_manifest": before, "loaded_source_sha256": loaded,
            "absolute_masks": list(MASK_IDS), "frozen_before_first_input_field_propagation": True})
        freezes[label] = {"path": freeze_file, "sha256": _sha(out / freeze_file)}
        for mask in MASK_IDS:
            gray = np.load(Path(plan["pilot_directory"]) / f"SLM_{mask}_CANONICAL600_RAW_GRAY_NO_LUT.npy", allow_pickle=False)
            ccd, features = predictor.predict(gray)
            if ccd.shape != (554, 552) or not np.issubdtype(ccd.dtype, np.floating) or not np.isfinite(ccd).all() or np.any((ccd < 0) | (ccd > 255)):
                raise ValueError("finite native expected-code ADU prediction required")
            direct = np.einsum("khw,hw->k", regions.astype(np.float64), ccd.astype(np.float64))
            if features.shape != (8,) or not np.isfinite(features).all() or not np.allclose(features, direct, rtol=1e-12, atol=1e-7):
                raise ValueError("native8 prediction must use the original BROAD8 directly")
            files = {}
            for suffix, value in (("NATIVE_CCD_MEAN_ADU", ccd), ("NATIVE8_FEATURES", features)):
                name = f"{label}_{mask}_{suffix}.npy"
                with (out / name).open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                files[suffix] = {"path": name, "sha256": _sha(out / name)}
            rows.append({"model": label, "mask": mask, "files": files,
                         "units": "expected-code ADU and raw ADU sums; NOT real CCD"})
        if _canonical(predictor.manifest()) != _canonical(before):
            raise ValueError("model manifest/parameters changed during prediction")
        _check(loaded)
        _check(plan["source_and_asset_sha256"])
    if tuple(seen) != MODEL_IDS:
        raise ValueError("both fixed models must complete")
    _write(out / "WORKER_RESULT.json", {"schema": plan["schema"], "completed": True, "rows": rows,
        "model_freezes": freezes, "source_and_assets_unchanged": True,
        "CUDA_initialized": False, "unreleased": dict(RELEASE)})


def _run_worker(protocol_path, maximum_seconds, *, input_generation=False):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    function = "_input_worker_main" if input_generation else "_worker_main"
    if input_generation:
        env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LIBGL_ALWAYS_SOFTWARE="1")
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    script = f"import sys; from optical_lab.simulate import {function}; {function}(sys.argv[1])"
    log_name = "INPUT_WORKER.log" if input_generation else "WORKER.log"
    with (protocol_path.parent / log_name).open("xb") as log:
        subprocess.run([sys.executable, "-c", script, str(protocol_path)], env=env,
                       cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                       timeout=maximum_seconds, check=True)


def prepare_predictions(pilot_dir, output, legacy_root=None, *, execute=False, maximum_seconds=180,
                        case_manifest_sha256=None):
    """Default: validate identities and return a plan WITHOUT creating output.

    Explicit ``execute=True`` freezes a new output, then uses this interpreter
    in an isolated CPU child (therefore execution needs the existing Torch
    environment). No model source or physical mapping is rewritten. Failed
    or timed-out attempts retain their partial files and FAILED.json.
    """
    if type(execute) is not bool or not math.isfinite(maximum_seconds) or not 0 < maximum_seconds <= 600:
        raise ValueError("explicit execution bool and CPU bound in (0,600] seconds required")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("new prediction output required")
    plan = (_preflight(pilot_dir, legacy_root) if case_manifest_sha256 is None else
            _preflight(pilot_dir, legacy_root, case_manifest_sha256=case_manifest_sha256))
    plan.update(output_directory=str(output.resolve()), execution_backend="isolated_CPU_threads2_no_CUDA",
                maximum_seconds=float(maximum_seconds), execute_requested=execute,
                interpreter=str(Path(sys.executable).absolute()))
    if not execute:
        return {"schema": plan["schema"], "completed": False, "metadata_validated": True,
                "propagation_executed": False, "output_created": False, "plan": plan,
                "unreleased": dict(RELEASE)}
    started = time.monotonic()
    output.mkdir(exist_ok=False)
    protocol_path = output.resolve() / "PROSPECTIVE_PROTOCOL.json"
    _write(protocol_path, plan)
    protocol_sha = _sha(protocol_path)
    try:
        _run_worker(protocol_path, float(maximum_seconds))
        _check(plan["source_and_asset_sha256"])
        if _sha(protocol_path) != protocol_sha:
            raise ValueError("pre-prediction protocol changed")
        worker = _json(output / "WORKER_RESULT.json")
        if (worker["schema"] != plan["schema"] or worker["completed"] is not True
                or worker["CUDA_initialized"] is not False or worker["unreleased"] != RELEASE
                or worker.get("source_and_assets_unchanged") is not True
                or [(row["model"], row["mask"]) for row in worker["rows"]] != [(m, g) for m in MODEL_IDS for g in MASK_IDS]):
            raise ValueError("complete fixed six CPU predictions required")
        expected_files = {"PROSPECTIVE_PROTOCOL.json", "WORKER.log", "WORKER_RESULT.json"}
        for row in worker["rows"]:
            if set(row["files"]) != {"NATIVE_CCD_MEAN_ADU", "NATIVE8_FEATURES"}:
                raise ValueError("exact CCD and native8 output pair required")
            for suffix, item in row["files"].items():
                expected = f"{row['model']}_{row['mask']}_{suffix}.npy"
                path = output / expected
                if item["path"] != expected or path.is_symlink() or _sha(path) != item["sha256"]:
                    raise ValueError("prediction output identity differs")
                expected_files.add(expected)
        for label in MODEL_IDS:
            item = worker["model_freezes"][label]
            expected = f"{label}_PRE_PREDICTION_FREEZE.json"
            path = output / expected
            if item["path"] != expected or path.is_symlink() or _sha(path) != item["sha256"]:
                raise ValueError("pre-prediction model freeze differs")
            expected_files.add(expected)
        if ({p.name for p in output.iterdir()} != expected_files
                or any(p.is_symlink() or not p.is_file() for p in output.iterdir())):
            raise ValueError("unexpected or non-regular CPU prediction output")
        result = {"schema": plan["schema"], "completed": True, "propagation_executed": True,
            "backend": "CPU_threads2_no_CUDA", "protocol_path": str(protocol_path), "protocol_sha256": protocol_sha,
            "worker_result_sha256": _sha(output / "WORKER_RESULT.json"), "wall_seconds": time.monotonic() - started,
            "rows": worker["rows"], "model_freezes": worker["model_freezes"],
            "source_and_assets_unchanged": True, "G2_model_qualified": False, "unreleased": dict(RELEASE),
            "limitations": ["historical fitted physics remains unqualified", "new probe support is 512 inside 600",
                "phase response/window/native scale are historical hypotheses", "not real CCD or control performance",
                "same existing input; NOT independent input-mask-time holdout", "no per-image alignment or gain refit"]}
        if case_manifest_sha256 is not None:
            _, identity, _ = _case_identity(Path(plan["case_asset_manifest"]["path"]), case_manifest_sha256)
            if _canonical(identity) != _canonical({k: plan[k] for k in identity}):
                raise ValueError("frozen case identity changed after propagation")
            result.update(identity, samples=plan["samples"], sensor_mode=plan["sensor_mode"],
                native_shape=plan["native_shape"], readout_family=plan["readout_family"])
            result["limitations"][4] = "input/probe roles are prospectively declared; no whole-state-distribution novelty claim"
        _write(output / "RESULT.json", result)
        files = {path.name: _sha(path) for path in output.iterdir() if path.is_file()}
        _write(output / "MANIFEST.json", {"schema": plan["schema"], "completed": True, "file_sha256": files,
            "manifest_excludes_itself": True, "unreleased": dict(RELEASE)})
        return result
    except Exception as exc:
        _write(output / "FAILED.json", {"schema": plan["schema"], "completed": False,
            "error": type(exc).__name__ + ": " + str(exc), "partial_files_preserved": True,
            "unreleased": dict(RELEASE)})
        raise


def _validate_input_design(design):
    if (design["schema"] != INPUT_PLAN_SCHEMA or design["mode"] != INPUT_MODE
            or design["control_dt_s"] != .02 or design["history_steps"] != [0, 3, 6, 9]
            or design["history_ages_s"] != [.18, .12, .06, 0.]
            or design.get("hardware_actions_enabled", False) is not False):
        raise ValueError("explicit prescribed pose plan and sparse60 chronology required")
    if not _INPUT_ID.fullmatch(design["id"]) or type(design["seed"]) is not int or not 0 <= design["seed"] < 2**32:
        raise ValueError("finite fixed plan id/seed required")
    rows = design["trajectories"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
        raise ValueError("one to 32 predeclared trajectories required")
    ids = []
    for row in rows:
        if set(row) != {"id", "role", "query_xy_m", "velocity_xy_m_s"}:
            raise ValueError("exact input trajectory fields required")
        if not _INPUT_ID.fullmatch(row["id"]) or row["role"] not in ("train", "select", "calibration", "selection", "validation", "test"):
            raise ValueError("predeclared trajectory id/role required")
        xy = np.asarray(row["query_xy_m"], dtype=np.float64)
        velocity = np.asarray(row["velocity_xy_m_s"], dtype=np.float64)
        if (xy.shape != (2,) or velocity.shape != (2,) or not np.isfinite(xy).all()
                or not np.isfinite(velocity).all() or np.linalg.norm(velocity) > .18):
            raise ValueError("finite 2D pose and speed <=0.18m/s required")
        points = xy[None] - (.18 - .02 * np.arange(10))[:, None] * velocity[None]
        if np.max(np.abs(points)) > .20:
            raise ValueError("all ten predeclared poses must lie within +/-0.20m per axis")
        ids.append(row["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("unique trajectory ids required")
    for asset in (design["xml"], design["seen_input_keys"]):
        if set(asset) != {"path", "sha256"} or not _HASH.fullmatch(asset["sha256"]):
            raise ValueError("explicit path and external SHA for input plan assets required")


def _input_preflight(plan_path, expected_plan_sha256, legacy_root):
    plan_path = Path(plan_path).resolve(strict=True)
    if not _HASH.fullmatch(expected_plan_sha256) or _sha(plan_path) != expected_plan_sha256:
        raise ValueError("input design external SHA differs")
    design = _json(plan_path)
    _validate_input_design(design)
    legacy = (Path(legacy_root) if legacy_root is not None else ROOT.parent / "ballplate_fourframe96_deploy").resolve(strict=True)
    environment = legacy.parent / "ballplate_optical_diffraction_ppo"
    refs = {str(environment / name): sha for name, sha in _INPUT_ENV_SOURCES.items()}
    refs.update({str(plan_path): expected_plan_sha256, str(Path(__file__).resolve()): _sha(__file__)})
    # Only the exact resize implementation is reused, never generate_fixture,
    # its forced EGL device, teacher labels, replay bank or old RAW loaders.
    resize_source = legacy / "model96.py"
    refs[str(resize_source)] = "e91db6b1f3e8c8b62f4b5df8a58d87bc42cd8ba7d352cfb90a380beaca42389e"
    refs[str(legacy / "runtime96.py")] = "8335fe377699651b1fcbbb9f6b71db1512fd9af9a87c84736dd4f563b627b756"
    for asset in (design["xml"], design["seen_input_keys"]):
        refs[asset["path"]] = asset["sha256"]
    _check(refs)
    seen = _json(design["seen_input_keys"]["path"])
    if (seen["schema"] != "optical-lab-seen-input-keys-1" or not isinstance(seen["keys"], list)
            or any(not isinstance(k, str) or not _HASH.fullmatch(k) for k in seen["keys"])
            or len(set(seen["keys"])) != len(seen["keys"])):
        raise ValueError("explicit unique SHA256 RAW-pixel keys required, not old RAW arrays")
    # Hash declared metadata catalogs only. The root separately prepared the
    # pixel-key index; do not deserialize old fixtures or reopen old RAW banks.
    for item in seen.get("catalogs", []):
        if not _HASH.fullmatch(item["sha256"]):
            raise ValueError("seen index catalog SHA required")
        refs[item["path"]] = item["sha256"]
    _check(refs)
    snapshots = {"generator_source.py": str(Path(__file__).resolve()),
        "model.xml": design["xml"]["path"], "seen_input_keys.json": design["seen_input_keys"]["path"],
        "RAW_resize_model96.py": str(resize_source), "RAW_runtime96.py": str(legacy / "runtime96.py")}
    snapshots.update({"ENV_" + name.replace("/", "_"): str(environment / name) for name in _INPUT_ENV_SOURCES})
    return {"schema": INPUT_SCHEMA, "design": design, "design_path": str(plan_path),
        "design_sha256": expected_plan_sha256, "legacy_root": str(legacy), "environment_root": str(environment),
        "source_and_asset_sha256": refs, "backend": "CPU_OSMesa_software_threads2_no_CUDA",
        "new_input_novelty": "new_prescribed_position_and_apparent_velocity_renderings_only_NOT_entire_state_distribution",
        "seen_check_scope": "exact_pixel_keys_against_declared_archive_index_only",
        "seen_key_derivation_reexecuted": False, "seen_archive_RAW_arrays_loaded": False,
        "snapshot_sources": snapshots,
        "no_teacher_or_action_labels": True, "no_old_fixture_deserialization": True,
        "no_scored_history_padding": True, "free_dynamics_rollout": False, "unreleased": dict(RELEASE)}


def _input_renderer(plan, seed):
    """Explicit OSMesa CPU child only. Ten independent pose renders, no step."""
    if os.environ.get("MUJOCO_GL") != "osmesa" or os.environ.get("PYOPENGL_PLATFORM") != "osmesa":
        raise ValueError("explicit CPU OSMesa rendering child required")
    sys.path.insert(0, plan["legacy_root"])
    import runtime96  # supplies the archived environment import path, not setup()
    import torch
    from env.ballplate_env import BallPlateEnv
    from model96 import resize_binary
    import inspect
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("input generation must not initialize CUDA")
    if Path(inspect.getfile(BallPlateEnv)).resolve() != Path(plan["environment_root"]) / "env/ballplate_env.py":
        raise ValueError("wrong archived environment family")
    env = BallPlateEnv(seed=seed, xml=Path(plan["design"]["xml"]["path"]).read_text(), duration=.20)
    if abs(env.control_dt - .02) > 1e-12:
        env.close()
        raise ValueError("archived environment cadence differs from declared 20ms")

    class Renderer:
        def render(self, point):
            image = env.reset(point)
            return resize_binary(image), env.position().tolist()
        def close(self):
            env.close()
    return Renderer()


def _input_worker_main(protocol_path):
    protocol_path = Path(protocol_path).resolve(strict=True)
    plan = _json(protocol_path)
    if plan["schema"] != INPUT_SCHEMA or plan["backend"] != "CPU_OSMesa_software_threads2_no_CUDA":
        raise ValueError("frozen explicit CPU input protocol required")
    _validate_input_design(plan["design"])
    _check(plan["source_and_asset_sha256"])
    output = protocol_path.parent
    protocol_identity = {"path": str(protocol_path), "sha256": _sha(protocol_path)}
    seen = set(_json(plan["design"]["seen_input_keys"]["path"])["keys"])
    rows = []
    loaded = None
    for index, spec in enumerate(plan["design"]["trajectories"]):
        renderer = _input_renderer(plan, plan["design"]["seed"] + index)
        try:
            if loaded is None:
                loaded = _loaded_sources(plan["legacy_root"])
                _check(loaded)
                _write(output / "RENDER_SOURCE_FREEZE.json", {"loaded_source_sha256": loaded,
                    "frozen_before_first_pose_render": True, "backend": plan["backend"]})
            images, actual = [], []
            requested = (np.asarray(spec["query_xy_m"])[None]
                - (.18 - .02 * np.arange(10))[:, None] * np.asarray(spec["velocity_xy_m_s"])[None])
            for point in requested:
                image, position = renderer.render(point)
                image = np.asarray(image)
                if image.shape != (1, 96, 96) or image.dtype != np.uint8 or not np.isin(image, [0, 255]).all():
                    raise ValueError("one exact RAW96 binary render required per pose")
                position = np.asarray(position, dtype=np.float64)
                if (position.shape != (3,) or not np.isfinite(position).all()
                        or not np.allclose(position[:2], point, rtol=0, atol=1e-8)):
                    raise ValueError("rendered environment pose differs from predeclared pose")
                images.append(image.copy())
                actual.append(position.tolist())
            raw = np.stack([images[j] for j in (0, 3, 6, 9)])
            mosaic = np.block([[raw[0, 0], raw[1, 0]], [raw[2, 0], raw[3, 0]]])
            dmd = np.pad(mosaic.repeat(3, 0).repeat(3, 1), 12)
            files = {}
            for suffix, value in (("RAW96", raw), ("DMD", dmd)):
                path = output / (spec["id"] + "_" + suffix + ".npy")
                with path.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                files[suffix] = {"path": str(path.resolve()), "sha256": _sha(path)}
            key = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
            reasons = []
            if np.any(raw.sum((1, 2, 3)) == 0):
                reasons.append("empty_source_frame")
            if any(np.any(part) for part in (raw[..., 0, :], raw[..., -1, :], raw[..., :, 0], raw[..., :, -1])):
                reasons.append("source_ball_touches_frame_edge")
            if key in seen:
                reasons.append("pixel_identical_to_declared_seen_archive")
            rows.append({"id": spec["id"], "input_id": spec["id"], "role": spec["role"],
                "source_kind": INPUT_MODE, "source_generation_protocol": protocol_identity,
                "trajectory_group": plan["design"]["id"] + ":" + spec["id"],
                "seed": plan["design"]["seed"] + index, "mode": INPUT_MODE,
                "query_xy_m": spec["query_xy_m"], "prescribed_velocity_xy_m_s": spec["velocity_xy_m_s"],
                "history_steps": [0, 3, 6, 9], "history_times_s": [0., .06, .12, .18],
                "history_ages_s": [.18, .12, .06, 0.], "rendered_pose_count": 10,
                "requested_positions_m": requested.tolist(), "rendered_positions_m": actual,
                "raw_path": files["RAW96"]["path"], "raw_sha256": files["RAW96"]["sha256"],
                "dmd_path": files["DMD"]["path"], "dmd_sha256": files["DMD"]["sha256"],
                "input_key": key, "eligible": not reasons, "ineligible_reasons": reasons,
                "source_was_newly_rendered": True, "entire_state_distribution_unseen": False,
                "source_already_measured_on_hardware": False, "teacher_labels_generated": False})
        finally:
            renderer.close()
    # Label blind collisions invalidate ALL copies, including same-role copies;
    # never redraw until a favorable/unique image happens to occur.
    for row in rows:
        copies = [r for r in rows if r["input_key"] == row["input_key"]]
        if len(copies) > 1:
            row["ineligible_reasons"].append("pixel_identical_across_new_trajectories")
        row["eligible"] = not row["ineligible_reasons"]
        metadata = output / (row["id"] + "_INPUT.json")
        _write(metadata, row)
        row["metadata_path"], row["metadata_sha256"] = str(metadata.resolve()), _sha(metadata)
    _check(loaded or {})
    _check(plan["source_and_asset_sha256"])
    _write(output / "INPUTS.json", {"schema": INPUT_SCHEMA, "completed": True, "rows": rows,
        "source_generation_protocol": protocol_identity,
        "frozen_source_and_asset_snapshots": plan["frozen_source_and_asset_snapshots"],
        "render_source_freeze": {"path": str(output / "RENDER_SOURCE_FREEZE.json"),
                                 "sha256": _sha(output / "RENDER_SOURCE_FREEZE.json")},
        "requested_trajectories": len(plan["design"]["trajectories"]), "attempted_trajectories": len(rows),
        "rendered_pose_count": 10 * len(rows), "all_invalid_inputs_retained_no_replacement": True,
        "source_and_assets_unchanged": True, "CUDA_initialized": False,
        "new_input_novelty": plan["new_input_novelty"], "unreleased": dict(RELEASE)})


def prepare_inputs(plan_path, output, *, expected_plan_sha256, execute=False, maximum_seconds=120, legacy_root=None):
    """Freeze a prescribed-input plan; only explicit execution renders on CPU.

    Pose/velocity settings are prospective assessment probes, NOT a new free
    MuJoCo control rollout. No teacher, actor, labels, SDK or display is used.
    """
    if type(execute) is not bool or not math.isfinite(maximum_seconds) or not 0 < maximum_seconds <= 600:
        raise ValueError("explicit execution bool and CPU bound in (0,600] required")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("new input output required")
    plan = _input_preflight(plan_path, expected_plan_sha256, legacy_root)
    plan.update(output_directory=str(output.resolve()), maximum_seconds=float(maximum_seconds), execute_requested=execute)
    if not execute:
        return {"schema": INPUT_SCHEMA, "metadata_validated": True, "executed": False,
            "output_created": False, "plan": plan, "unreleased": dict(RELEASE)}
    output.mkdir(exist_ok=False)
    protocol = output / "PROSPECTIVE_PROTOCOL.json"
    try:
        snapshots = {}
        for name, source in plan["snapshot_sources"].items():
            data = Path(source).read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if sha != plan["source_and_asset_sha256"][source]:
                raise ValueError("input source changed before snapshot: " + source)
            target = output / name
            with target.open("xb") as stream:
                stream.write(data)
            snapshots[name] = {"path": str(target.resolve()), "sha256": sha,
                               "original_source_path": source, "original_source_sha256": sha}
            plan["source_and_asset_sha256"][str(target.resolve())] = sha
        plan["frozen_source_and_asset_snapshots"] = snapshots
        _write(protocol, plan)
        protocol_sha = _sha(protocol)
        _run_worker(protocol, float(maximum_seconds), input_generation=True)
        _check(plan["source_and_asset_sha256"])
        if _sha(protocol) != protocol_sha:
            raise ValueError("input generation protocol changed")
        result = _json(output / "INPUTS.json")
        expected_ids = [r["id"] for r in plan["design"]["trajectories"]]
        if (result["schema"] != INPUT_SCHEMA or result["completed"] is not True
                or result["CUDA_initialized"] is not False or result["unreleased"] != RELEASE
                or [r["id"] for r in result["rows"]] != expected_ids
                or result["attempted_trajectories"] != len(expected_ids)
                or result["rendered_pose_count"] != 10 * len(expected_ids)
                or result["source_and_assets_unchanged"] is not True):
            raise ValueError("complete predeclared CPU input bank required")
        expected = {"PROSPECTIVE_PROTOCOL.json", "INPUT_WORKER.log", "RENDER_SOURCE_FREEZE.json", "INPUTS.json", *snapshots}
        for row in result["rows"]:
            spec = plan["design"]["trajectories"][expected_ids.index(row["id"])]
            if (row["input_id"] != row["id"] or row["role"] != spec["role"]
                    or row["query_xy_m"] != spec["query_xy_m"]
                    or row["prescribed_velocity_xy_m_s"] != spec["velocity_xy_m_s"]):
                raise ValueError("generated trajectory identity differs from frozen plan")
            metadata = output / (row["id"] + "_INPUT.json")
            if (row["metadata_path"] != str(metadata.resolve()) or _sha(metadata) != row["metadata_sha256"]
                    or _canonical(_json(metadata)) != _canonical({k: v for k, v in row.items()
                        if k not in ("metadata_path", "metadata_sha256")})):
                raise ValueError("generated trajectory metadata binding differs")
            expected.add(metadata.name)
            for key in ("raw", "dmd"):
                path = output / (row["id"] + ("_RAW96.npy" if key == "raw" else "_DMD.npy"))
                if row[key + "_path"] != str(path.resolve()) or _sha(path) != row[key + "_sha256"]:
                    raise ValueError("generated input asset changed")
                expected.add(path.name)
        if {p.name for p in output.iterdir()} != expected or any(p.is_symlink() or not p.is_file() for p in output.iterdir()):
            raise ValueError("exact new input output inventory required")
        files = {p.name: _sha(p) for p in output.iterdir()}
        _write(output / "MANIFEST.json", {"schema": INPUT_SCHEMA, "completed": True,
            "file_sha256": files, "manifest_excludes_itself": True, "unreleased": dict(RELEASE)})
        return result
    except Exception as exc:
        _write(output / "FAILED.json", {"schema": INPUT_SCHEMA, "completed": False,
            "error": type(exc).__name__ + ": " + str(exc), "partial_files_preserved": True,
            "unreleased": dict(RELEASE)})
        raise
