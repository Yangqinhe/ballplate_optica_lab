"""Explicit local input observability experiment; offline except legacy transport.

Nine prescribed histories, two encodings and three immutable raw masks. This
module prepares evidence and plans, never connects a camera or creates windows.
The independent existing-owner transport is the only projection entry point.
"""
from pathlib import Path
import json

import numpy as np

from .storage import PROJECT, read_json, sha256, utc_now, write_json_new
from .study import _input, encode_history
from .probes import _npy_bytes, _png_bytes

INPUT_IDS = ("REF", "X_PLUS", "X_MINUS", "Y_PLUS", "Y_MINUS",
             "VX_PLUS", "VX_MINUS", "VY_PLUS", "VY_MINUS")
SCHEMA = "optical-lab-observability-bank-1"
PREDICTION_CHECK_SCOPE = "numeric_arrays_full_initial_preload_final__metadata_each_control_check"


def _prediction_control_bindings(root, digest, ledger):
    """Large offline derivatives are not command assets; audit fully at edges.

    Runtime source/actual DMD/SLM/RAW assets remain checked every control check.
    All prediction files are still rehashed by _archive before/after capture.
    """
    names = {"RESULT.json", "PROSPECTIVE_PROTOCOL.json", "identification_source.py",
             "M0_REFERENCE_PRE_PREDICTION_FREEZE.json", "M1_RADIAL_PRE_PREDICTION_FREEZE.json",
             "SENSITIVITY_M0_PRE_PREDICTION_FREEZE.json"}
    if not names.issubset(ledger):
        raise ValueError("complete prospective model/source freeze metadata required")
    return {str(root / "MANIFEST.json"): digest, **{str(root / n): ledger[n] for n in names}}


def _verify_predictions(root, ledger, bank, bank_sha256):
    """Independently read all nominal outputs; never trust a max/range flag."""
    from .simulate import FIXED, REFERENCE, WAVEFIT, REGIONS
    result = read_json(root / "RESULT.json")
    protocol = read_json(root / "PROSPECTIVE_PROTOCOL.json")
    legacy = PROJECT.parent / "ballplate_fourframe96_deploy"
    models = {"M0_REFERENCE": dict(path=str(legacy / REFERENCE), sha256=FIXED[REFERENCE]),
              "M1_RADIAL": dict(path=str(legacy / WAVEFIT), sha256=FIXED[WAVEFIT])}
    if (result.get("complete") is not True or result.get("hardware_accessed") is not False
            or result.get("bank_sha256") != bank_sha256 or result.get("nominal_prediction_count") != 108
            or result.get("sensitivity_command_count") != 27 or result.get("no_parameters_fitted") is not True
            or result.get("all_nominal_means_below_capture_limit240") is not True
            or protocol.get("models") != models or protocol.get("bank_sha256") != bank_sha256
            or protocol.get("readout_sha256") != FIXED[REGIONS]):
        raise ValueError("complete unchanged M0/M1/BROAD8 numerical pre-predictions required")
    regions_path = legacy / REGIONS
    if sha256(regions_path) != FIXED[REGIONS]:
        raise ValueError("native BROAD8 identity changed")
    regions = np.load(regions_path, allow_pickle=False).astype(float)
    expected = {(m, i, k) for m in models for i in bank["inputs"] for k in bank["masks"]}
    actual = [(r["model"], r["input_id"], r["mask_id"]) for r in result["rows"]]
    if len(actual) != 108 or set(actual) != expected:
        raise ValueError("all108 unique model/input/mask predictions required")
    for row in result["rows"]:
        arrays = {}
        if set(row["files"]) != {"ccd", "native8"}:
            raise ValueError("both nominal native outputs required")
        for name, record in row["files"].items():
            p = root / record["path"]
            if p.parent != root or ledger.get(p.name) != record["sha256"]:
                raise ValueError("prediction row/file ledger association differs")
            arrays[name] = np.load(p, allow_pickle=False)
        a, f = arrays["ccd"], arrays["native8"]
        if (a.dtype != np.float32 or a.shape != (554, 552) or not np.isfinite(a).all()
                or a.min() < 0 or a.max() >= 240 or f.shape != (8,) or not np.isfinite(f).all()
                or float(a.max()) != row["mean_CCD_max_ADU"]
                or row["predicted_max_below_capture_limit240"] is not True
                or not np.allclose(np.einsum("khw,hw->k", regions, a.astype(float)), f, rtol=0, atol=1e-6)):
            raise ValueError("nominal prediction shape/ADU/readout integrity failed")
    return result


def _same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def _archive(root, digest):
    root = Path(root).resolve(strict=True)
    if sha256(root / "MANIFEST.json") != digest or (root / "FAILED.json").exists():
        raise ValueError("explicit complete archive MANIFEST identity required")
    ledger = read_json(root / "MANIFEST.json")["file_sha256"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "MANIFEST.json"}
    if actual != set(ledger):
        raise ValueError("archive inventory changed")
    for name, expected in ledger.items():
        p = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or p.is_symlink() or sha256(p) != expected:
            raise ValueError("archive asset changed: " + name)
    return root, ledger


def prepare_bank(design_path, cpu_bank, output, *, design_sha256, cpu_manifest_sha256):
    """CPU-only deterministic assets, never choose a successful subset."""
    design_path = Path(design_path).resolve(strict=True)
    if sha256(design_path) != design_sha256:
        raise ValueError("external design SHA required")
    design = read_json(design_path)
    if (design.get("schema") != "optical-lab-observability-design-1"
            or design.get("hardware_actions_enabled") is not False or design.get("radii") != [0, 2]
            or design.get("frames_per_state") != 8 or design.get("states_per_batch") != 11
            or set(design["masks"]) != {"learned", "lens", "flat"}):
        raise ValueError("exact explicit E005 design required")
    source, ledger = _archive(cpu_bank, cpu_manifest_sha256)
    data = read_json(source / "INPUTS.json")
    if [r["id"] for r in data["rows"]] != list(INPUT_IDS) or not all(r["eligible"] for r in data["rows"]):
        raise ValueError("all nine predeclared eligible inputs required; retain failures, no resampling")
    root = Path(output).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir()
    files = {"DESIGN.json": design_path.read_bytes(), "SOURCE_INPUTS.json": (source / "INPUTS.json").read_bytes(),
             "SOURCE_MANIFEST.json": (source / "MANIFEST.json").read_bytes(),
             "observability_source.py": Path(__file__).read_bytes(),
             "encoding_source.py": Path(__file__).with_name("study.py").read_bytes(),
             "png_source.py": Path(__file__).with_name("probes.py").read_bytes()}
    inputs, masks, geometry = {}, {}, {}
    for row in data["rows"]:
        raw = np.load(row["raw_path"], allow_pickle=False)
        dmd = np.load(row["dmd_path"], allow_pickle=False)
        for key in ("raw", "dmd"):
            if sha256(row[key + "_path"]) != row[key + "_sha256"]:
                raise ValueError("source input changed")
        _input(raw, dmd)
        if row["role"] != "calibration" or row["history_ages_s"] != [.18, .12, .06, 0.]:
            raise ValueError("local calibration-only four-image histories required")
        for radius in (0, 2):
            encoded, info = encode_history(raw, radius)
            panel = np.pad(np.block([[encoded[0, 0], encoded[1, 0]], [encoded[2, 0], encoded[3, 0]]]).repeat(3, 0).repeat(3, 1), 12)
            _input(encoded, panel)
            key = f"R{radius}_{row['id']}"
            files[key + "_RAW96.npy"] = _npy_bytes(encoded)
            files[key + "_DMD.npy"] = _npy_bytes(panel)
            files[key + "_DMD.png"] = _png_bytes(panel)
            geometry[key] = info
            inputs[key] = dict(path=str(root / (key + "_DMD.npy")), sha256=None,
                raw96=dict(path=str(root / (key + "_RAW96.npy")), sha256=None), id=key,
                selection_rule="predeclared_local_calibration_not_outcome_selected",
                history_ages_s=[.18, .12, .06, 0.])
    for key, item in design["masks"].items():
        p = Path(item["path"])
        if not p.is_absolute() or p.is_symlink() or sha256(p) != item["sha256"]:
            raise ValueError("frozen original raw mask identity changed")
        a = np.load(p, allow_pickle=False)
        if a.dtype != np.uint8 or a.shape != (600, 600) or not np.array_equal(a, a[::4, ::4].repeat(4, 0).repeat(4, 1)):
            raise ValueError("original canonical600 macro4-compatible mask required")
        files["SLM_" + key + ".npy"] = p.read_bytes()
        files["SLM_" + key + ".png"] = _png_bytes(a)
        masks[key] = dict(path=str(root / ("SLM_" + key + ".npy")), sha256=item["sha256"])
    for name, blob in files.items():
        with (root / name).open("xb") as stream:
            stream.write(blob)
    for value in inputs.values():
        value["sha256"] = sha256(value["path"])
        value["raw96"]["sha256"] = sha256(value["raw96"]["path"])
    bank = dict(schema=SCHEMA, created_utc=utc_now(), design=design, inputs=inputs, masks=masks,
        encoding_geometry=geometry, source_cpu_bank=dict(path=str(source), sha256=cpu_manifest_sha256),
        source_design=dict(path=str(design_path), sha256=design_sha256),
        role="development_calibration_NOT_blind_validation", hardware_accessed=False,
        source_bindings={str(source / "MANIFEST.json"): cpu_manifest_sha256,
            **{str(source / name): digest for name, digest in ledger.items()},
            str(design_path): design_sha256},
        production_qualified=False, control_release_approved=False)
    write_json_new(root / "BANK.json", bank)
    write_json_new(root / "MANIFEST.json", dict(schema=SCHEMA,
        file_sha256={p.name: sha256(p) for p in root.iterdir() if p.is_file()}, manifest_excludes_itself=True))
    verify_bank(root, sha256(root / "MANIFEST.json"))
    return dict(path=str(root), sha256=sha256(root / "MANIFEST.json"), input_count=len(inputs), mask_count=len(masks))


def verify_bank(root, expected_sha256):
    root, ledger = _archive(root, expected_sha256)
    bank = read_json(root / "BANK.json")
    if bank.get("schema") != SCHEMA or set(bank["inputs"]) != {f"R{r}_{s}" for r in (0, 2) for s in INPUT_IDS}:
        raise ValueError("complete eighteen-input bank required")
    for key, item in bank["inputs"].items():
        for record in (item, item["raw96"]):
            p = Path(record["path"])
            if p.parent != root or ledger.get(p.name) != record["sha256"]:
                raise ValueError("input identity outside bank")
        raw, dmd = [np.load(p, allow_pickle=False) for p in (item["raw96"]["path"], item["path"])]
        _input(raw, dmd)
        if item["id"] != key or item["history_ages_s"] != [.18, .12, .06, 0.]:
            raise ValueError("input metadata mismatch")
        if key.startswith("R2_"):
            source = bank["inputs"]["R0_" + key[3:]]
            expected, geometry = encode_history(np.load(source["raw96"]["path"], allow_pickle=False), 2)
            frozen = bank["encoding_geometry"][key]
            # NumPy1.21 (public UI client) and NumPy2.2 (CPU renderer) can
            # differ by one final norm ULP. Pixels/all primitive geometry stay
            # exact; the derived L2 scalar alone has 1e-14 absolute tolerance.
            scalar = "centroid_delta_L2_raw_px"
            geometry_same = len(geometry) == len(frozen) and all(
                _same({k: v for k, v in a.items() if k != scalar}, {k: v for k, v in b.items() if k != scalar})
                and abs(a[scalar] - b[scalar]) <= 1e-14 for a, b in zip(geometry, frozen))
            if not np.array_equal(raw, expected) or not geometry_same:
                raise ValueError("encoding differs from frozen disk operation")
    if set(bank["masks"]) != {"learned", "lens", "flat"}:
        raise ValueError("three frozen mask identities required")
    for key, record in bank["masks"].items():
        p = Path(record["path"])
        if p.parent != root or ledger.get(p.name) != record["sha256"] or record["sha256"] != bank["design"]["masks"][key]["sha256"]:
            raise ValueError("mask identity mismatch")
        a = np.load(p, allow_pickle=False)
        if a.dtype != np.uint8 or a.shape != (600, 600) or not np.array_equal(a, a[::4, ::4].repeat(4, 0).repeat(4, 1)):
            raise ValueError("mask geometry changed")
    return bank, ledger


def verify_bank_and_plan(plan):
    """Called lazily by original-owner transport; no legacy or device imports."""
    bank, ledger = verify_bank(plan["observability_bank"]["path"], plan["observability_bank"]["sha256"])
    batch = [r for r in bank["design"]["batch_order"] if r["id"] == plan.get("batch_id")]
    if len(batch) != 1 or not _same(batch[0], plan.get("batch_spec")):
        raise ValueError("predeclared batch identity required")
    spec = batch[0]
    family = ("REF", "X_PLUS", "X_MINUS", "Y_PLUS", "Y_MINUS") if spec["axis_family"] == "POS" else ("REF", "VX_PLUS", "VX_MINUS", "VY_PLUS", "VY_MINUS")
    expected = {k: bank["inputs"][f"R{spec['radius']}_{k}"] for k in family}
    if (plan.get("profile") != "observability_v1" or plan.get("axis_family") != spec["axis_family"]
            or plan.get("mask_key") != spec["mask_key"] or not _same(plan["inputs"], expected)
            or not _same(plan["input"], expected["REF"])
            or not _same(plan["masks"], {k: bank["masks"][spec["mask_key"]] for k in ("Z", "A", "B")})):
        raise ValueError("batch input/mask/profile binding mismatch")
    from .protocol import observability_states
    if plan["states"] != observability_states(spec["axis_family"]):
        raise ValueError("exact alternating reference input sequence required")
    root = Path(plan["observability_bank"]["path"])
    required = {str(root / "MANIFEST.json"): plan["observability_bank"]["sha256"],
        **{str(root / name): digest for name, digest in ledger.items()},
        **bank["source_bindings"], str(Path(__file__).resolve()): sha256(__file__)}
    if any(plan["source_sha256"].get(p) != h for p, h in required.items()):
        raise ValueError("complete bank/source ledger must be frozen in plan")
    pred = plan["prospective_predictions"]
    pred_root, pred_ledger = _archive(pred["path"], pred["manifest_sha256"])
    result = _verify_predictions(pred_root, pred_ledger, bank, plan["observability_bank"]["sha256"])
    if (result.get("complete") is not True or result.get("hardware_accessed") is not False
            or result.get("bank_sha256") != plan["observability_bank"]["sha256"]
            or result.get("nominal_prediction_count") != 108
            or result.get("all_nominal_means_below_capture_limit240") is not True
            or pred.get("verification_scope") != PREDICTION_CHECK_SCOPE):
        raise ValueError("complete prospective predictions bound to exact input bank required")
    for path, digest in _prediction_control_bindings(pred_root, pred["manifest_sha256"], pred_ledger).items():
        if plan["source_sha256"].get(path) != digest:
            raise ValueError("complete prediction archive not frozen in plan")
    return dict(verified=True, input_count=len(expected), fixed_mask=spec["mask_key"])


def prepare_plan(bank_root, output, *, bank_sha256, batch_id, authority_statement,
                 predictions_root, predictions_sha256):
    """Read-only owner status + identity freeze; never public UI operations."""
    from .capture import SESSION, SCHEMA as PLAN_SCHEMA
    from .legacy_gui import dependency_sources, _heartbeat, Clock
    from .references import verify
    from .protocol import observability_states
    bank, ledger = verify_bank(bank_root, bank_sha256)
    root = Path(bank_root).resolve(strict=True)
    pred, pred_ledger = _archive(predictions_root, predictions_sha256)
    predicted = _verify_predictions(pred, pred_ledger, bank, bank_sha256)
    if (predicted.get("complete") is not True or predicted.get("bank_sha256") != bank_sha256
            or predicted.get("hardware_accessed") is not False
            or predicted.get("nominal_prediction_count") != 108
            or predicted.get("all_nominal_means_below_capture_limit240") is not True):
        raise ValueError("complete prospective CPU predictions bound to bank required")
    specs = [b for b in bank["design"]["batch_order"] if b["id"] == batch_id]
    if len(specs) != 1 or len(authority_statement.strip()) < 10:
        raise ValueError("explicit predeclared batch and current authorization required")
    spec = specs[0]
    legacy = PROJECT.parent / "ballplate_fourframe96_deploy"
    if not verify("hardware", legacy_root=legacy)["verified"]:
        raise ValueError("original hardware source identities changed")
    session = legacy / SESSION
    runtime = read_json(session / "runtime_sources.json")
    contract_path = legacy / "outputs/learned_real_control_20261022_v31/BUNDLE/CONTRACT.json"
    if sha256(contract_path) != "59c833b91ac4b1c3395905e12a89c918fa994294fc3ad89880eeed886589b992":
        raise ValueError("camera and geometry contract changed")
    contract = read_json(contract_path)
    keys = ("x", "y", "full_panel", "square", "focal_mm", "period")
    h = _heartbeat(session, dict(expected_pid=runtime["pid"], camera_expected=dict(active=contract["expected_active"], roi=contract["roi"]),
        manual_geometry_expected={k: contract["settings"][k] for k in keys}), Clock())
    sources = {**dependency_sources(legacy), **runtime["sources"], **runtime["source_configs"],
        **bank["source_bindings"], str(root / "MANIFEST.json"): bank_sha256,
        **{str(root / n): d for n, d in ledger.items()},
        **_prediction_control_bindings(pred, predictions_sha256, pred_ledger)}
    for p in (session / "runtime_sources.json", legacy / "workbench/current.json", contract_path,
              PROJECT / "references/hardware/manifest.json", Path(__file__),
              Path(__file__).with_name("legacy_gui.py"), Path(__file__).with_name("protocol.py"),
              Path(__file__).with_name("references.py"), Path(__file__).with_name("storage.py"),
              Path(__file__).with_name("simulate.py"), Path(__file__).with_name("study.py"),
              Path(__file__).with_name("probes.py"), Path(__file__).with_name("capture.py")):
        sources[str(p.resolve())] = sha256(p)
    for p, digest in sources.items():
        if sha256(p) != digest:
            raise ValueError("source changed before freeze: " + p)
    family = ("REF", "X_PLUS", "X_MINUS", "Y_PLUS", "Y_MINUS") if spec["axis_family"] == "POS" else ("REF", "VX_PLUS", "VX_MINUS", "VY_PLUS", "VY_MINUS")
    inputs = {k: bank["inputs"][f"R{spec['radius']}_{k}"] for k in family}
    plan = dict(schema=PLAN_SCHEMA, profile="observability_v1", created_utc=utc_now(),
        expected_pid=runtime["pid"], session=str(session), legacy_root=str(legacy),
        observability_bank=dict(path=str(root), sha256=bank_sha256), batch_id=batch_id, batch_spec=spec,
        axis_family=spec["axis_family"], mask_key=spec["mask_key"], inputs=inputs, input=inputs["REF"],
        masks={k: bank["masks"][spec["mask_key"]] for k in ("Z", "A", "B")},
        states=observability_states(spec["axis_family"]), frames_per_state=8, total_planned_frames=88,
        source_sha256=sources, hardware_contract=dict(path=str(contract_path), sha256=sha256(contract_path)),
        camera_expected=dict(active=h["camera_active_readback"], roi=h["actual_roi"]),
        manual_geometry_expected={k: h["manual_settings"][k] for k in keys},
        prospective_predictions=dict(path=str(pred), manifest_sha256=predictions_sha256,
            verification_scope=PREDICTION_CHECK_SCOPE),
        regions=dict(path=str(legacy / "outputs/readout_coverage_20261022_v34/BROAD_NATIVE8.npy"),
                     sha256="0a76e770d8d9f2761a9edce492b4a846eab98ebf0e6f719271711202954601c5"),
        authorization=dict(confirmed_utc=utc_now(), statement=authority_statement,
            scope="bounded static original window only; no rebuild/settings/closed-loop"),
        maximum_seconds=300, cleanup_reserve_seconds=12, minimum_state_start_seconds=20,
        inter_block_wait_seconds=0, software_settle_wait_seconds=1.8, minimum_postcommit_frame_start_seconds=1.5,
        frame_pattern_association_verified=False, optical_settling_verified=False,
        sensor_physical_dark_measurement_verified=False, new_LUT_applied=False,
        mapping="canonical600 existing owner embeds and flips SLM once; no new LUT or preflip",
        background_kind="DMD_SLM_zero_optical_background_NOT_physical_sensor_dark",
        no_training=True, no_closed_loop=True, real_actuators=False,
        production_qualified=False, control_release_approved=False, gate_states={k: "UNKNOWN" for k in ("G0", "G1", "G2")})
    verify_bank_and_plan(plan)
    output = Path(output)
    output.mkdir(exist_ok=False)
    write_json_new(output / "HEARTBEAT_BEFORE_PLAN.json", h)
    write_json_new(output / "PLAN.json", plan)
    return dict(plan=str((output / "PLAN.json").resolve()), plan_sha256=sha256(output / "PLAN.json"),
                source_count=len(sources), planned_frames=88, hardware_accessed=False)
