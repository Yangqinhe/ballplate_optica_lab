"""Prepare one prospective legacy pilot or explicit static case, never a device.

The separately invoked legacy_gui transport controls only the already-running
original window. This file selects evidence, not hardware settings or weights.
"""
from pathlib import Path
import time

from .probes import verify_pilot_assets
from .protocol import bracketed_states
from .references import verify
from .storage import PROJECT, read_json, sha256, utc_now, write_json_new


PILOT_SHA = "0e829e0dce87e4e8010b75c7baca0138886d8d228767a1cad69e9a8a81c87f3c"
SCHEMA = "optical-lab-existing-owner-pilot-plan-1"
SESSION = "outputs/live_calibration/INTERACTIVE_LOOP_V23_20261021_1930/GUI"


def pilot_states():
    return bracketed_states(16)


def _predictions(folder, expected_manifest_sha256, *, case_manifest_sha256=None, case_directory=None):
    root = Path(folder).resolve(strict=True)
    if (root / "FAILED.json").exists():
        raise ValueError("failed prediction archive is not a prospective reference")
    if sha256(root / "MANIFEST.json") != expected_manifest_sha256:
        raise ValueError("external prospective prediction manifest identity changed")
    result, manifest = read_json(root / "RESULT.json"), read_json(root / "MANIFEST.json")
    case = None
    if case_manifest_sha256 is not None:
        from .study import verify_case_assets
        if case_directory is None:
            raise ValueError("explicit case directory required for new-case predictions")
        case_root = Path(case_directory).resolve(strict=True)
        case = verify_case_assets(case_root, expected_manifest_sha256=case_manifest_sha256)["manifest"]
    elif case_directory is not None:
        raise ValueError("case directory cannot substitute for external case manifest identity")
    expected_schema = ("optical-lab-static-case-cpu-predictions-1" if case is not None
                       else "optical-lab-fixed-pilot-cpu-predictions-1")
    if (result.get("schema") != expected_schema
            or result.get("completed") is not True or result.get("propagation_executed") is not True
            or result.get("backend") != "CPU_threads2_no_CUDA"
            or result.get("source_and_assets_unchanged") is not True
            or result.get("G2_model_qualified") is not False):
        raise ValueError("complete explicitly unqualified fixed CPU predictions required")
    ledger = manifest["file_sha256"]
    required = {"RESULT.json", "PROSPECTIVE_PROTOCOL.json", "WORKER_RESULT.json", "WORKER.log"}
    for model in ("M0_REFERENCE", "M1_RADIAL"):
        required.add(f"{model}_PRE_PREDICTION_FREEZE.json")
        required.update(f"{model}_{mask}_{suffix}.npy" for mask in ("Z", "A", "B")
                        for suffix in ("NATIVE_CCD_MEAN_ADU", "NATIVE8_FEATURES"))
    if set(ledger) != required:
        raise ValueError("complete fixed prediction artifact set required")
    actual = {p.name for p in root.iterdir() if p.is_file() and p.name != "MANIFEST.json"}
    if actual != set(ledger):
        raise ValueError("prediction archive file set changed")
    bound = {str(root / "MANIFEST.json"): sha256(root / "MANIFEST.json")}
    for name, digest in ledger.items():
        if Path(name).name != name or (root / name).is_symlink() or sha256(root / name) != digest:
            raise ValueError("prediction archive identity changed: " + name)
        bound[str(root / name)] = digest
    rows = [(r["model"], r["mask"]) for r in result["rows"]]
    if rows != [(m, s) for m in ("M0_REFERENCE", "M1_RADIAL") for s in ("Z", "A", "B")]:
        raise ValueError("exact fixed six prospective predictions required")
    protocol = read_json(root / "PROSPECTIVE_PROTOCOL.json")
    if (result["protocol_sha256"] != ledger["PROSPECTIVE_PROTOCOL.json"]
            or result["worker_result_sha256"] != ledger["WORKER_RESULT.json"]
            or protocol["samples"] != [2, 3] or protocol["sensor_mode"] != "mean"
            or protocol["native_shape"] != [554, 552]):
        raise ValueError("frozen protocol/model output association failed")
    from .simulate import FIXED, RAW, DMD, REFERENCE, WAVEFIT, REGIONS, HEAD, CONTRACT
    legacy = Path(protocol["legacy_root"])
    fixed = protocol["fixed_input"]
    if case is None:
        if (protocol["pilot_manifest_sha256"] != PILOT_SHA
                or fixed["raw_path"] != str(legacy / RAW) or fixed["raw_sha256"] != FIXED[RAW]
                or fixed["dmd_path"] != str(legacy / DMD) or fixed["dmd_sha256"] != FIXED[DMD]
                or any(protocol["source_and_asset_sha256"].get(str(legacy / name)) != FIXED[name] for name in (RAW,DMD))):
            raise ValueError("fixed original-pilot T01 input identity required")
    else:
        expected_case = {"case_asset_manifest": {"path":str(case_root),"sha256":case_manifest_sha256},
            "case_id":case["design"]["case_id"],"role":case["design"]["role"],
            "design":case["design"],"fixed_input":case["fixed_input"],"probe_spec":case["probe_spec"]}
        import json
        same=lambda a,b:json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
        if "pilot_manifest_sha256" in protocol or "pilot_manifest_sha256" in result or protocol.get("schema") != expected_schema or any(
                not same(protocol.get(key),value) or not same(result.get(key),value) for key,value in expected_case.items()):
            raise ValueError("case prediction identity/input/probe/design/role mismatch")
        required_case = {str(case_root/"MANIFEST.json"):case_manifest_sha256,
            **{str(case_root/name):digest for name,digest in case["file_sha256"].items()},
            case["design"]["path"]:case["design"]["sha256"]}
        if any(protocol["source_and_asset_sha256"].get(path)!=digest for path,digest in required_case.items()):
            raise ValueError("prediction protocol does not bind complete case assets/design")
    if (protocol["models"] != {"M0_REFERENCE": {"path": str(legacy / REFERENCE), "sha256": FIXED[REFERENCE]},
                                     "M1_RADIAL": {"path": str(legacy / WAVEFIT), "sha256": FIXED[WAVEFIT]}}
            or any(protocol["source_and_asset_sha256"].get(str(legacy / name)) != FIXED[name]
                   for name in (REFERENCE, WAVEFIT, REGIONS, HEAD, CONTRACT))):
        raise ValueError("fixed M0/M1/BROAD8/head identity required")
    for row in result["rows"]:
        if set(row["files"]) != {"NATIVE_CCD_MEAN_ADU", "NATIVE8_FEATURES"}:
            raise ValueError("both prediction outputs required")
        for suffix, item in row["files"].items():
            name = f"{row['model']}_{row['mask']}_{suffix}.npy"
            if item != {"path": name, "sha256": ledger[name]}:
                raise ValueError("prediction row is not associated with its frozen file")
    for model, item in result["model_freezes"].items():
        name = f"{model}_PRE_PREDICTION_FREEZE.json"
        if model not in ("M0_REFERENCE", "M1_RADIAL") or item != {"path": name, "sha256": ledger[name]}:
            raise ValueError("model freeze output association failed")
    if set(result["model_freezes"]) != {"M0_REFERENCE", "M1_RADIAL"}:
        raise ValueError("both prospective model freezes required")
    return root, result, bound


def prepare_capture(pilot_dir, predictions_dir, output, *, authority_statement, prediction_manifest_sha256,
                    case_manifest_sha256=None, frames_per_state=16):
    """Freeze a new plan using read-only files/heartbeat, never public UI calls.

    Requires a current user confirmation, but does NOT turn that string into a
    hardware/control release. Execution must separately request the bounded
    original-window workflow and check this exact plan SHA.
    """
    states=bracketed_states(frames_per_state)
    if case_manifest_sha256 is None and frames_per_state != 16:
        raise ValueError("the original pilot remains fixed16; eight repeats require an explicit new case")
    if not isinstance(authority_statement, str) or len(authority_statement.strip()) < 10:
        raise ValueError("current specific unchanged-condition confirmation required")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("new plan directory required")
    pilot = Path(pilot_dir).resolve(strict=True)
    case = None
    if case_manifest_sha256 is None:
        verify_pilot_assets(pilot, expected_manifest_sha256=PILOT_SHA)
        pred_root, prediction, bindings = _predictions(predictions_dir, prediction_manifest_sha256)
    else:
        from .study import verify_case_assets
        case = verify_case_assets(pilot, expected_manifest_sha256=case_manifest_sha256)["manifest"]
        pred_root, prediction, bindings = _predictions(predictions_dir, prediction_manifest_sha256,
            case_manifest_sha256=case_manifest_sha256,case_directory=pilot)
    protocol = read_json(pred_root / "PROSPECTIVE_PROTOCOL.json")
    if case is None and protocol["pilot_manifest_sha256"] != PILOT_SHA:
        raise ValueError("predictions did not bind the frozen E002 assets")
    legacy = Path(protocol["legacy_root"]).resolve(strict=True)
    reference = verify("hardware", legacy_root=legacy)
    if reference["verified"] is not True:
        raise ValueError("preserved hardware reference identity failed")
    session = legacy / SESSION
    runtime = read_json(session / "runtime_sources.json")
    contract_path = legacy / "outputs/learned_real_control_20261022_v31/BUNDLE/CONTRACT.json"
    contract = read_json(contract_path)
    geometry_keys = ("x", "y", "full_panel", "square", "focal_mm", "period")
    # This dependency discovery parses source files only; it must not import
    # the original GUI, SDK, X11 or Torch during plan preparation.
    from .legacy_gui import dependency_sources, _heartbeat, Clock
    try:
        heartbeat = _heartbeat(session,dict(expected_pid=runtime["pid"],
            camera_expected=dict(active=contract["expected_active"],roi=contract["roi"]),
            manual_geometry_expected={k:contract["settings"][k] for k in geometry_keys}),Clock())
    except RuntimeError as exc:
        raise ValueError("fresh original idle camera/photometry/trigger/ROI/manual geometry contract required: "+str(exc)) from exc
    active, roi = heartbeat["camera_active_readback"], heartbeat["actual_roi"]
    bindings.update(dependency_sources(legacy))
    for group in ("sources", "source_configs"):
        for name, digest in runtime[group].items():
            if name in bindings and bindings[name] != digest:
                raise ValueError("conflicting frozen original source identity")
            bindings[name] = digest
    for name in (session / "runtime_sources.json", legacy / "workbench/current.json",
                 PROJECT / "references/hardware/manifest.json", Path(__file__).resolve(),
                 Path(__file__).with_name("legacy_gui.py"), Path(__file__).with_name("probes.py"),
                 Path(__file__).with_name("protocol.py"),
                 Path(__file__).with_name("references.py"), Path(__file__).with_name("storage.py")):
        bindings[str(name)] = sha256(name)
    for name, digest in protocol["source_and_asset_sha256"].items():
        if name in bindings and bindings[name] != digest:
            raise ValueError("prediction/source identity conflicts with original owner")
        bindings[name] = digest
    if case is not None:
        for path,digest in {str(pilot/"MANIFEST.json"):case_manifest_sha256,
                **{str(pilot/name):digest for name,digest in case["file_sha256"].items()},
                case["design"]["path"]:case["design"]["sha256"],
                str(Path(__file__).with_name("study.py")):sha256(Path(__file__).with_name("study.py"))}.items():
            if path in bindings and bindings[path]!=digest:
                raise ValueError("conflicting explicit case source/asset identity")
            bindings[path]=digest
    for name, digest in bindings.items():
        if sha256(name) != digest:
            raise ValueError("predeclared source/asset changed: " + name)
    fixed = protocol["fixed_input"]
    masks = {k: {"path": str(pilot / f"SLM_{k}_CANONICAL600_RAW_GRAY_NO_LUT.npy"),
                 "sha256": sha256(pilot / f"SLM_{k}_CANONICAL600_RAW_GRAY_NO_LUT.npy")}
             for k in ("Z", "A", "B")}
    plan = {
        "schema": SCHEMA, "created_utc": utc_now(), "session": str(session), "legacy_root": str(legacy),
        "display": read_json(pilot / (case["design"]["snapshot"] if case is not None else "protocol_snapshot.json"))["display"],
        "expected_pid": heartbeat["pid"],
        "input": {"path": fixed["dmd_path"], "sha256": fixed["dmd_sha256"],
                  "raw96": {"path": fixed["raw_path"], "sha256": fixed["raw_sha256"]},
                  "id": fixed["id"], "selection_rule": fixed["selection_rule"]},
        "masks": masks, "states": states, "total_planned_frames": 17*frames_per_state,
        "source_sha256": bindings,
        "hardware_contract": {"path": str(contract_path),
                              "sha256": "59c833b91ac4b1c3395905e12a89c918fa994294fc3ad89880eeed886589b992"},
        "camera_expected": {"active": active, "roi": roi},
        "manual_geometry_expected": {k: heartbeat["manual_settings"][k]
                                     for k in geometry_keys},
        "frozen_predictions": {"path": str(pred_root), "manifest_sha256": sha256(pred_root / "MANIFEST.json"),
                               "result_sha256": sha256(pred_root / "RESULT.json")},
        "regions": {"path": str(legacy / "outputs/readout_coverage_20261022_v34/BROAD_NATIVE8.npy"),
                    "sha256": "0a76e770d8d9f2761a9edce492b4a846eab98ebf0e6f719271711202954601c5"},
        "authorization": {"confirmed_utc": utc_now(), "statement": authority_statement,
                          "scope": "bounded static original window only; no rebuild/settings/closed-loop"},
        "maximum_seconds": 300, "cleanup_reserve_seconds": 12, "minimum_state_start_seconds": 20,
        "inter_block_wait_seconds": 0,
        "block_independence_claimed": False, "optical_settling_verified": False,
        "software_settle_wait_seconds": 1.8, "minimum_postcommit_frame_start_seconds": 1.5,
        "settling_rule": "1.8s conservative software wait per state; not measured liquid-crystal settling",
        "background_kind": "DMD_SLM_zero_optical_background_NOT_physical_sensor_dark",
        "sensor_physical_dark_measurement_verified": False,
        "frame_pattern_association_verified": False, "camera_clock_relationship_verified": False,
        "mapping": "existing owner canonical600 embed then exactly one owner fliplr; no new LUT or preflip",
        "model_geometry_qualified": False, "new_LUT_applied": False,
        "no_training": True, "no_closed_loop": True, "real_actuators": False,
        "gate_states": {"G0": "UNKNOWN", "G1": "UNKNOWN", "G2": "UNKNOWN"},
        "production_qualified": False, "control_release_approved": False,
    }
    if case is not None:
        plan.update(profile="static_case_v1",case_manifest_sha256=case_manifest_sha256,
            frames_per_state=frames_per_state,
            case_asset_manifest=dict(path=str(pilot),sha256=case_manifest_sha256),
            case_id=case["design"]["case_id"],role=case["design"]["role"],design=case["design"],
            case_spec=case["probe_spec"],probe_spec=case["probe_spec"],fixed_input=case["fixed_input"])
        plan["input"]["history_ages_s"]=case["fixed_input"]["history_ages_s"]
    else:
        plan["pilot_manifest_sha256"]=PILOT_SHA
    output.mkdir(exist_ok=False)
    write_json_new(output / "HEARTBEAT_BEFORE_PLAN.json", heartbeat)
    write_json_new(output / "PLAN.json", plan)
    return {"plan": str(output.resolve() / "PLAN.json"), "plan_sha256": sha256(output / "PLAN.json"),
            "source_count": len(bindings), "planned_frames": 17*frames_per_state, "hardware_accessed": False,
            "projection_executed": False, "control_release_approved": False}


def seal_capture(folder, expected_plan_sha256):
    """Seal an exited capture, including failures; never call any live API.

    Call only after the transport child and its watchdog have exited. This is
    archive identity, not a reclassification of any failed or missing frame.
    """
    root = Path(folder).resolve(strict=True)
    if (root / "MANIFEST.json").exists():
        raise FileExistsError("capture already sealed")
    if sha256(root / "PLAN.json") != expected_plan_sha256:
        raise ValueError("capture copied PLAN identity differs")
    plan, result = read_json(root / "PLAN.json"), read_json(root / "RESULT.json")
    if result.get("schema") != "optical-lab-existing-owner-pilot-capture-1":
        raise ValueError("original-window capture result required")
    if not (root / "FINISHED.json").is_file():
        raise ValueError("transport has not recorded finalization")
    if (root / "PREFLIGHT.json").exists() and not (root / "WATCHDOG_RESULT.json").is_file():
        raise ValueError("watchdog terminal evidence missing; do not seal while it may write")
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=False)
    copied = {}
    for name, digest in plan["source_sha256"].items():
        path = Path(name)
        if path.suffix == ".py" and path.resolve().is_relative_to(PROJECT):
            if sha256(path) != digest:
                raise ValueError("capture implementation changed before archival: " + name)
            target = provenance / (str(len(copied)).zfill(2) + "_" + path.name)
            with target.open("xb") as stream:
                stream.write(path.read_bytes())
            copied[name] = {"path": str(target.relative_to(root)), "sha256": digest}
    write_json_new(provenance / "NEW_CODE_SNAPSHOT.json", copied)
    if result.get("complete") is not True:
        write_json_new(root / "FAILED.json", {
            "capture_complete": False, "reason": result.get("error"), "raw_partials_preserved": True,
            "source_result_unchanged": True, "control_release_approved": False})
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("no symlinks in independent capture archive")
        if path.is_file():
            files[str(path.relative_to(root))] = sha256(path)
    manifest = {"schema": "optical-lab-original-window-capture-archive-1", "sealed_utc": utc_now(),
                "file_sha256": files, "manifest_excludes_itself": True,
                "capture_result_complete": result.get("complete") is True,
                "archive_is_not_hardware_qualification": True, "control_release_approved": False}
    write_json_new(root / "MANIFEST.json", manifest)
    return {"files": len(files), "manifest_sha256": sha256(root / "MANIFEST.json"),
            "capture_complete": result.get("complete") is True, "control_release_approved": False}
