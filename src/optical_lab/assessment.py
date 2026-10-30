"""Historical paired-data assessment only. No training, SDK or legacy imports."""
import csv
import hashlib
import platform
from pathlib import Path

from .historical import load_historical_sources, load_historical_cell
from .metrics import describe_real, compare_model, paired_response
from .reporting import cell_figure, summary_figures, write_reports
from .storage import Experiment, PROJECT, read_json, write_json_new, sha256, utc_now, safe_id


def assess(sources, protocol_path, output):
    protocol = read_json(protocol_path)
    if (protocol.get("schema") != "BALLPLATE_OPTICAL_ASSESSMENT_PROTOCOL_1"
            or protocol.get("hardware_actions_enabled") is not False):
        raise ValueError("explicit offline assessment protocol required")
    catalog = load_historical_sources(sources)
    if not catalog["cells"]:
        raise ValueError("an explicit nonempty historical cohort is required")
    code_before = {str(p.relative_to(PROJECT)): sha256(p)
                   for p in sorted((PROJECT/"src/optical_lab").glob("*.py"))}
    config_before = {str(Path(p).resolve()): sha256(p) for p in (sources, protocol_path)}
    experiment = Experiment(output)
    root = experiment.root
    write_json_new(root/"provenance"/"assessment_protocol.json", protocol)
    write_json_new(root/"provenance"/"historical_sources.json", read_json(sources))
    write_json_new(root/"provenance"/"cohort_index.json", catalog)
    write_json_new(root/"provenance"/"code_sha256.json", code_before)
    snapshot_paths = [*sorted((PROJECT/"src/optical_lab").glob("*.py")),
                      *sorted((PROJECT/"tests").glob("test_*.py")), PROJECT/"run_lab.py", PROJECT/"pyproject.toml"]
    snapshot = {}
    for p in snapshot_paths:
        blob = p.read_bytes()
        snapshot[str(p.relative_to(PROJECT))] = {
            "sha256":hashlib.sha256(blob).hexdigest(), "utf8":blob.decode("utf-8")}
    write_json_new(root/"provenance"/"code_snapshot.json", snapshot)
    import numpy as np
    import matplotlib
    write_json_new(root/"provenance"/"environment.json", {
        "python":platform.python_version(), "numpy":np.__version__, "matplotlib":matplotlib.__version__,
        "platform":platform.platform(), "GPU_used":False, "hardware_connected":False})
    cells, loaded, name_map = [], {}, {}
    try:
        for i, record in enumerate(catalog["cells"]):
            cid = safe_id("C"+str(i).zfill(2)+"_"+record["cell"])
            data = load_historical_cell(record)
            background = data.get("dark_stack") if data["metadata"].get("dark_metadata_matched") is True else None
            real = describe_real(data["real_stack"], background, data.get("regions"))
            models = {name: compare_model(data["real_stack"], array, data.get("regions"), data.get("head"), background)
                      for name, array in data["models"].items()}
            item = dict(id=cid, source_cell_id=record["id"], batch=record["batch"],
                        input_id=record["input_id"], mask_id=record["mask_id"],
                        real=real, models=models, metadata=data["metadata"], sources=data["sources"])
            experiment.archive_arrays(cid, data["real_stack"], data.get("dark_stack"), data["models"], data.get("regions"))
            cell_figure(root/"figures", cid, data["real_stack"], data["models"], data.get("regions"), protocol)
            cells.append(item)
            # Only keep the pairs' small stacks until probe calculation. No training.
            loaded[record["id"]] = data
            name_map[record["id"]] = cid
            print("assessed "+cid+" ("+str(i+1)+"/"+str(len(catalog["cells"]))+")", flush=True)
        probes = []
        for i, group in enumerate(catalog["probe_groups"]):
            a, b = loaded[group["minus_cell_id"]], loaded[group["plus_cell_id"]]
            # Both endpoints must share the exact same captured regions/input.
            import numpy as np
            if not np.array_equal(a["regions"], b["regions"]):
                raise ValueError("probe pair mixes region definitions")
            common = [name for name in a["models"] if name in b["models"]]
            if not common:
                model_a = model_b = None
                model_name = None
            else:
                # The pre-capture prediction has an explicit, fixed role; never
                # choose the most accurate post-hoc model from probe outcomes.
                if "pre_capture_frozen" not in common:
                    raise ValueError("probe model choice must be pre-capture frozen, not chosen by performance")
                model_name = "pre_capture_frozen"
                model_a, model_b = a["models"][model_name], b["models"][model_name]
            if a["metadata"]["cell"]["input"]["raw96"]["sha256"] != b["metadata"]["cell"]["input"]["raw96"]["sha256"]:
                raise ValueError("probe pair does not share one exact DMD input")
            metrics = paired_response(a["real_stack"], b["real_stack"], a["regions"], model_a, model_b)
            probes.append(dict(id="P"+str(i).zfill(2), source_pair=group,
                               minus_cell=name_map[group["minus_cell_id"]], plus_cell=name_map[group["plus_cell_id"]],
                               kind=group["kind"], model=model_name, metrics=metrics,
                               causal_pattern_frame_association_verified=False,
                               independent_hardware_qualification=False))
        summary_figures(root/"figures", cells, probes, protocol)
        write_json_new(root/"cells.json", cells)
        write_json_new(root/"probes.json", probes)
        fields = ["id", "batch", "input_id", "mask_id", "model", "NCC", "raw_RMSE_ADU", "raw_MAE_ADU",
                  "native8_raw_error_over_measured_net_signal_L2", "raw_head_mean_action_MAE_rad"]
        with (root/"metrics.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for c in cells:
                for name, metrics in c["models"].items():
                    writer.writerow({**{k:c[k] for k in ("id","batch","input_id","mask_id")}, "model":name,
                                     **{k:metrics.get(k) for k in fields[5:]}})
        for path, expected in config_before.items():
            if sha256(path) != expected:
                raise ValueError("configuration changed during assessment")
        if {str(p.relative_to(PROJECT)): sha256(p) for p in sorted((PROJECT/"src/optical_lab").glob("*.py"))} != code_before:
            raise ValueError("assessment source changed during execution")
        for relative, record in snapshot.items():
            if sha256(PROJECT/relative) != record["sha256"]:
                raise ValueError("snapshotted source or test changed during execution")
        source_ledger = {}
        for c in cells:
            for ref in c["sources"]:
                if ref["path"] in source_ledger and source_ledger[ref["path"]] != ref["sha256"]:
                    raise ValueError("conflicting historical source identity")
                source_ledger[ref["path"]] = ref["sha256"]
        for path, expected in source_ledger.items():
            if sha256(path) != expected:
                raise ValueError("historical source changed before finalization")
        write_json_new(root/"provenance"/"all_historical_source_sha256.json", source_ledger)
        signal_ids = {s["raw"]["path"] for c in catalog["cells"] for s in c["capture"]["snapshots"]}
        background_ids = {s["raw"]["path"] for c in catalog["cells"] for b in c["dark_captures"] for s in b["snapshots"]}
        result = dict(schema="optical-lab-historical-assessment-1", complete=True,
            started_utc=experiment.started, completed_utc=utc_now(),
            kind="HISTORICAL_SEEN_EXPLORATORY_NOT_FRESH_VALIDATION", cell_count=len(cells),
            real_signal_frames=sum(c["real"]["n_frames"] for c in cells), probe_pair_count=len(probes),
            unique_real_signal_frames=len(signal_ids), unique_zero_output_background_frames=len(background_ids),
            cells_with_incomplete_freshness_evidence=sum(not c["metadata"]["complete_freshness_evidence"] for c in cells),
            cells_without_matched_background=sum(not c["metadata"]["dark_metadata_matched"] for c in cells),
            cells_without_causal_frame_pattern_association=sum(c["metadata"]["frame_pattern_association_verified"] is not True for c in cells),
            background_kind="DMD_SLM_zero_optical_background_NOT_physical_sensor_dark",
            independent_input_count=len({c["metadata"]["cell"]["input"]["raw96"]["sha256"] for c in cells}),
            frame_pattern_association_proven=False, hardware_connected=False,
            production_qualified=False, hardware_release_approved=False,
            platform_modulation_efficiency_acceptable="UNKNOWN", absolute_optical_efficiency=None,
            gates={"trusted_new_capture":"UNKNOWN", "independent_sim_real_gap":"UNKNOWN",
                   "relative_modulation_sufficiency":"UNKNOWN", "absolute_efficiency":"UNKNOWN"},
            config_sha256=config_before, cells=cells, probes=probes)
        write_reports(root, result)
        experiment.finish(result)
        return result
    except Exception as exc:
        write_json_new(root/"FAILED.json", dict(error=repr(exc), created_utc=utc_now(),
            completed_cells=[c["id"] for c in cells], complete=False, originals_not_modified=True,
            hardware_connected=False, failure_records_retained=True))
        raise


# E002 deliberately has its own capture/analysis schema. The historical assess()
# above remains unchanged; real capture identity is never relabelled historical.
PILOT_ANALYSIS_SCHEMA = "optical-lab-pilot-analysis-1"
PILOT_CAPTURE_SCHEMA = "optical-lab-existing-owner-pilot-capture-1"
PILOT_MODEL_SCHEMA = "optical-lab-fixed-pilot-cpu-predictions-1"
CASE_MODEL_SCHEMA = "optical-lab-static-case-cpu-predictions-1"
PILOT_MODELS = ("M0_REFERENCE", "M1_RADIAL")
BROAD8_SHA256 = "0a76e770d8d9f2761a9edce492b4a846eab98ebf0e6f719271711202954601c5"
BROAD8_ROI = {"OffsetX": 888, "OffsetY": 416, "Width": 552, "Height": 554}


def _pilot_read(path, ledger, expected=None, role="input"):
    path = Path(path).resolve(strict=True)
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("pilot SHA mismatch: " + str(path))
    if str(path) in ledger and ledger[str(path)]["sha256"] != digest:
        raise ValueError("pilot input changed during analysis: " + str(path))
    ledger[str(path)] = {"path": str(path), "sha256": digest, "role": role}
    return blob


def _pilot_json(path, ledger, expected=None, role="metadata"):
    import json
    return json.loads(_pilot_read(path, ledger, expected, role))


def _pilot_npy(path, ledger, expected=None, role="array"):
    import io
    import numpy as np
    return np.load(io.BytesIO(_pilot_read(path, ledger, expected, role)), allow_pickle=False)


def _pilot_local(root, name):
    root = Path(root).resolve()
    name = Path(name)
    path = name if name.is_absolute() else root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("pilot artifact escapes its declared directory")
    return path.resolve()


def _pilot_reference(plan, capture, field):
    ref = plan[field]
    path = Path(ref["path"])
    return (path if path.is_absolute() else capture / path).resolve(), ref["sha256"]


def _pilot_predictions(plan, capture, prediction_dir, regions, ledger):
    """Consume exactly six already-frozen predictions, never old model imports."""
    import numpy as np
    pinned = plan.get("frozen_predictions")
    if not pinned:
        if plan.get("case_asset_manifest"):
            raise ValueError("explicit static case requires its frozen six predictions")
        return {}, {"available": False, "reason": "capture plan did not pin a prediction manifest"}
    supplied = Path(prediction_dir).resolve() if prediction_dir is not None else Path(pinned["path"])
    if not supplied.is_absolute():
        supplied = capture / supplied
    if supplied.name == "MANIFEST.json":
        supplied = supplied.parent
    root = supplied.resolve(strict=True)
    if (root / "FAILED.json").exists():
        raise ValueError("FAILED prediction output is not a completed model reference")
    declared = Path(pinned["path"])
    declared = (declared if declared.is_absolute() else capture / declared).resolve()
    if declared.name == "MANIFEST.json":
        declared = declared.parent
    if declared != root:
        raise ValueError("prediction directory differs from capture-plan identity")
    manifest = _pilot_json(root / "MANIFEST.json", ledger, pinned["manifest_sha256"], "frozen_prediction_manifest")
    prediction_schema = manifest.get("schema")
    is_case = prediction_schema == CASE_MODEL_SCHEMA
    if prediction_schema not in (PILOT_MODEL_SCHEMA, CASE_MODEL_SCHEMA) or manifest.get("completed") is not True:
        raise ValueError("completed fixed CPU prediction manifest required")
    for name, digest in manifest["file_sha256"].items():
        _pilot_read(_pilot_local(root, name), ledger, digest, "frozen_prediction_artifact")
    result = _pilot_json(root / "RESULT.json", ledger)
    if (result.get("schema") != prediction_schema or result.get("completed") is not True
            or result.get("source_and_assets_unchanged") is not True
            or result.get("G2_model_qualified") is not False
            or result.get("unreleased", {}).get("parameters_fitted") is not False):
        raise ValueError("unchanged unqualified M0/M1 prediction result required")
    protocol = _pilot_json(root / "PROSPECTIVE_PROTOCOL.json", ledger, result["protocol_sha256"])
    asset_pin = plan.get("case_asset_manifest") if is_case else None
    if is_case and (not isinstance(asset_pin, dict) or set(asset_pin) != {"path", "sha256"}):
        raise ValueError("explicit capture case_asset_manifest identity required")
    asset_sha = asset_pin["sha256"] if is_case else plan["pilot_manifest_sha256"]
    if is_case and (plan.get("profile") != "static_case_v1"
                    or plan.get("case_manifest_sha256") != asset_sha or "pilot_manifest_sha256" in plan):
        raise ValueError("explicit static_case_v1 plan required without a historical pilot alias")
    if ((not is_case and protocol.get("pilot_manifest_sha256") != asset_sha)
            or (is_case and ("pilot_manifest_sha256" in protocol or "pilot_manifest_sha256" in result))
            or protocol.get("sensor_mode") != "mean"
            or protocol.get("native_shape") != list(regions.shape[1:])
            or protocol.get("readout_family") != "ORIGINAL_BROAD_NATIVE8_and_original_diagnostic_head_NOT_W7"
            or protocol["fixed_input"]["dmd_sha256"] != plan["input"]["sha256"]):
        raise ValueError("prediction/input/asset/readout contract mismatch")
    if plan["input"].get("raw96"):
        if plan["input"]["raw96"]["sha256"] != protocol["fixed_input"]["raw_sha256"]:
            raise ValueError("prediction RAW96 identity differs")
    for model, expected in (("M0_REFERENCE", "ba9d9f69b01301fb78a8e7365a46f4f200f8612fb9b733ad70efc572184afa52"),
                             ("M1_RADIAL", "a67d37a64d52f7bfddf928ebd66e53286048f03a7176dc15ae37122413ca92a0")):
        if protocol["models"][model]["sha256"] != expected:
            raise ValueError("only the two declared frozen M0/M1 hypotheses are allowed")
    pilot = Path(protocol["pilot_directory"]).resolve(strict=True)
    if is_case:
        from . import study, probes
        expected_case_path = Path(asset_pin["path"])
        if not expected_case_path.is_absolute():
            expected_case_path = capture / expected_case_path
        expected_case_path = expected_case_path.resolve(strict=True)
        if expected_case_path != pilot:
            raise ValueError("capture and prediction case directories differ")
        verification = study.verify_case_assets(pilot, expected_manifest_sha256=asset_sha)
        if verification.get("verified") is not True or verification.get("manifest_sha256") != asset_sha:
            raise ValueError("verified externally pinned static case required")
        asset_manifest = _pilot_json(pilot / "MANIFEST.json", ledger, asset_sha)
        if verification["manifest"] != asset_manifest:
            raise ValueError("case verifier and captured manifest differ")
        expected_identity = {"case_asset_manifest": {"path": str(pilot), "sha256": asset_sha},
            "case_id": asset_manifest["design"]["case_id"], "role": asset_manifest["design"]["role"],
            "design": asset_manifest["design"], "fixed_input": asset_manifest["fixed_input"],
            "probe_spec": asset_manifest["probe_spec"]}
        for key, value in expected_identity.items():
            if protocol.get(key) != value or result.get(key) != value:
                raise ValueError("fixed case prediction identity differs: " + key)
        for key in ("case_id", "role", "design", "fixed_input", "probe_spec"):
            if plan.get(key) != expected_identity[key]:
                raise ValueError("capture case metadata differs: " + key)
        if plan.get("case_spec") != expected_identity["probe_spec"]:
            raise ValueError("capture case_spec differs from frozen probe specification")
        fixed = asset_manifest["fixed_input"]
        raw_ref = plan["input"].get("raw96")
        if (plan["input"]["sha256"] != fixed["dmd_sha256"] or not isinstance(raw_ref, dict)
                or raw_ref.get("sha256") != fixed["raw_sha256"]):
            raise ValueError("case capture must pin its exact DMD and RAW96 input")
        implementation = asset_manifest["implementation"]
        if implementation.get("snapshot") != "generator_source.py":
            raise ValueError("case generator must be its frozen local source snapshot")
        _pilot_read(_pilot_local(pilot, "generator_source.py"), ledger,
                    implementation["source_sha256"], "frozen_case_generator_source_NOT_executed")
        # These are the current verifier implementations, not a requirement that
        # the archived generator equal today's actively developed study module.
        _pilot_read(Path(study.__file__), ledger, role="case_verifier_source")
        _pilot_read(Path(probes.__file__), ledger, role="case_PNG_verifier_helper_source")
        profile = dict(family="explicit_static_case", **expected_identity)
    else:
        if plan.get("case_asset_manifest"):
            raise ValueError("explicit case cannot silently use the historical pilot prediction schema")
        from . import probes
        probes.verify_pilot_assets(pilot, expected_manifest_sha256=asset_sha)
        asset_manifest = _pilot_json(pilot / "MANIFEST.json", ledger, asset_sha)
        _pilot_read(Path(probes.__file__), ledger, asset_manifest["implementation"]["source_sha256"],
                    "verified_pilot_generator_source")
        profile = {"family": "original_fixed_E002_pilot", "case_id": "E002", "role": "pilot",
                   "probe_spec": asset_manifest["pilot_spec"]}
    # The verifier actually reads these small assets and its implementation.
    # Include them in this analysis' final unchanged check, not merely the
    # inherited model provenance; no legacy RAW or model execution is needed.
    for name, digest in asset_manifest["file_sha256"].items():
        _pilot_read(_pilot_local(pilot, name), ledger, digest, "verified_pilot_asset")
    masks = plan["masks"]
    masks = {m["id"]: m for m in masks} if isinstance(masks, list) else masks
    for state in ("Z", "A", "B"):
        name = asset_manifest["assets"][state]["npy"]
        if masks[state]["sha256"] != asset_manifest["file_sha256"][name]:
            raise ValueError("absolute capture mask differs from frozen prediction mask")
    rows = result["rows"]
    if [(r["model"], r["mask"]) for r in rows] != [(m, k) for m in PILOT_MODELS for k in ("Z", "A", "B")]:
        raise ValueError("all six fixed M0/M1 Z/A/B predictions required")
    predictions = {m: {} for m in PILOT_MODELS}
    for row in rows:
        refs = row["files"]
        a = _pilot_npy(_pilot_local(root, refs["NATIVE_CCD_MEAN_ADU"]["path"]), ledger,
                       refs["NATIVE_CCD_MEAN_ADU"]["sha256"], "model_expected_ADU_NOT_real_CCD")
        f = _pilot_npy(_pilot_local(root, refs["NATIVE8_FEATURES"]["path"]), ledger,
                       refs["NATIVE8_FEATURES"]["sha256"], "model_original_native8")
        if (a.shape != regions.shape[1:] or not np.issubdtype(a.dtype, np.floating)
                or not np.isfinite(a).all() or np.any((a < 0) | (a > 255))):
            raise ValueError("native floating expected-code ADU model required")
        direct = np.einsum("khw,hw->k", regions, a.astype(float))
        if f.shape != (8,) or not np.allclose(f, direct, atol=1e-7, rtol=1e-12):
            raise ValueError("frozen prediction native8 does not match BROAD8")
        predictions[row["model"]][row["mask"]] = a
    return predictions, {"available": True, "capture_plan_pinned_manifest": True,
        "manifest_sha256": pinned["manifest_sha256"], "models": list(PILOT_MODELS),
        "parameters_refitted": False, "legacy_prediction_provenance_inherited_NOT_model_reexecuted": protocol["source_and_asset_sha256"],
        "pilot_assets_and_generator_reverified_in_analysis_ledger": True,
        "case_profile": profile,
        "limitations": result.get("limitations", [])}


def _pilot_contrast_metrics(image, regions, model=None):
    import numpy as np
    from .metrics import ratio
    image = np.asarray(image, float)
    f = np.einsum("khw,hw->k", regions, image)
    out = {"image_contrast_RMS_ADU": float(np.sqrt(np.mean(image**2))),
           "native8_contrast_ADU_sum": f.tolist(), "native8_contrast_norm_ADU_sum": float(np.linalg.norm(f))}
    if model is not None:
        model = np.asarray(model, float)
        p = np.einsum("khw,hw->k", regions, model)
        out.update(model_native8_contrast_ADU_sum=p.tolist(),
            image_model_contrast_RMSE_ADU=float(np.sqrt(np.mean((image-model)**2))),
            native8_model_contrast_relative_L2=ratio(np.linalg.norm(p-f), np.linalg.norm(f)),
            native8_model_contrast_cosine=ratio(float(p @ f), float(np.linalg.norm(p)*np.linalg.norm(f))))
    return out


def _pilot_repeat_count(plan):
    """Static sample count, never the controller's four input image history."""
    import json
    from .protocol import bracketed_states
    is_case = plan.get("profile") == "static_case_v1"
    repeats = plan.get("frames_per_state", 16)
    if type(repeats) is not int or repeats not in (8, 16) or (not is_case and repeats != 16):
        raise ValueError("only explicit static cases may use integer8; legacy pilot remains integer16")
    if type(plan.get("total_planned_frames")) is not int or plan["total_planned_frames"] != 17*repeats:
        raise ValueError("static PLAN total must equal seventeen times its explicit repeats")
    canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
    if is_case and canonical(plan.get("states")) != canonical(bracketed_states(repeats)):
        raise ValueError("static case requires the shared exact17 bracketed_states(repeats) sequence")
    return repeats


def _pilot_states(plan, capture_result, attempts, capture, regions, ledger):
    import numpy as np
    from .metrics import describe_real
    repeats = _pilot_repeat_count(plan)
    planned = plan["states"]
    if (len(planned) != 17 or len({s["id"] for s in planned}) != 17
            or sum(s["kind"] == "BG" for s in planned) != 2
            or planned[0]["kind"] != "BG" or planned[-1]["kind"] != "BG"
            or any(type(s.get("repeat", s.get("repeat_expected", 16))) is not int
                   or s.get("repeat", s.get("repeat_expected", 16)) != repeats for s in planned)):
        raise ValueError("all17 states must have the same explicitly prescribed static repeats")
    if (("frames_per_state" in capture_result and (type(capture_result["frames_per_state"]) is not int or capture_result["frames_per_state"] != repeats))
            or ("expected_frames" in capture_result and (type(capture_result["expected_frames"]) is not int or capture_result["expected_frames"] != 17*repeats))):
        raise ValueError("capture result static denominator differs from its frozen PLAN")
    blocks = list(dict.fromkeys(s["time_block"] for s in planned if s["kind"] != "BG"))
    if len(blocks) != 3:
        raise ValueError("three declared time blocks required")
    for block in blocks:
        kinds = [s["kind"] for s in planned if s["time_block"] == block and s["kind"] != "BG"]
        if kinds not in (["Z", "A", "Z", "B", "Z"], ["Z", "B", "Z", "A", "Z"]):
            raise ValueError("each time block requires Z/probe/Z/probe/Z")
    reported = {s["id"]: s for s in capture_result.get("states", [])}
    if len(reported) != len(capture_result.get("states", [])) or set(reported) - {s["id"] for s in planned}:
        raise ValueError("duplicate or undeclared capture states")
    ids = [a["id"] for a in attempts]
    if len(set(ids)) != len(ids) or any(a["state_id"] not in {s["id"] for s in planned} for a in attempts):
        raise ValueError("duplicate attempt or undeclared attempted state")
    active = plan["camera_expected"]["active"]
    height, width = regions.shape[1:]
    started = [a.get("host_get_started_unix_ns") for a in attempts if isinstance(a.get("host_get_started_unix_ns"), int)]
    origin = min(started) if started else None
    seen_raw, seen_frame = set(), set()
    states, arrays, means, last_id, last_tick = [], {}, {}, -1, -1
    for state in planned:
        sid = state["id"]
        observed = reported.get(sid, {})
        if observed and (observed.get("kind") != state["kind"] or observed.get("time_block") != state["time_block"]):
            raise ValueError("capture state kind/time-block identity differs from frozen PLAN")
        rows = [a for a in attempts if a["state_id"] == sid]
        pixels, times, rejected = [], [], []
        revision = observed.get("display_revision")
        for attempt in rows:
            if attempt.get("status") != "valid" or not attempt.get("raw") or not attempt.get("metadata"):
                rejected.append({"id": attempt["id"], "reason": attempt.get("error") or "not a valid saved snapshot"})
                continue
            rawref, metaref = attempt["raw"], attempt["metadata"]
            path = _pilot_local(capture, rawref["path"])
            if str(path) in seen_raw:
                raise ValueError("same captured raw path counted more than once")
            seen_raw.add(str(path))
            raw = _pilot_npy(path, ledger, rawref["sha256"], "real_single_CCD_uint8_no_copy")
            meta = _pilot_json(_pilot_local(capture, metaref["path"]), ledger, metaref["sha256"])
            if raw.dtype != np.uint8 or raw.shape != (height, width):
                raise ValueError("native uint8 raw CCD required; no implicit crop/cast/resize")
            frame = meta.get("frame", {})
            fid, tick, host = frame.get("FrameID"), frame.get("TimeStamp"), frame.get("host_get_started_unix_ns")
            minimum_wait = plan.get("minimum_postcommit_frame_start_seconds")
            wait_valid = (isinstance(minimum_wait, (int, float)) and not isinstance(minimum_wait, bool)
                          and np.isfinite(minimum_wait) and minimum_wait > 0)
            software_settle_ns = int(float(minimum_wait) * 1e9) if wait_valid else None
            verified = observed.get("verified_unix_ns")
            checks = [meta.get("camera_active") == active, attempt.get("camera_actual") == active,
                attempt.get("kind") == state["kind"], attempt.get("time_block") == state["time_block"],
                frame == attempt.get("frame"), attempt.get("host_get_started_unix_ns") == host,
                meta.get("display_revision") == revision == attempt.get("display_revision"),
                frame.get("Width") == width, frame.get("Height") == height, frame.get("Stride") == width,
                frame.get("BufferSize") == width*height, frame.get("pixel_format") == "Mono8",
                frame.get("IsIncomplete") is not True,
                isinstance(fid, int) and fid > last_id, isinstance(tick, int) and tick > last_tick,
                wait_valid and isinstance(host, int) and isinstance(verified, int) and host >= verified + software_settle_ns,
                int(raw.max()) < 240]
            key = (fid, tick)
            if key in seen_frame:
                checks.append(False)
            if not all(checks):
                rejected.append({"id": attempt["id"], "reason": "independent camera/frame/freshness/layout/max-ADU gate"})
                continue
            seen_frame.add(key)
            last_id, last_tick = fid, tick
            pixels.append(raw)
            times.append(float(host-origin))
        count = len(pixels)
        gate_issues = []
        if [a.get("repeat_index") for a in rows] != list(range(repeats)):
            gate_issues.append(f"repeat indices are not the complete prescribed 0..{repeats-1} sequence")
        if "repeat_expected" in observed and (type(observed["repeat_expected"]) is not int or observed["repeat_expected"] != repeats):
            gate_issues.append("reported per-state denominator differs from the frozen PLAN")
        if observed.get("attempt_ids") != [a["id"] for a in rows]:
            gate_issues.append("state attempt_ids disagree with the immutable journal")
        readback = observed.get("display_readback", {})
        display_verified = bool(isinstance(readback, dict) and all(
            isinstance(readback.get(role), dict) and readback[role].get("matches") is True
            and readback[role].get("mismatched_pixels") == 0 for role in ("dmd", "slm")))
        if not display_verified:
            gate_issues.append("both desktop RGB comparisons must have matches=True and mismatched_pixels=0")
        eligible = bool(count == repeats and len(rows) == repeats and not rejected and observed.get("status") == "complete"
                        and observed.get("valid_frame_count") == count and display_verified and not gate_issues)
        item = {"id": sid, "kind": state["kind"], "time_block": state["time_block"],
            "planned_repeats": repeats, "attempts": len(rows), "valid_frames": count, "rejected": rejected,
            "reported_status": observed.get("status", "not_started"), "numeric_eligible": eligible,
            "state_gate_issues": gate_issues,
            "minimum_postcommit_frame_start_seconds": plan.get("minimum_postcommit_frame_start_seconds"),
            "display_verification": "desktop RGB compare to expected arrays; NOT independent physical-panel readback hash",
            "effective_host_time_relative_ns": float(np.mean(times)) if times else None,
            "effective_time_is_verified_exposure_time": False, "complete_causal_frame_pattern_association": False,
            "stats": None}
        if pixels:
            stack = np.stack(pixels)
            arrays[sid] = stack
            means[sid] = stack.mean(0)
            if count >= 2:
                item["stats"] = describe_real(stack, regions=regions)
        states.append(item)
    return states, arrays, means, blocks, origin


def _render_pilot(output, states, contrasts, images, regions, display):
    """Scientific data plots using the same project style, no GUI rendering."""
    import numpy as np
    from .reporting import plotting, save_figure
    plt = plotting()
    dpi = display.get("dpi", 150)
    case_label = display.get("analysis_case_label", "E002")
    repeats = display.get("frames_per_state", 16)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for axis, key, label in ((ax[0], "raw_mean_ADU", "Raw native mean / ADU"),
                              (ax[1], "single_frame_repeat_pixel_RMS_ADU", "Within-state repeat RMS / ADU")):
        axis.plot([s["stats"].get(key) if s["stats"] and s["numeric_eligible"] else np.nan for s in states], "o-")
        axis.set_ylabel(label)
    ax[1].set_xticks(range(len(states)), [s["id"] for s in states], rotation=70, ha="right", fontsize=6)
    fig.suptitle(case_label + f" | all17 states / {17*repeats} planned samples | native ADU | missing states remain NA")
    fig.tight_layout(); save_figure(fig, output / "figures/01_all17_states", dpi); plt.close(fig)
    fig, axes = plt.subplots(6, 1, figsize=(9, 13), squeeze=False)
    for axis, row in zip(axes[:, 0], contrasts):
        axis.set_title(row["id"] + " / " + row["status"], fontsize=8)
        if row["metrics"] is None:
            axis.text(.5, .5, "NA: " + row["reason"], ha="center", va="center", transform=axis.transAxes)
        else:
            axis.plot(row["metrics"]["native8_contrast_ADU_sum"], "o-", color="#222222", label="Measured probe - bracketed Z")
            for name, metrics in row["model_comparisons"].items():
                axis.plot(metrics["model_native8_contrast_ADU_sum"], "s--", label=name)
            axis.legend(frameon=False, fontsize=7)
        axis.axhline(0, color=".7", lw=.5); axis.set_ylabel("ADU sum")
    fig.suptitle(f"All6 mean-level contrasts / {repeats} samples per state\nshared Z is NOT independent repetitions | no qualification")
    fig.tight_layout(); save_figure(fig, output / "figures/02_all6_native8_contrasts", dpi); plt.close(fig)
    fig, axes = plt.subplots(6, 3, figsize=(10, 17), squeeze=False)
    limit = display.get("difference_adu_limit", 5)
    for i, row in enumerate(contrasts):
        for j, name in enumerate(("measured", *PILOT_MODELS)):
            value = images.get(row["id"] + "_" + name)
            if value is None:
                axes[i, j].text(.5, .5, "NA", ha="center", va="center", transform=axes[i, j].transAxes)
            else:
                axes[i, j].imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[i, j].set_title(row["id"] + " / " + name, fontsize=7)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    fig.suptitle(f"Probe - Z / static mean N={repeats} / fixed ±{limit} ADU\nmodel is NOT real CCD | no per-image normalization")
    fig.tight_layout(rect=(0, 0, 1, .96)); save_figure(fig, output / "figures/03_all6_contrast_images", dpi); plt.close(fig)


def assess_pilot(capture_dir, output, *, expected_capture_manifest_sha256=None):
    """Offline analysis of new E002 static capture; no training/device imports.

    Raw stays in capture_dir. Partial captures are retained with all17/all6
    planned denominators and NA; a FAILED capture is never upgraded to success.
    Only mean-level contrasts are constructed; no repeated reference is forged.
    """
    import json
    import numpy as np
    from .metrics import bracketed_contrast, validate_regions
    from .reporting import number
    if output is None:
        raise ValueError("explicit NEW analysis output required")
    capture = Path(capture_dir).resolve(strict=True)
    out = Path(output)
    if out.resolve().is_relative_to(capture):
        raise ValueError("analysis output must not modify the source capture tree")
    if out.exists() or out.is_symlink():
        raise FileExistsError("new pilot analysis directory required")
    out.mkdir(exist_ok=False)
    for name in ("derived", "figures", "provenance"):
        (out / name).mkdir()
    ledger = {}
    try:
        # Freeze all capture artifacts before pixel decoding, including orphan /
        # partial files, without inventing a state association for those files.
        for path in sorted(capture.rglob("*")):
            if path.is_file():
                _pilot_read(_pilot_local(capture, path), ledger, role="capture_artifact")
        seal_audit = {"present": (capture / "MANIFEST.json").exists(), "external_identity_checked": False}
        if expected_capture_manifest_sha256 is not None or seal_audit["present"]:
            seal = _pilot_json(capture / "MANIFEST.json", ledger, expected_capture_manifest_sha256, "sealed_capture_manifest")
            entries = seal["file_sha256"]
            actual = {str(p.relative_to(capture)) for p in capture.rglob("*") if p.is_file() and p != capture / "MANIFEST.json"}
            if set(entries) != actual:
                raise ValueError("sealed capture file list changed")
            for name, digest in entries.items():
                _pilot_read(_pilot_local(capture, name), ledger, digest, "sealed_capture_artifact")
            seal_audit["external_identity_checked"] = expected_capture_manifest_sha256 is not None
        plan = _pilot_json(capture / "PLAN.json", ledger)
        if plan.get("schema") != "optical-lab-existing-owner-pilot-plan-1":
            raise ValueError("explicit fixed static capture PLAN schema required")
        repeats = _pilot_repeat_count(plan)
        sequence_source = Path(__file__).with_name("protocol.py").resolve()
        expected_sequence_sha = plan.get("source_sha256", {}).get(str(sequence_source))
        if repeats == 8 and expected_sequence_sha is None:
            raise ValueError("new8 static cases must bind the shared sequence source SHA")
        _pilot_read(sequence_source, ledger, expected_sequence_sha, "shared_static_sequence_source")
        result = _pilot_json(capture / "RESULT.json", ledger) if (capture / "RESULT.json").exists() else {}
        if result and result.get("schema") != PILOT_CAPTURE_SCHEMA:
            raise ValueError("new existing-owner pilot capture schema required, not historical assess")
        events_present = (capture / "events.jsonl").exists()
        events, malformed_events = [], []
        event_blob = _pilot_read(capture / "events.jsonl", ledger) if events_present else b""
        for line_no, line in enumerate(event_blob.splitlines(), 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("event row must be an object")
                events.append(event)
            except (ValueError, TypeError) as exc:
                malformed_events.append({"line": line_no, "error": str(exc), "bytes_sha256": hashlib.sha256(line).hexdigest()})
        attempts, malformed = [], []
        attempts_present = (capture / "attempts.jsonl").exists()
        blob = _pilot_read(capture / "attempts.jsonl", ledger) if attempts_present else b""
        for line_no, line in enumerate(blob.splitlines(), 1):
            try:
                attempt = json.loads(line)
                if not isinstance(attempt, dict):
                    raise ValueError("attempt row must be an object")
                attempts.append(attempt)
            except (ValueError, TypeError) as exc:
                malformed.append({"line": line_no, "error": str(exc), "bytes_sha256": hashlib.sha256(line).hexdigest()})
        frame_intents = sum(event.get("event") == "frame_attempt_intent" for event in events)
        missing_expected = bool(not attempts_present and events_present and not malformed_events and not frame_intents
            and result.get("complete") is not True and result.get("captured_valid_frames", 0) == 0
            and not any(state.get("attempt_ids") for state in result.get("states", []))
            and not any(capture.rglob("CCD_raw.npy")))
        attempt_journal = {"path": "attempts.jsonl", "present": attempts_present,
            "expected_missing": missing_expected, "recorded_attempt_count": len(attempts),
            "recorded_frame_attempt_intent_count": frame_intents,
            "absence_reason": ("capture stopped before any recorded frame request; original journal was never created"
                if missing_expected else "missing journal cannot be established as pre-frame failure" if not attempts_present else None),
            "missing_journal_created_by_analysis": False}
        for file in (Path(__file__), Path(__file__).with_name("metrics.py"), Path(__file__).with_name("reporting.py")):
            _pilot_read(file, ledger, role="analysis_source")
        region_path, region_sha = _pilot_reference(plan, capture, "regions")
        if region_sha != BROAD8_SHA256:
            raise ValueError("E002 requires the original fixed BROAD8 identity, not W7 or a refit readout")
        regions = _pilot_npy(region_path, ledger, region_sha, "original_BROAD8")
        roi = plan["camera_expected"]["roi"]
        if roi != BROAD8_ROI:
            raise ValueError("fixed BROAD8 native ROI identity differs; no silent translation")
        regions = validate_regions(regions, (int(roi["Height"]), int(roi["Width"])))
        predictions, prediction_audit = _pilot_predictions(plan, capture, None, regions, ledger)
        profile = prediction_audit.get("case_profile", {"family": "unspecified_no_frozen_prediction",
            "case_id": "E002", "role": "unknown", "probe_spec": None})
        is_static_case = profile["family"] == "explicit_static_case"
        scope = ("one fixed input and exactly the declared static case probe_spec only; NOT complete G1/G2"
                 if is_static_case else "one fixed input, macro16/P128/X/raw0-64 only; NOT complete G1/G2")
        states, stacks, means, blocks, origin = _pilot_states(plan, result, attempts, capture, regions, ledger)
        by_id = {s["id"]: s for s in states}
        contrasts, derived, raw_ab, drift = [], {}, [], []
        absolute = []
        for state in states:
            sid, kind = state["id"], state["kind"]
            if sid in means:
                derived[sid + "_measured_mean_ADU"] = means[sid]
            if kind != "BG":
                absolute.append({"state_id": sid, "numeric_eligible": state["numeric_eligible"],
                    "models": ({m: compare_model(stacks[sid], p[kind], regions=regions) for m, p in predictions.items()}
                               if state["numeric_eligible"] else None),
                    "reason_if_NA": None if state["numeric_eligible"] else "incomplete or unverified prescribed state"})
        for block_index, block in enumerate(blocks):
            sequence = [s for s in states if s["time_block"] == block and s["kind"] != "BG"]
            zrows = [s for s in sequence if s["kind"] == "Z"]
            drift.append({"time_block": block, "z_states": [s["id"] for s in zrows],
                "all_Z_complete": all(s["numeric_eligible"] for s in zrows),
                "Z_last_minus_first": (_pilot_contrast_metrics(means[zrows[-1]["id"]]-means[zrows[0]["id"]], regions)
                                       if all(s["numeric_eligible"] for s in zrows) else None),
                "is_noise_null_or_independence_test": False})
            block_contrasts = {}
            for kind in ("A", "B"):
                probe = next(s for s in sequence if s["kind"] == kind)
                position = sequence.index(probe)
                before, after = sequence[position-1], sequence[position+1]
                cid = f"TB{block_index}_{kind}_minus_bracketed_Z"
                row = {"id": cid, "time_block": block, "kind": kind, "probe_state": probe["id"],
                    "Z_before": before["id"], "Z_after": after["id"], "status": "NA", "reason": None,
                    "weights_before_after": None, "metrics": None, "model_comparisons": {},
                    "bracket_Z_after_minus_before": None,
                    "mean_level_only": True, "shared_Z_references_NOT_independent_observations": True}
                if not all(s["numeric_eligible"] for s in (probe, before, after)):
                    row["reason"] = f"probe or bracketing Z not a complete valid{repeats} state"
                else:
                    try:
                        contrast = bracketed_contrast(means[probe["id"]], probe["effective_host_time_relative_ns"],
                            means[before["id"]], before["effective_host_time_relative_ns"],
                            means[after["id"]], after["effective_host_time_relative_ns"])
                    except ValueError as exc:
                        row["reason"] = str(exc)
                    else:
                        measured = contrast["contrast"]
                        row.update(status="descriptive", weights_before_after=contrast["weights_before_after"],
                                   metrics=_pilot_contrast_metrics(measured, regions),
                                   bracket_Z_after_minus_before=_pilot_contrast_metrics(means[after["id"]]-means[before["id"]], regions))
                        derived[cid + "_measured"] = measured
                        block_contrasts[kind] = measured
                        for model, p in predictions.items():
                            modeled = p[kind] - p["Z"]
                            row["model_comparisons"][model] = _pilot_contrast_metrics(measured, regions, modeled)
                            derived[cid + "_" + model] = modeled
                contrasts.append(row)
            a, b = (next(s for s in sequence if s["kind"] == kind) for kind in ("A", "B"))
            raw_ab.append({"time_block": block, "direction": "A_minus_B", "time_drift_removed": False,
                "raw_AB": (_pilot_contrast_metrics(means[a["id"]]-means[b["id"]], regions)
                           if a["numeric_eligible"] and b["numeric_eligible"] else None),
                "raw_AB_model_comparisons": ({m: _pilot_contrast_metrics(means[a["id"]]-means[b["id"]], regions, p["A"]-p["B"])
                                              for m, p in predictions.items()}
                                             if a["numeric_eligible"] and b["numeric_eligible"] else None),
                "individually_Z_referenced_A_minus_B": (_pilot_contrast_metrics(block_contrasts["A"]-block_contrasts["B"], regions)
                           if set(block_contrasts) == {"A", "B"} else None)})
        background = [s for s in states if s["kind"] == "BG"]
        background_drift = (_pilot_contrast_metrics(means[background[-1]["id"]]-means[background[0]["id"]], regions)
                            if all(s["numeric_eligible"] for s in background) else None)
        capture_complete = bool(result.get("complete") is True and not (capture / "FAILED.json").exists())
        complete = bool(capture_complete and events_present and attempts_present and not malformed and not malformed_events
                        and all(s["numeric_eligible"] for s in states)
                        and all(r["metrics"] is not None for r in contrasts))
        report = {"schema": PILOT_ANALYSIS_SCHEMA, "analysis_complete": True, "capture_complete": capture_complete,
            "planned_capture_and_analysis_complete": complete, "planned_states": 17, "planned_frames": 17*repeats,
            "frames_per_state": repeats, "static_repeats_are_NOT_controller_input_history_length": True,
            "controller_input_history_images": 4,
            "planned_primary_contrasts": 6, "available_primary_contrasts": sum(r["metrics"] is not None for r in contrasts),
            "valid_frames": sum(s["valid_frames"] for s in states), "attempted_frame_count": len(attempts),
            "malformed_attempt_journal_entries": malformed, "states": states, "contrasts": contrasts,
            "event_journal_available": events_present, "malformed_event_journal_entries": malformed_events,
            "attempt_journal_available": attempts_present, "attempt_journal_expected_missing": missing_expected,
            "attempt_journal": attempt_journal, "capture_seal": seal_audit,
            "capture_FAILED_present": (capture / "FAILED.json").exists(), "capture_error": result.get("error"),
            "capture_summary_metadata": {k: v for k, v in result.items() if k != "states"},
            "raw_AB_descriptions": raw_ab, "Z_drift": drift, "zero_output_background_drift": background_drift,
            "absolute_state_model_comparisons": absolute, "predictions": prediction_audit,
            "case_profile": profile,
            "time_clock": "host_get_started_unix_ns, relative common origin; NOT verified exposure timestamp",
            "time_origin_unix_ns": origin, "independent_time_blocks_proven": False,
            "capture_inter_block_wait_seconds": plan.get("inter_block_wait_seconds"),
            "shared_Z_and_within_state_frames_NOT_independent": True, "noise_null_matching_proven": False,
            "physical_sensor_dark_taken": False, "background_kind": "DMD_SLM_zero_optical_background_NOT_physical_sensor_dark",
            "raw_copied_into_analysis": False, "model_or_head_or_parameters_fitted": False,
            "online_electronic_layer_added": False, "analysis_hardware_accessed": False, "GPU_used": False,
            "gate_states": {"G0_acquisition": "UNKNOWN", "G1_modulation": "UNKNOWN", "G2_model": "UNKNOWN", "absolute_efficiency": "UNKNOWN"},
            "production_qualified": False, "hardware_release_approved": False,
            "limitations": [scope,
                "no matched-time null or statistical independence established", "no physical dark or power calibration",
                "model mean is NOT real single CCD; no per-image fitting or normalization", "all failed/partial planned states remain in denominator"]}
        with (out / "derived/mean_level_ADU_arrays.npz").open("xb") as stream:
            np.savez_compressed(stream, **derived)
        display = dict(plan.get("display", {}))
        display["analysis_case_label"] = profile["case_id"] + " / " + profile["role"]
        display["frames_per_state"] = repeats
        _render_pilot(out, states, contrasts, derived, regions, display)
        lines = ["# 原生ADU静态case：离线分析", "",
            f"case：{profile['case_id']}；角色：{profile['role']}。", "",
            "声明probe_spec：`" + json.dumps(profile["probe_spec"], ensure_ascii=False, sort_keys=True) + "`", "",
            "仅分析已保存帧，不连接硬件、不训练、不拟合。原始raw未复制到分析目录。", "",
            f"计划17状态×每状态{repeats}张CCD＝{17*repeats}帧／6个主contrast；有效帧{report['valid_frames']}，可算contrast {report['available_primary_contrasts']}/6。",
            "静态重复采样数不等于控制器输入历史：网络仍为4幅RAW96→当前CCD→8积分。",
            f"capture完整：{capture_complete}；计划采集及分析完整：{complete}。partial/FAILED不升级成功。", "",
            f"原attempts.jsonl存在：{attempts_present}；预期的零请求前缺失：{missing_expected}。分析不补造原journal。", "",
            "时间用host_get_started同一clock的相对时间，不是已证曝光时刻。A/B分别减前后Z时间插值；共享Z不可当独立重复，未构造伪reference stack。",
            "所有qualification均UNKNOWN。" + scope + "；无物理dark、绝对效率或20ms闭环结论。", "",
            "| State | Kind | 有效/计划 | 数值完整 |", "|---|---|---:|---|"]
        lines += [f"| {s['id']} | {s['kind']} | {s['valid_frames']}/{repeats} | {s['numeric_eligible']} |" for s in states]
        lines += ["", "| Contrast | 状态 | 模型 | 8区方向余弦 | 相对L2误差 |", "|---|---|---|---:|---:|"]
        for row in contrasts:
            if not row["model_comparisons"]:
                lines.append(f"| {row['id']} | {row['status']} | NA | NA | NA |")
            for model, values in row["model_comparisons"].items():
                lines.append(f"| {row['id']} | {row['status']} | {model} | {number(values.get('native8_model_contrast_cosine'))} | {number(values.get('native8_model_contrast_relative_L2'))} |")
        lines += ["", "![全部状态](figures/01_all17_states.png)", "![全部contrast](figures/02_all6_native8_contrasts.png)",
                  "![固定ADU图像](figures/03_all6_contrast_images.png)"]
        with (out / "总结.md").open("x", encoding="utf8") as stream:
            stream.write("\n".join(lines) + "\n")
        # Every file actually read (including raw/metadata/model files) must still
        # match. Inherited legacy model references are explicitly not reread.
        for ref in list(ledger.values()):
            _pilot_read(ref["path"], ledger, ref["sha256"], ref["role"])
        if not attempts_present and (capture / "attempts.jsonl").exists():
            raise ValueError("attempt journal appeared during analysis; missing-journal evidence changed")
        write_json_new(out / "provenance/source_sha256.json", list(ledger.values()))
        write_json_new(out / "RESULT.json", report)
        files = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file()}
        write_json_new(out / "MANIFEST.json", {"schema": PILOT_ANALYSIS_SCHEMA, "analysis_complete": True,
            "file_sha256": files, "manifest_excludes_itself": True, "production_qualified": False,
            "hardware_release_approved": False})
        return report
    except Exception as exc:
        write_json_new(out / "FAILED.json", {"schema": PILOT_ANALYSIS_SCHEMA, "analysis_complete": False,
            "error": repr(exc), "partial_analysis_files_preserved": True, "capture_not_modified": True,
            "hardware_release_approved": False})
        raise


def assess_inputs(bank, output, expected_manifest_sha256):
    """Read-only four-input CPU-bank audit and 4x4 RAW96 publication figure.

    No saved source snapshot is imported/executed, and no original legacy file
    is reopened. This checks archived identities/metadata, not a re-render or
    free-dynamics/real-CCD experiment. assess_pilot's behavior is unchanged.
    """
    import json
    import re
    import numpy as np
    from .reporting import plotting, save_figure

    schema, bank_schema = "optical-lab-prescribed-input-audit-1", "optical-lab-prescribed-input-bank-1"
    source = Path(bank).resolve(strict=True)
    out = Path(output)
    if out.resolve().is_relative_to(source):
        raise ValueError("input audit output must not modify the source bank")
    if out.exists() or out.is_symlink():
        raise FileExistsError("new input audit output required")
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("external exact input-bank manifest SHA256 required")
    out.mkdir(exist_ok=False)
    ledger = {}
    try:
        for path in (Path(__file__), Path(__file__).with_name("reporting.py"), Path(__file__).with_name("storage.py")):
            _pilot_read(path, ledger, role="input_audit_source")
        manifest = _pilot_json(source / "MANIFEST.json", ledger, expected_manifest_sha256, "CPU_input_bank_manifest")
        if (manifest.get("schema") != bank_schema or manifest.get("completed") is not True
                or manifest.get("unreleased", {}).get("hardware_release_approved") is not False):
            raise ValueError("complete unreleased CPU input bank required")
        inventory = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file() and p != source / "MANIFEST.json"}
        if inventory != set(manifest["file_sha256"]) or (source / "FAILED.json").exists():
            raise ValueError("exact completed input bank inventory required")
        for name, digest in manifest["file_sha256"].items():
            _pilot_read(_pilot_local(source, name), ledger, digest, "CPU_input_bank_artifact")
        result = _pilot_json(source / "INPUTS.json", ledger)
        protocol = _pilot_json(source / "PROSPECTIVE_PROTOCOL.json", ledger)
        releases = manifest["unreleased"]
        if (result.get("unreleased") != releases or protocol.get("unreleased") != releases
                or releases.get("captured_frames") != 0 or any(releases.get(k) is not False for k in (
                    "hardware_accessed", "projection_executed", "parameters_fitted", "hardware_release_approved",
                    "control_release_approved", "production_qualified"))):
            raise ValueError("CPU input bank must retain its original no-hardware/no-fit/no-release boundary")
        protocol_ref = {"path": str(source / "PROSPECTIVE_PROTOCOL.json"),
                        "sha256": manifest["file_sha256"]["PROSPECTIVE_PROTOCOL.json"]}
        if (result.get("schema") != bank_schema or result.get("completed") is not True
                or result.get("CUDA_initialized") is not False or result.get("source_and_assets_unchanged") is not True
                or result.get("source_generation_protocol") != protocol_ref
                or protocol.get("schema") != bank_schema or protocol.get("free_dynamics_rollout") is not False
                or protocol.get("backend") != "CPU_OSMesa_software_threads2_no_CUDA"):
            raise ValueError("explicit completed CPU prescribed, not free-dynamics input bank required")
        snapshots = protocol["frozen_source_and_asset_snapshots"]
        required_snapshots = {"generator_source.py", "model.xml", "seen_input_keys.json", "RAW_resize_model96.py",
            "RAW_runtime96.py", "ENV_env_ballplate_env.py", "ENV_env_ball_camera.py", "ENV_env_reward.py", "ENV_config.py"}
        if set(snapshots) != required_snapshots or result["frozen_source_and_asset_snapshots"] != snapshots:
            raise ValueError("complete frozen input source/XML/seen snapshot table required")
        for name, ref in snapshots.items():
            if (_pilot_local(source, ref["path"]) != source / name
                    or ref["sha256"] != manifest["file_sha256"][name]
                    or ref["original_source_sha256"] != ref["sha256"]
                    or protocol["snapshot_sources"][name] != ref["original_source_path"]
                    or protocol["source_and_asset_sha256"].get(ref["path"]) != ref["sha256"]
                    or protocol["source_and_asset_sha256"].get(ref["original_source_path"]) != ref["sha256"]):
                raise ValueError("input source snapshot association mismatch: " + name)
        seen = _pilot_json(source / "seen_input_keys.json", ledger)
        seen_keys = seen.get("keys")
        if (seen.get("schema") != "optical-lab-seen-input-keys-1" or not isinstance(seen_keys, list)
                or len(set(seen_keys)) != len(seen_keys)
                or any(not isinstance(k, str) or not re.fullmatch(r"[0-9a-f]{64}", k) for k in seen_keys)):
            raise ValueError("frozen unique seen-pixel-key index required")
        seen_keys = set(seen_keys)
        rows, specifications = result["rows"], protocol["design"]["trajectories"]
        if (len(rows) != 4 or len(specifications) != 4 or [r["id"] for r in rows] != [s["id"] for s in specifications]
                or len({r["id"] for r in rows}) != 4 or result["attempted_trajectories"] != 4
                or result["requested_trajectories"] != 4 or result["rendered_pose_count"] != 40):
            raise ValueError("all four predeclared input trajectories / forty prescribed poses required")
        audited, arrays = [], []
        for row, spec in zip(rows, specifications):
            if (row["input_id"] != row["id"] or row["role"] != spec["role"]
                    or row["query_xy_m"] != spec["query_xy_m"]
                    or row["prescribed_velocity_xy_m_s"] != spec["velocity_xy_m_s"]
                    or row["source_generation_protocol"] != protocol_ref
                    or row["mode"] != "prescribed_pose_sequence_NOT_free_dynamics"
                    or row["teacher_labels_generated"] is not False or row["entire_state_distribution_unseen"] is not False
                    or row["history_steps"] != [0, 3, 6, 9] or row["history_times_s"] != [0., .06, .12, .18]
                    or row["history_ages_s"] != [.18, .12, .06, 0.] or row["rendered_pose_count"] != 10):
                raise ValueError("prescribed input identity/role/chronology differs: " + row["id"])
            metadata = _pilot_json(_pilot_local(source, row["metadata_path"]), ledger, row["metadata_sha256"])
            if metadata != {k: v for k, v in row.items() if k not in ("metadata_path", "metadata_sha256")}:
                raise ValueError("per-input metadata differs from the complete bank")
            raw = _pilot_npy(_pilot_local(source, row["raw_path"]), ledger, row["raw_sha256"], "binary_RAW96_input_NOT_CCD")
            dmd = _pilot_npy(_pilot_local(source, row["dmd_path"]), ledger, row["dmd_sha256"], "exact_DMD_mosaic")
            if raw.shape != (4, 1, 96, 96) or raw.dtype != np.uint8 or not np.isin(raw, [0, 255]).all():
                raise ValueError("four chronological native binary RAW96 images required")
            packed = np.pad(np.block([[raw[0, 0], raw[1, 0]], [raw[2, 0], raw[3, 0]]]).repeat(3, 0).repeat(3, 1), 12)
            key = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
            if dmd.dtype != np.uint8 or not np.array_equal(dmd, packed) or key != row["input_key"]:
                raise ValueError("DMD packing or whole-history pixel identity differs")
            expected = np.asarray(spec["query_xy_m"], float)[None] - (.18-.02*np.arange(10))[:, None]*np.asarray(spec["velocity_xy_m_s"], float)[None]
            requested, actual = np.asarray(row["requested_positions_m"], float), np.asarray(row["rendered_positions_m"], float)
            if (requested.shape != (10, 2) or actual.shape != (10, 3) or not np.isfinite(actual).all()
                    or not np.isfinite(requested).all() or not np.allclose(requested, expected, atol=1e-8, rtol=0)
                    or not np.allclose(actual[:, :2], expected, atol=1e-8, rtol=0)):
                raise ValueError("ten prescribed/requested/rendered poses disagree")
            counts = np.count_nonzero(raw, axis=(1, 2, 3))
            border = any(np.any(x) for x in (raw[..., 0, :], raw[..., -1, :], raw[..., :, 0], raw[..., :, -1]))
            reasons = (["empty_source_frame"] if np.any(counts == 0) else []) + (["source_ball_touches_frame_edge"] if border else [])
            if key in seen_keys:
                reasons.append("pixel_identical_to_declared_seen_archive")
            arrays.append(raw)
            audited.append(dict(id=row["id"], role=row["role"], trajectory_group=row["trajectory_group"],
                query_xy_m=row["query_xy_m"], prescribed_velocity_xy_m_s=row["prescribed_velocity_xy_m_s"],
                actual_positions_m=actual.tolist(), history_actual_positions_m=actual[[0, 3, 6, 9]].tolist(),
                input_key=key, raw_sha256=row["raw_sha256"], dmd_sha256=row["dmd_sha256"],
                on_pixels_per_frame=counts.tolist(), DMD_on_pixels=int(np.count_nonzero(dmd)), border_touched=bool(border),
                matches_declared_seen_key=key in seen_keys, recomputed_ineligible_reasons=reasons,
                eligible=row["eligible"], declared_ineligible_reasons=row["ineligible_reasons"]))
        keys = [row["input_key"] for row in audited]
        for checked, original in zip(audited, rows):
            if keys.count(checked["input_key"]) > 1:
                checked["recomputed_ineligible_reasons"].append("pixel_identical_across_new_trajectories")
            if (checked["recomputed_ineligible_reasons"] != original["ineligible_reasons"]
                    or original["eligible"] is not (not checked["recomputed_ineligible_reasons"])):
                raise ValueError("input eligibility is not consistent with source pixels")
        reversed_pairs = []
        for i in range(4):
            for j in range(i+1, 4):
                a, b = audited[i], audited[j]
                if (a["query_xy_m"] == b["query_xy_m"] and np.array_equal(
                        np.asarray(a["prescribed_velocity_xy_m_s"]), -np.asarray(b["prescribed_velocity_xy_m_s"]))):
                    reversed_pairs.append(dict(ids=[a["id"], b["id"]], same_query_xy=True,
                        same_current_frame=bool(np.array_equal(arrays[i][-1], arrays[j][-1])),
                        same_whole_history=bool(np.array_equal(arrays[i], arrays[j])),
                        differing_pixels_per_frame=np.count_nonzero(arrays[i] != arrays[j], axis=(1, 2, 3)).tolist(),
                        independent_state_domains=False))
        (out / "figures").mkdir()
        plt = plotting()
        fig, axes = plt.subplots(4, 4, figsize=(11, 11), squeeze=False)
        for i, (row, raw) in enumerate(zip(audited, arrays)):
            for j, age in enumerate((180, 120, 60, 0)):
                ax = axes[i, j]
                ax.imshow(raw[j, 0], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                xy = row["history_actual_positions_m"][j][:2]
                ax.set_title(f"age {age} ms | xy=({xy[0]:.4f},{xy[1]:.4f}) m", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(row["id"]+" / "+row["role"]+"\neligible="+str(row["eligible"]), fontsize=8)
        fig.suptitle("CPU prescribed RAW96 histories | oldest to current | NOT free dynamics / NOT real CCD", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .97)); save_figure(fig, out / "figures/01_all4_RAW96_histories", 180); plt.close(fig)
        report = dict(schema=schema, audit_completed=True, bank_manifest_sha256=expected_manifest_sha256,
            bank_path=str(source), bank_file_count=len(manifest["file_sha256"]), whole_bank_file_SHA_verified=True,
            source_snapshots_verified=True, original_external_sources_reopened=False, re_rendered=False,
            independent_pose_pixel_association_reverified=False, position_consistency_atol_m=1e-8,
            rows=audited, row_count=4, unique_whole_history_keys=len(set(keys)),
            declared_seen_key_count=len(seen_keys), seen_key_scope=protocol["seen_check_scope"],
            complete_against_declared_archives=seen.get("complete_against_declared_archives"),
            all_rows_eligible=all(row["eligible"] for row in audited), reversed_history_pairs=reversed_pairs,
            raw_display_limits=[0, 255], dpi=180, history_ages_s=[.18, .12, .06, 0.],
            history_times_are_prescribed_virtual_samples_NOT_free_dynamics=True, hardware_accessed=False,
            GPU_used=False, models_or_teachers_imported=False, hardware_release_approved=False,
            control_release_approved=False, production_qualified=False,
            optical_modulation_qualified="UNKNOWN", real_control_qualified="UNKNOWN",
            entire_state_distribution_unseen=False, original_bank_not_modified=True)
        lines = ["# CPU输入历史：离线审计与可视化", "",
            "固定0–255灰度、180 dpi；4行×4张历史，左旧右新。不是CCD、不是自由动力学rollout，也不是全部状态未见的证明。", "",
            "仅核归档字节、拼接和姿态元数据一致性；没有重渲染或独立验证像素对应姿态。原外部源码/旧RAW未重新打开，冻结source/XML/seen快照已逐SHA检查。", "",
            f"银行manifest SHA：`{expected_manifest_sha256}`；唯一完整历史{len(set(keys))}/4；声明seen keys {len(seen_keys)}。", "",
            "| 输入 | Role | 有效 | 各帧亮像素 | DMD亮像素 | 触边 |", "|---|---|---|---|---:|---|"]
        lines += [f"| {row['id']} | {row['role']} | {row['eligible']} | {row['on_pixels_per_frame']} | {row['DMD_on_pixels']} | {row['border_touched']} |" for row in audited]
        for pair in reversed_pairs:
            lines += ["", f"反向配对{pair['ids']}：当前帧相同={pair['same_current_frame']}，完整历史相同={pair['same_whole_history']}，各时刻差异像素={pair['differing_pixels_per_frame']}。同几何配对，不是独立状态域。"]
        lines += ["", "![全部四组历史](figures/01_all4_RAW96_histories.png)"]
        with (out / "summary.md").open("x", encoding="utf8") as stream:
            stream.write("\n".join(lines)+"\n")
        for ref in list(ledger.values()):
            _pilot_read(ref["path"], ledger, ref["sha256"], ref["role"])
        if inventory != {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file() and p != source / "MANIFEST.json"}:
            raise ValueError("input bank inventory changed during audit")
        write_json_new(out / "source_sha256.json", list(ledger.values()))
        write_json_new(out / "RESULT.json", report)
        write_json_new(out / "MANIFEST.json", dict(schema=schema, audit_completed=True,
            file_sha256={str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file()}, manifest_excludes_itself=True))
        return report
    except Exception as exc:
        write_json_new(out / "FAILED.json", dict(schema=schema, audit_completed=False, error=repr(exc),
            partial_files_preserved=True, original_bank_not_modified=True, hardware_release_approved=False))
        raise


def _encoding_manifest(ref, ledger, *, full=True):
    root = Path(ref["path"]).resolve(strict=True)
    manifest = _pilot_json(root / "MANIFEST.json", ledger, ref["manifest_sha256"], "encoding_study_bound_manifest")
    if full:
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "MANIFEST.json"}
        if actual != set(manifest["file_sha256"]):
            raise ValueError("encoding study artifact inventory differs: " + str(root))
        for name, digest in manifest["file_sha256"].items():
            _pilot_read(_pilot_local(root, name), ledger, digest, "encoding_study_artifact")
    return root, manifest


def _encoding_measure_case(slot, bank_row, bank_root, bank_manifest, protocol, ledger, seen_frames):
    """Independent raw8 recomputation, inheriting accepted-frame gates only."""
    import json
    import numpy as np
    from .metrics import validate_regions, ratio
    case_root, case_manifest = _encoding_manifest(slot["case"], ledger)
    prediction_root, prediction_manifest = _encoding_manifest(slot["predictions"], ledger)
    if (case_manifest["design"]["case_id"] != slot["case_id"]
            or case_manifest["design"]["path"] != str(bank_root / "STUDY_DESIGN.json")
            or case_manifest["design"]["sha256"] != bank_manifest["file_sha256"]["STUDY_DESIGN.json"]
            or case_manifest["fixed_input"]["raw_sha256"] != bank_row["raw_sha256"]
            or case_manifest["fixed_input"]["dmd_sha256"] != bank_row["dmd_sha256"]
            or case_manifest["probe_spec"] != protocol["probe_spec"]):
        raise ValueError("encoding case/input/probe identity differs from the prospective bank")
    if slot["capture"] is None:
        if slot["report"] is not None:
            raise ValueError("a report cannot be paired without its capture seal")
        return dict(status="NOT_EXECUTED", prepared_assets_present=True, measurement=None)
    cap_root, cap_manifest = _encoding_manifest(slot["capture"], ledger, full=False)
    plan = _pilot_json(cap_root / "PLAN.json", ledger, cap_manifest["file_sha256"]["PLAN.json"])
    cap_result = _pilot_json(cap_root / "RESULT.json", ledger, cap_manifest["file_sha256"]["RESULT.json"])
    if (cap_result.get("schema") != PILOT_CAPTURE_SCHEMA
            or plan.get("case_asset_manifest") != {"path": str(case_root), "sha256": slot["case"]["manifest_sha256"]}
            or plan["frozen_predictions"] != {"path": str(prediction_root), "manifest_sha256": slot["predictions"]["manifest_sha256"],
                                             "result_sha256": prediction_manifest["file_sha256"]["RESULT.json"]}
            or plan["case_id"] != slot["case_id"] or _pilot_repeat_count(plan) != protocol["frames_per_state"]):
        raise ValueError("encoding capture PLAN is paired with different assets/predictions/repeats")
    if (not isinstance(plan.get("manual_geometry_expected"), dict)
            or set(plan["manual_geometry_expected"]) != {"x", "y", "full_panel", "square", "focal_mm", "period"}):
        raise ValueError("complete six-field manual geometry identity required")
    if slot["report"] is None:
        return dict(status="NOT_ANALYZED", measurement=None, capture_complete=cap_result.get("complete") is True)
    rep_root, _ = _encoding_manifest(slot["report"], ledger)
    report = _pilot_json(rep_root / "RESULT.json", ledger)
    if report.get("schema") != PILOT_ANALYSIS_SCHEMA or report.get("analysis_complete") is not True:
        raise ValueError("a finished single-case analysis, not a repaired failed analysis, is required")
    inherited = _pilot_json(rep_root / "provenance/source_sha256.json", ledger)
    inherited = {r["path"]: r["sha256"] for r in inherited}
    for path in (cap_root / "MANIFEST.json", cap_root / "PLAN.json", cap_root / "RESULT.json"):
        if inherited.get(str(path)) != ledger[str(path)]["sha256"]:
            raise ValueError("single-case report source identity differs from indexed capture")
    if (report["case_profile"]["case_asset_manifest"] != plan["case_asset_manifest"]
            or report["case_profile"]["case_id"] != slot["case_id"]
            or report["predictions"]["manifest_sha256"] != slot["predictions"]["manifest_sha256"]
            or report["planned_states"] != 17 or report["planned_frames"] != plan["total_planned_frames"]
            or report["planned_primary_contrasts"] != 6 or report["hardware_release_approved"] is not False
            or any(v != "UNKNOWN" for v in report["gate_states"].values())):
        raise ValueError("single-case report identity, denominator or release boundary differs")
    repeats = _pilot_repeat_count(plan)
    planned = plan["states"]
    if ([(s["id"], s["kind"], s["time_block"], s["planned_repeats"]) for s in report["states"]]
            != [(s["id"], s["kind"], s["time_block"], repeats) for s in planned]
            or len(report["contrasts"]) != 6):
        raise ValueError("single-case report omitted or reordered prescribed states/contrasts")
    role_keys=("id", "time_block", "kind", "probe_state", "Z_before", "Z_after")
    expected_roles=[]
    for block in range(3):
        sequence=[s for s in planned if s["time_block"]==block]
        for kind in ("A", "B"):
            j=next(i for i,s in enumerate(sequence) if s["kind"]==kind)
            probe=sequence[j]
            expected_roles.append(dict(id=probe["id"]+"_minus_bracketed_Z",time_block=block,
                kind=probe["kind"],probe_state=probe["id"],Z_before=sequence[j-1]["id"],Z_after=sequence[j+1]["id"]))
    actual_roles=[{k:c.get(k) for k in role_keys} for c in report["contrasts"]]
    canonical=lambda value:json.dumps(value,sort_keys=True,allow_nan=False)
    if canonical(actual_roles)!=canonical(expected_roles):
        raise ValueError("exact six unique contrast roles/order and local Z associations required")
    region_path, region_sha = _pilot_reference(plan, cap_root, "regions")
    if region_sha != BROAD8_SHA256 or plan["camera_expected"]["roi"] != BROAD8_ROI:
        raise ValueError("encoding comparison requires the same original fixed BROAD8/native ROI")
    shape = (BROAD8_ROI["Height"], BROAD8_ROI["Width"])
    regions = validate_regions(_pilot_npy(region_path, ledger, region_sha), shape)
    labels = regions.argmax(0).ravel()
    predictions, _ = _pilot_predictions(plan, cap_root, prediction_root, regions, ledger)
    journal = _pilot_read(cap_root / "attempts.jsonl", ledger, cap_manifest["file_sha256"]["attempts.jsonl"]) if "attempts.jsonl" in cap_manifest["file_sha256"] else b""
    attempts = [json.loads(line) for line in journal.splitlines() if line.strip()]
    if len(attempts) != report["attempted_frame_count"] or len({a["id"] for a in attempts}) != len(attempts):
        raise ValueError("single-case attempt denominator differs")
    state_rows, mean_images, mean8, times, noise, absolute_host = [], {}, {}, {}, {}, []
    for state in report["states"]:
        rejected = {r["id"] for r in state["rejected"]}
        selected = [a for a in attempts if a["state_id"] == state["id"] and a.get("status") == "valid"
                    and a.get("raw") and a.get("metadata") and a["id"] not in rejected]
        if len(selected) != state["valid_frames"]:
            raise ValueError("accepted-frame subset differs from the frozen single-case report")
        features, host = [], []
        summed = np.zeros(shape, np.float64)
        for attempt in selected:
            loaded = {}
            for key in ("raw", "metadata"):
                ref = attempt[key]; path = _pilot_local(cap_root, ref["path"])
                name = str(path.relative_to(cap_root))
                if cap_manifest["file_sha256"].get(name) != ref["sha256"] or inherited.get(str(path)) != ref["sha256"]:
                    raise ValueError("accepted raw/metadata is not bound by both capture and report")
                loaded[key] = (_pilot_npy(path, ledger, ref["sha256"], "accepted_real_CCD_uint8") if key == "raw"
                               else _pilot_json(path, ledger, ref["sha256"]))
            raw, meta = loaded["raw"], loaded["metadata"]
            frame = meta["frame"]
            host_ns = frame.get("host_get_started_unix_ns")
            if (raw.shape != shape or raw.dtype != np.uint8 or int(raw.max()) >= 240
                    or frame != attempt["frame"] or meta["camera_active"] != plan["camera_expected"]["active"]
                    or frame.get("IsIncomplete") is True or type(host_ns) is not int
                    or host_ns != attempt["host_get_started_unix_ns"]
                    or (absolute_host and host_ns <= absolute_host[-1])):
                raise ValueError("accepted native frame/camera association differs")
            uid = (plan["camera_expected"]["active"].get("DeviceSerialNumber"), frame["FrameID"], frame["TimeStamp"])
            if uid in seen_frames:
                raise ValueError("same camera frame reused within/across encoding cases")
            seen_frames.add(uid)
            absolute_host.append(host_ns)
            summed += raw
            features.append(np.bincount(labels, weights=raw.ravel(), minlength=8))
            host.append(float(frame["host_get_started_unix_ns"] - report["time_origin_unix_ns"]))
        count = len(features)
        measured = np.asarray(features, np.float64)
        item = dict(id=state["id"], kind=state["kind"], valid_frames=count, numeric_eligible=state["numeric_eligible"],
                    raw_total_ADU=None, native8_mean_ADU=None, single_frame_covariance_ADU2=None,
                    single_frame_repeat_scale_ADU=None, net_total_after_optical_BG_ADU=None,
                    covariance_rank_upper_bound=min(8, max(0, count-1)), covariance_inverse_used=False)
        if count:
            mean_images[state["id"]], mean8[state["id"]], times[state["id"]] = summed/count, measured.mean(0), float(np.mean(host))
            if state["stats"] is not None and not np.allclose(measured.mean(0), state["stats"]["native8_mean_ADU"], rtol=1e-12, atol=1e-6):
                raise ValueError("independent raw8 mean disagrees with the frozen report")
        if state["numeric_eligible"]:
            covariance = np.cov(measured, rowvar=False, ddof=1)
            if count != state["planned_repeats"] or not np.allclose(covariance, state["stats"]["native8_covariance_ADU2"], rtol=1e-12, atol=1e-6):
                raise ValueError("independent single-frame raw8 covariance disagrees")
            scale = float(np.sqrt(np.trace(covariance))); noise[state["id"]] = scale
            item.update(raw_total_ADU=float(measured.mean(0).sum()), native8_mean_ADU=measured.mean(0).tolist(),
                        single_frame_covariance_ADU2=covariance.tolist(), single_frame_repeat_scale_ADU=scale)
        state_rows.append(item)
    if sum(s["valid_frames"] for s in state_rows) != report["valid_frames"]:
        raise ValueError("independently loaded valid-frame total differs from the report")
    by_id = {s["id"]: s for s in state_rows}
    bg = [s for s in state_rows if s["kind"] == "BG"]
    zero_minus_bg = []
    if len(bg) == 2 and all(s["numeric_eligible"] for s in bg):
        b0, b1 = (s["id"] for s in bg)
        for state in state_rows:
            sid = state["id"]
            if state["kind"] != "BG" and state["numeric_eligible"] and times[b0] < times[sid] < times[b1]:
                w = (times[sid]-times[b0])/(times[b1]-times[b0])
                state["net_total_after_optical_BG_ADU"] = float((mean8[sid]-(1-w)*mean8[b0]-w*mean8[b1]).sum())
                if state["kind"] == "Z":
                    zero_minus_bg.append(mean_images[sid]-(1-w)*mean_images[b0]-w*mean_images[b1])
    contrasts = []
    for original in report["contrasts"]:
        item = dict(id=original["id"], kind=original["kind"], status=original["status"],
            response8norm_ADU=None, modulation_total_minus_local_Z_ADU=None, probe_single_frame_repeat_scale_ADU=None,
            descriptive_effect_over_single_frame_repeat_scale=None, models=None)
        if original["metrics"] is not None:
            p, z0, z1 = (original[k] for k in ("probe_state", "Z_before", "Z_after"))
            if not all(by_id[s]["numeric_eligible"] for s in (p, z0, z1)) or not times[z0] < times[p] < times[z1]:
                raise ValueError("reported contrast lacks complete bracketed states")
            w = (times[p]-times[z0])/(times[z1]-times[z0])
            image = mean_images[p]-(1-w)*mean_images[z0]-w*mean_images[z1]
            actual = mean8[p]-(1-w)*mean8[z0]-w*mean8[z1]
            if not np.allclose(actual, original["metrics"]["native8_contrast_ADU_sum"], rtol=1e-12, atol=1e-6):
                raise ValueError("independent raw8 bracketed contrast disagrees")
            item.update(response8norm_ADU=float(np.linalg.norm(actual)), modulation_total_minus_local_Z_ADU=float(actual.sum()),
                probe_single_frame_repeat_scale_ADU=noise[p],
                descriptive_effect_over_single_frame_repeat_scale=ratio(np.linalg.norm(actual), noise[p]), models={})
            for model, fields in predictions.items():
                delta = fields[original["kind"]]-fields["Z"]
                predicted = np.bincount(labels, weights=delta.ravel(), minlength=8)
                values = dict(native8_model_contrast_relative_L2=ratio(np.linalg.norm(actual-predicted), np.linalg.norm(actual)),
                    native8_model_contrast_cosine=ratio(actual @ predicted, np.linalg.norm(actual)*np.linalg.norm(predicted)),
                    image_model_contrast_RMSE_ADU=float(np.sqrt(np.mean((image-delta)**2))))
                for key, value in values.items():
                    saved = original["model_comparisons"][model][key]
                    if (value is None) != (saved is None) or (value is not None and not np.isclose(value, saved, rtol=1e-10, atol=1e-6)):
                        raise ValueError("independent frozen-model contrast error disagrees")
                item["models"][model] = values
        contrasts.append(item)
    complete = report["planned_capture_and_analysis_complete"] is True and cap_result.get("complete") is True
    if complete and (report["valid_frames"] != 17*repeats or len(contrasts) != 6
                     or any(not s["numeric_eligible"] for s in state_rows)
                     or any(c["response8norm_ADU"] is None for c in contrasts)):
        raise ValueError("complete status cannot replace complete frame/state/contrast denominators")
    noise_summary = {}
    for kind in ("BG", "Z", "A", "B"):
        available = [s for s in state_rows if s["kind"] == kind and s["single_frame_repeat_scale_ADU"] is not None]
        if available:
            high = max(s["single_frame_repeat_scale_ADU"] for s in available)
            noise_summary[kind] = dict(count=len(available),
                minimum_ADU_sum=min(s["single_frame_repeat_scale_ADU"] for s in available), maximum_ADU_sum=high,
                maximum_state_ids=[s["id"] for s in available if s["single_frame_repeat_scale_ADU"] == high])
        else:
            noise_summary[kind] = dict(count=0, minimum_ADU_sum=None, maximum_ADU_sum=None, maximum_state_ids=[])
    return dict(status="COMPLETE" if complete else "INCOMPLETE", measurement=dict(
        valid_frames=report["valid_frames"], planned_frames=report["planned_frames"], states=state_rows, contrasts=contrasts,
        available_contrasts=sum(c["response8norm_ADU"] is not None for c in contrasts),
        camera_expected=plan["camera_expected"], manual_geometry_expected=plan.get("manual_geometry_expected"),
        static_repeats=_pilot_repeat_count(plan), capture_error=report.get("capture_error"),
        accepted_host_get_started_range_unix_ns=([absolute_host[0], absolute_host[-1]] if absolute_host else None),
        host_clock_is_NOT_verified_exposure_clock=True,
        single_frame_repeat_scale_range_by_kind=noise_summary,
        accepted_frame_subset_inherited_from_sealed_report=True, independent_raw8_recomputed=True,
        covariance_estimator="within-state ddof1 sample covariance; correlated frames, not globally unbiased noise"),
        _display_Z_minus_BG=(np.mean(zero_minus_bg, axis=0) if zero_minus_bg else None))


def assess_encoding_study(index_path, output, *, expected_index_sha256):
    """Four predeclared encoding slots, no fit/selection/hardware or dashboard."""
    import json
    import numpy as np
    from .reporting import plotting, save_figure
    order = [("E00_BASELINE_BEFORE", 0), ("E01_DILATE_R1", 1), ("E02_DILATE_R2", 2), ("E03_BASELINE_AFTER", 0)]
    schema = "optical-lab-encoding-study-analysis-1"
    out = Path(output)
    if out.exists() or out.is_symlink():
        raise FileExistsError("new encoding analysis output required")
    ledger = {}; output_created = False
    try:
        index = _pilot_json(index_path, ledger, expected_index_sha256, "result_identity_index_NOT_prospective_protocol")
        if (index.get("schema") != "optical-lab-encoding-study-analysis-index-1"
                or [(s["case_id"], s["radius_logical_px"]) for s in index["rows"]] != order
                or any(type(s["radius_logical_px"]) is not int for s in index["rows"])):
            raise ValueError("all four encoding slots in their predeclared order are required")
        protocol = _pilot_json(index["protocol"]["path"], ledger, index["protocol"]["sha256"], "prospective_encoding_protocol")
        if (protocol.get("schema") != "optical-lab-encoding-plan-1" or protocol.get("hardware_actions_enabled") is not False
                or protocol["case_order"] != [dict(id=name, radius_logical_px=r) for name, r in order]
                or any(type(s["radius_logical_px"]) is not int for s in protocol["case_order"])
                or type(protocol.get("frames_per_state")) is not int or protocol["frames_per_state"] != 8):
            raise ValueError("prospective encoding protocol/order required; index is not that protocol")
        bank_root, bank_manifest = _encoding_manifest(index["bank"], ledger)
        if out.resolve().is_relative_to(bank_root):
            raise ValueError("encoding analysis cannot modify its source bank")
        bank = _pilot_json(bank_root / "RESULT.json", ledger)
        if (bank_manifest.get("schema") != "optical-lab-encoding-inputs-1" or bank_manifest.get("completed") is not True
                or bank.get("schema") != "optical-lab-encoding-inputs-1" or bank.get("completed") is not True
                or bank.get("hardware_accessed") is not False or bank["protocol"] != index["protocol"]
                or [(r["case_id"], r["radius_logical_px"]) for r in bank["rows"]] != order
                or any(type(r["radius_logical_px"]) is not int for r in bank["rows"])
                or bank["source_input"] != protocol["source_input"]):
            raise ValueError("encoding bank is not paired with the prospective protocol")
        design = _pilot_json(bank_root / "STUDY_DESIGN.json", ledger)
        if (design.get("schema") != "optical-lab-static-study-design-1" or design.get("hardware_actions_enabled") is not False
                or design["encoding_plan"] != index["protocol"] or design["collection_order"] != [name for name, _ in order]
                or [(r["id"], r["input_id"], r["probe_spec"]) for r in design["cases"]] != [(name, name, protocol["probe_spec"]) for name, _ in order]
                or any(design["inputs"][r["case_id"]][key] != r[key] for r in bank["rows"] for key in ("raw_sha256", "dmd_sha256"))):
            raise ValueError("encoding bank study design differs from its fixed protocol/inputs")
        raw0 = _pilot_npy(bank_root / "SOURCE_RAW96.npy", ledger, protocol["source_input"]["raw_sha256"])
        dmd0 = _pilot_npy(bank_root / "SOURCE_DMD.npy", ledger, protocol["source_input"]["dmd_sha256"])
        pack = lambda raw: np.pad(np.block([[raw[0, 0], raw[1, 0]], [raw[2, 0], raw[3, 0]]]).repeat(3, 0).repeat(3, 1), 12)
        if (raw0.shape != (4, 1, 96, 96) or raw0.dtype != np.uint8 or not np.isin(raw0, [0, 255]).all()
                or dmd0.shape != (600, 600) or dmd0.dtype != np.uint8 or not np.array_equal(pack(raw0), dmd0)):
            raise ValueError("exact original four binary RAW96/DMD source required")
        for slot in index["rows"]:
            for key in ("case", "predictions", "capture", "report"):
                ref = slot[key]
                if ref is not None and out.resolve().is_relative_to(Path(ref["path"]).resolve()):
                    raise ValueError("encoding output cannot modify any input artifact tree")
        out.mkdir(exist_ok=False); output_created = True
        (out / "figures").mkdir(); (out / "provenance").mkdir()
        source_snapshot = {}
        for path in (Path(__file__), Path(__file__).with_name("metrics.py"), Path(__file__).with_name("reporting.py"),
                     Path(__file__).with_name("storage.py"), Path(__file__).with_name("protocol.py")):
            blob = _pilot_read(path, ledger, role="encoding_analysis_source")
            target = out / "provenance" / path.name
            with target.open("xb") as stream: stream.write(blob)
            source_snapshot[str(path.resolve())] = dict(path=str(target.relative_to(out)), sha256=hashlib.sha256(blob).hexdigest())
        write_json_new(out / "provenance/CODE_SNAPSHOT.json", source_snapshot)
        rows, seen_frames, display_images, display_dmd = [], set(), [], []
        for slot, encoded in zip(index["rows"], bank["rows"]):
            radius = slot["radius_logical_px"]
            expected = np.zeros_like(raw0)
            padded = np.pad(raw0 > 0, ((0, 0), (0, 0), (radius, radius), (radius, radius)))
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if dy*dy+dx*dx <= radius*radius:
                        expected |= padded[:, :, radius+dy:radius+dy+96, radius+dx:radius+dx+96].astype(np.uint8)*255
            raw = _pilot_npy(_pilot_local(bank_root, encoded["raw_path"]), ledger, encoded["raw_sha256"])
            dmd = _pilot_npy(_pilot_local(bank_root, encoded["dmd_path"]), ledger, encoded["dmd_sha256"])
            if (not np.array_equal(raw, expected) or raw.dtype != np.uint8 or dmd.dtype != np.uint8
                    or not np.array_equal(dmd, pack(raw)) or encoded["raw_on"] != int(np.count_nonzero(raw))
                    or encoded["dmd_on"] != int(np.count_nonzero(dmd))
                    or encoded["dmd_canvas_fraction"] != np.count_nonzero(dmd)/dmd.size
                    or (radius == 0 and (encoded["raw_sha256"] != protocol["source_input"]["raw_sha256"]
                                         or encoded["dmd_sha256"] != protocol["source_input"]["dmd_sha256"]))):
                raise ValueError("encoded radius/kernel/packing/area or baseline exact bytes differ")
            row = dict(case_id=slot["case_id"], radius_logical_px=radius, raw_on=encoded["raw_on"], dmd_on=encoded["dmd_on"],
                dmd_canvas_fraction=encoded["dmd_canvas_fraction"], per_frame_geometry=encoded["per_frame_geometry"],
                planned_frames=17*protocol["frames_per_state"], planned_primary_contrasts=6,
                source_underlying_state_is_same_seen_T01=True)
            refs = [slot[k] for k in ("case", "predictions", "capture", "report")]
            if all(ref is None for ref in refs):
                row.update(status="NOT_EXECUTED", measurement=None)
            elif slot["case"] is None or slot["predictions"] is None:
                raise ValueError("declared execution requires explicit paired case and prediction identities")
            else:
                for ref in refs:
                    if ref is not None and out.resolve().is_relative_to(Path(ref["path"]).resolve()):
                        raise ValueError("encoding output cannot modify any input artifact tree")
                row.update(_encoding_measure_case(slot, encoded, bank_root, bank_manifest, protocol, ledger, seen_frames))
            display_images.append(row.pop("_display_Z_minus_BG", None)); display_dmd.append(dmd)
            rows.append(row)
        measurements = [r["measurement"] for r in rows if r["measurement"] is not None]
        contracts_equal = len(measurements) < 2 or all((m["camera_expected"], m["manual_geometry_expected"]) == (measurements[0]["camera_expected"], measurements[0]["manual_geometry_expected"]) for m in measurements[1:])
        if not contracts_equal:
            raise ValueError("camera/photometry/manual geometry differs across measured encoding cases")
        last_end = None
        measured_order = []
        for row in rows:
            m = row["measurement"]
            times = None if m is None else m["accepted_host_get_started_range_unix_ns"]
            if times is not None:
                if last_end is not None and times[0] <= last_end:
                    raise ValueError("actual accepted host times violate the prospective case order")
                last_end = times[1]; measured_order.append(row["case_id"])
        plt = plotting()
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        limits = protocol["display"]["ccd_adu_limits"]
        for i, row in enumerate(rows):
            axes[0, i].imshow(display_dmd[i], cmap="gray", vmin=0, vmax=255)
            axes[0, i].set_title(f"r={row['radius_logical_px']}, DMD ON={row['dmd_on']}", fontsize=9)
            if display_images[i] is None:
                axes[1, i].text(.5, .5, row["status"]+" / Z-BG NA", ha="center", transform=axes[1, i].transAxes, fontsize=8)
            else:
                axes[1, i].imshow(display_images[i], cmap="gray", vmin=limits[0], vmax=limits[1])
            axes[1, i].set_title(row["case_id"], fontsize=8)
            for j in range(2): axes[j, i].set_axis_off()
        fig.suptitle(f"Same seen T01: exact DMD (0..255) / offline mean Z minus same-case optical BG ({limits[0]}..{limits[1]} ADU)\nNOT single-frame or physical dark; common scales, no per-image normalization", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .90)); save_figure(fig, out / "figures/01_inputs_and_Z_minus_optical_BG", 180); plt.close(fig)
        colors = {"Z": "#222222", "A": "#0072B2", "B": "#D55E00", "BG": "#009E73"}
        fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
        for i, row in enumerate(rows):
            measured = row["measurement"]
            if measured is None:
                for axis in axes: axis.text(i, .5, row["status"], ha="center", fontsize=7, transform=axis.get_xaxis_transform())
                continue
            for kind, color in colors.items():
                values = [s for s in measured["states"] if s["kind"] == kind]
                axes[0].scatter([i]*len(values), [s["net_total_after_optical_BG_ADU"] if s["net_total_after_optical_BG_ADU"] is not None else np.nan for s in values], color=color, s=18, label=kind if i == 0 else None)
                axes[1].scatter([i]*len(values), [s["single_frame_repeat_scale_ADU"] if s["single_frame_repeat_scale_ADU"] is not None else np.nan for s in values], color=color, s=18)
            for c in row["measurement"]["contrasts"]:
                if c["response8norm_ADU"] is None: continue
                axes[2].scatter(i, c["response8norm_ADU"], color=colors[c["kind"]], marker="o" if c["kind"] == "A" else "s")
                value = c["descriptive_effect_over_single_frame_repeat_scale"]
                axes[3].scatter(i, value if value is not None else np.nan, color=colors[c["kind"]])
                for shift, (model, values) in zip((-.08, .08), c["models"].items()):
                    value = values["native8_model_contrast_relative_L2"]
                    axes[4].scatter(i+shift, value if value is not None else np.nan, marker="x" if model == "M0_REFERENCE" else "+", color=colors[c["kind"]])
        axes[0].set_ylabel("ROI sum minus optical BG / ADU sum")
        axes[1].set_ylabel("Every state: sqrt(trace(raw8 cov))\nADU sum; bursts NOT filtered")
        axes[2].set_ylabel("Probe - local Z: raw8 norm / ADU sum")
        axes[3].set_ylabel("Effect / single-frame repeat scale\nDESCRIPTIVE, NOT independent SNR")
        axes[4].set_ylabel("Both frozen models: raw8 relative L2\nx=M0, +=M1; blue=A, orange=B")
        axes[4].set_xticks(range(4), [r["case_id"] for r in rows], fontsize=8)
        axes[0].legend(frameon=False)
        fig.suptitle("All six mean-level contrasts per case | shared Z / time / ON-area confounded | no model selection")
        fig.tight_layout(rect=(0, 0, 1, .96)); save_figure(fig, out / "figures/02_all_contrasts_and_fixed_model_errors", 180); plt.close(fig)
        result = dict(schema=schema, analysis_complete=True, main_comparison_complete=all(r["status"] == "COMPLETE" for r in rows),
            index_sha256=expected_index_sha256, index_is_prospective_protocol=False, prospective_protocol=index["protocol"],
            bank=index["bank"], planned_case_count=4, rows=rows, contracts_equal_across_measured_cases=contracts_equal,
            planned_frames=4*17*protocol["frames_per_state"], planned_primary_contrasts=24,
            analyzed_valid_frames=sum(m["valid_frames"] for m in measurements),
            analyzed_available_contrasts=sum(m["available_contrasts"] for m in measurements),
            accepted_host_times_follow_prospective_case_order=True, measured_case_order=measured_order,
            host_time_order_is_NOT_physical_exposure_association=True,
            all_case_slots_preserved=True, covariance_definition="within-state ddof1 raw8 sample covariance, no sqrtN reduction",
            all_state_repeat_scales_preserved=True, repeat_scale_ranges_are_NOT_confidence_intervals=True,
            effect_ratio_name="单帧尺度描述性效应/重复变化比", effect_ratio_omits_shared_Z_mean_uncertainty=True,
            optical_background_is_NOT_physical_sensor_dark=True, ADU_sums_are_NOT_absolute_efficiency=True,
            same_underlying_seen_state_NOT_input_holdout=True, time_ON_area_photons_spectrum_centroid_are_confounded=True,
            baseline_after_is_endpoint_drift_diagnostic_NOT_full_nonlinear_drift_correction=True,
            source_snapshot=source_snapshot,
            models_refitted=False, models_selected=False, raw_or_images_normalized=False, hardware_accessed=False,
            control_release_approved=False, production_qualified=False,
            gate_states={"G0_acquisition": "UNKNOWN", "G1_modulation": "UNKNOWN", "G2_model": "UNKNOWN", "absolute_efficiency": "UNKNOWN"})
        lines = ["# E004 球编码宽度：归档对照分析", "", "INDEX仅登记结果身份，事前方案单独按SHA绑定。四槽全部保留，失败／未执行为NA；不拟合、不挑模型、不放行。", "",
            "同一已见T01及固定探针，半径按RAW96整数圆盘；不是变焦、改变物理球或新轨迹。ON面积、光子、频谱、量化质心与时段混杂，不能作SNR因果结论。", "",
            "总信号只减本case前后DMD0光学背景的时间插值（不是物理dark），调制单独减相邻Z；均值只用于离线分析。", "",
            "单帧尺度描述性效应/重复变化比＝响应8范数／sqrt(trace(该probe的单帧8区ddof1协方差))，不除sqrtN、不含共享Z均值不确定性，不是独立SNR或决策率。", "",
            "| Case | radius | RAW ON | DMD ON | 状态 | 有效/计划帧 | 可算对照 |", "|---|---:|---:|---:|---|---:|---:|"]
        for row in rows:
            m = row["measurement"]
            lines.append(f"| {row['case_id']} | {row['radius_logical_px']} | {row['raw_on']} | {row['dmd_on']} | {row['status']} | {str(m['valid_frames'])+'/'+str(m['planned_frames']) if m else 'NA/'+str(row['planned_frames'])} | {str(m['available_contrasts'])+'/6' if m else 'NA/6'} |")
        lines += ["", "所有state的单帧重复尺度逐项保留；下表只补充各类范围与最大项位置，不用均值／中位数掩盖突发。范围不是置信区间；8次样本的8维协方差秩至多7，不求逆。", "",
            "| Case | Kind | 可算state数 | 单帧重复尺度范围，ADU sum | 最大state |", "|---|---|---:|---:|---|"]
        for row in rows:
            m=row["measurement"]
            for kind in ("BG", "Z", "A", "B"):
                s=None if m is None else m["single_frame_repeat_scale_range_by_kind"][kind]
                span="NA" if not s or not s["count"] else f"{s['minimum_ADU_sum']:.6g}–{s['maximum_ADU_sum']:.6g}"
                lines.append(f"| {row['case_id']} | {kind} | {s['count'] if s else 0} | {span} | {','.join(s['maximum_state_ids']) if s else 'NA'} |")
        lines += ["", "首图下排为所有完整Z状态分别减本case前后光学背景插值后的离线均值；不是单帧或用于在线累计。固定事前ADU色标，不作面积或每图归一。", "", "![输入与固定ADU净CCD均值](figures/01_inputs_and_Z_minus_optical_BG.png)", "![全部调制与模型误差](figures/02_all_contrasts_and_fixed_model_errors.png)"]
        with (out / "总结.md").open("x", encoding="utf8") as stream: stream.write("\n".join(lines)+"\n")
        for ref in list(ledger.values()): _pilot_read(ref["path"], ledger, ref["sha256"], ref["role"])
        for ref in source_snapshot.values():
            if sha256(out / ref["path"]) != ref["sha256"]:
                raise ValueError("encoding analysis source snapshot changed")
        write_json_new(out / "source_sha256.json", list(ledger.values()))
        write_json_new(out / "RESULT.json", result)
        write_json_new(out / "MANIFEST.json", dict(schema=schema, analysis_complete=True, file_sha256={str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file()}, manifest_excludes_itself=True))
        return result
    except Exception as exc:
        if output_created: write_json_new(out / "FAILED.json", dict(schema=schema, analysis_complete=False, error=repr(exc), input_artifacts_not_modified=True))
        raise
