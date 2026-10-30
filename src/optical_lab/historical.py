"""Read-only, hash-bound OLD real CCD/model pairs. No legacy imports or devices.

``load_historical_sources(config)`` builds JSON-safe neutral records without
opening CCD pixels. ``load_historical_cell(cell)`` verifies captured bytes and
returns native uint8 repeats, ADU model arrays, old regions/head and darks.
Names ending ``mean.npy`` are NEVER evidence that an array is a real capture.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np


SCHEMA = "optical_lab_historical_readonly_v1"


def _digest(blob):
    return hashlib.sha256(blob).hexdigest()


def _path(root, value):
    root = Path(root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("historical reference escapes explicitly declared legacy root")
    return path


def _ref(root, path, sha, role):
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("explicit SHA256 required for " + role)
    return {"path": str(_path(root, path).relative_to(Path(root).resolve())),
            "sha256": sha, "role": role}


def _capture(root, ref, sources):
    path = _path(root, ref["path"])
    blob = path.read_bytes()
    if _digest(blob) != ref["sha256"]:
        raise ValueError("historical SHA mismatch: " + str(path))
    item = {"path": str(path), "sha256": ref["sha256"], "role": ref["role"]}
    if item not in sources:
        sources.append(item)
    return blob


def _json(root, ref, sources):
    return json.loads(_capture(root, ref, sources))


def _array(root, ref, sources):
    return np.load(io.BytesIO(_capture(root, ref, sources)), allow_pickle=False)


def _tail(sources):
    for ref in sources:
        if _digest(Path(ref["path"]).read_bytes()) != ref["sha256"]:
            raise ValueError("historical source changed during load: " + ref["path"])


def _stage(root, row, count):
    snapshots = []
    for snap in row.get("snapshots", []):
        folder = _path(root, snap["folder"])
        snapshots.append({
            "raw": _ref(root, folder / "CCD_raw.npy", snap["raw_sha256"], "real_CCD_uint8"),
            "metadata": _ref(root, folder / "metadata.json", snap["metadata_sha256"], "snapshot_metadata"),
            "recorded_frame": deepcopy(snap.get("frame")),
        })
    return {"name": row["name"], "revision": row.get("revision"),
            "verified_unix_ns": row.get("verified_unix_ns"),
            "actual_framebuffer_checked": row.get("actual_framebuffer_checked") is True,
            "expected_repeats": int(count), "snapshots": snapshots}


def load_historical_sources(config_path):
    """Return {cells, probe_groups, audit}; metadata-only, no CCD/model decoding.

    Every declared cell is retained. Historical split names are provenance, not
    new holdouts. Files referenced by a plan are matched by exact cell/input/mask
    identities. Optional WIDE models require an existing SHA-pinned index.
    """
    cp = Path(config_path).resolve()
    blob = cp.read_bytes()
    config = json.loads(blob)
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported historical configuration schema")
    root = (cp.parent / config["legacy_root"]).resolve()
    sources = [{"path": str(cp), "sha256": _digest(blob), "role": "historical_config"},
               {"path": str(Path(__file__).resolve()),
                "sha256": _digest(Path(__file__).read_bytes()), "role": "historical_loader_source"}]
    cells, probes, batch_audit = [], [], []
    for batch in config["batches"]:
        local = []
        result = _json(root, batch["capture_result"], local)
        protocol = _json(root, batch["capture_protocol"], local)
        plan_ref = _ref(root, protocol["plan_path"], protocol["plan_sha256"], "capture_plan")
        plan = _json(root, plan_ref, local)
        watcher = _json(root, batch["watchdog_result"], local)
        contract_ref = _ref(root, plan["hardware_contract"], plan["hardware_contract_sha256"], "camera_contract")
        spec_ref = _ref(root, plan["base_forward_spec"], plan["base_forward_spec_sha256"], "original_forward_spec")
        contract = _json(root, contract_ref, local)
        spec = _json(root, spec_ref, local)
        for path, sha in protocol.get("source_sha256", {}).items():
            _capture(root, _ref(root, path, sha, "original_capture_source"), local)
        header_ok = bool(result.get("completed") is True and result.get("error") is None
                         and result.get("actual_outputs_zero") is True and not result.get("cleanup_error")
                         and not watcher.get("error") and not watcher.get("intervened")
                         and watcher.get("actual_all_zero") is True)
        stages = {r["name"]: r for r in result["stages"]}
        if len(stages) != len(result["stages"]):
            raise ValueError("duplicate capture stage identity")
        expected_names = {c["id"] for c in plan["cells"]} | {"DARK_BEFORE", "DARK_AFTER"}
        stage_coverage = set(stages) == expected_names
        inputs = {r["id"]: r for r in plan["inputs"]}
        masks = {r["id"]: r for r in plan["masks"]}
        region_ref = _ref(root, spec["readout_path"], spec["readout_sha256"], "original_NATIVE8_regions_NOT_W7")
        head_ref = _ref(root, spec["head_path"], spec["head_sha256"], "original_RAW18_diagnostic_head")
        _capture(root, region_ref, local)
        _capture(root, head_ref, local)
        extra = batch.get("wide_models")
        wide_rows, pinned = {}, {}
        if extra:
            wide = _json(root, extra["result"], local)
            pin = _json(root, extra["pinned_index"], local)
            if not wide.get("completed") or wide.get("error"):
                raise ValueError("completed frozen WIDE model evidence required")
            if pin["capture_RESULT"]["sha256"] != batch["capture_result"]["sha256"]:
                raise ValueError("WIDE index belongs to a different capture")
            if pin["WIDE_RESULT"]["sha256"] != extra["result"]["sha256"]:
                raise ValueError("WIDE result/index identity differs")
            wide_rows = {r["cell"]: r for r in wide["rows"]}
            if set(wide_rows) != {c["id"] for c in plan["cells"]}:
                raise ValueError("all frozen WIDE cells must be retained")
            pinned = pin["source_and_reference_sha256"]
        darks = [_stage(root, stages.get(name, {"name": name}), plan["dark_frames_per_bookend"])
                 for name in ("DARK_BEFORE", "DARK_AFTER")]
        batch_cells = []
        for c in plan["cells"]:
            inp, mask = inputs[c["input_id"]], masks[c["mask_id"]]
            input_ref = _ref(root, inp["raw96"], inp["raw96_sha256"], "DMD_RAW96_identity_only")
            mask_ref = _ref(root, mask["path"], mask["sha256"], "SLM_raw_gray_identity_only")
            identity_sources = []
            _capture(root, input_ref, identity_sources)
            _capture(root, mask_ref, identity_sources)
            models = {}
            if "frozen_prediction" in c:
                m = c["frozen_prediction"]
                models["pre_capture_frozen"] = _ref(root, m["path"], m["sha256"], "pre_capture_MODEL_ADU_not_real")
                models["pre_capture_frozen"]["provenance"] = "frozen_before_original_capture; old seen replay"
            if extra:
                row = wide_rows[c["id"]]
                if row["input_id"] != c["input_id"] or row["mask_id"] != c["mask_id"]:
                    raise ValueError("WIDE cell/input/mask association mismatch")
                for label in ("baseline_zero_wavefront", "fitted_shared_wavefront"):
                    # Original evaluate() recipe + PREEXISTING byte-pinned visual
                    # index, not a claim that basename alone establishes identity.
                    path = _path(root, extra["directory"]) / (c["id"] + "_" + label + "_CCD_MEAN.npy")
                    sha = pinned.get(str(path))
                    if sha is None:
                        raise ValueError("model lacks preexisting explicit array hash")
                    models["wide_" + label] = _ref(root, path, sha, "postfit_seen_MODEL_ADU_not_real")
                    models["wide_" + label]["provenance"] = "postfit same-seen-capture model; NOT fresh validation"
            model_sources = []
            for ref in models.values():
                _capture(root, ref, model_sources)
            record = dict(schema=SCHEMA, id=batch["id"] + ":" + c["id"], batch=batch["id"],
                cell=c["id"], legacy_root=str(root), input_id=c["input_id"], mask_id=c["mask_id"],
                original_split=c.get("split"), historical_seen_development=True, new_holdout=False,
                input={"raw96": input_ref, "history_times_s": inp.get("history_times_s"),
                       "on_pixels": inp.get("on_pixels"), "uniform_illumination_assumed": False},
                mask={"asset": mask_ref, "kind": mask.get("kind"), "family": mask.get("family"),
                      "sign": mask.get("sign"), "simulation_key": mask.get("simulation_key")},
                capture=_stage(root, stages.get(c["id"], {"name": c["id"]}), plan["frames_per_cell"]),
                dark_captures=deepcopy(darks), models=models, regions=region_ref, head=head_ref,
                roi=deepcopy(contract["roi"]), expected_active=deepcopy(contract["expected_active"]),
                freshness_settle_ns=int(round(float(plan.get("settle_seconds", 1.5)) * 1e9)),
                capture_header_complete=header_ok, capture_stage_coverage_complete=stage_coverage,
                original_frame_pattern_sync_verified=not plan.get("hardware_frame_pattern_sync_NOT_verified", True),
                source_refs=deepcopy(local + identity_sources + model_sources),
                units={"real": "native Mono8 ADU", "models": "native expected-code ADU, float; no normalization"},
                limitations=["OLD seen data; not a new holdout or qualification", "ADU sums are not absolute diffraction efficiency",
                             "no per-cell resize, mirror, registration, gain normalization, or automatic dark subtraction"])
            record["loader_sources"] = deepcopy(sources[:2])
            sources.extend(x for x in identity_sources + model_sources if x not in sources)
            cells.append(record)
            batch_cells.append(record)
        for pair in plan.get("directional_probe_pairs", []):
            for inp in plan["inputs"]:
                matches = [next((r for r in batch_cells if r["input_id"] == inp["id"] and r["mask_id"] == name), None)
                           for name in pair]
                if len(pair) != 2 or any(r is None for r in matches):
                    raise ValueError("declared probe does not have both exact input/mask cells")
                signed = pair[0].endswith("_minus") and pair[1] == pair[0][:-6] + "_plus"
                probes.append(dict(id=batch["id"] + ":" + inp["id"] + ":" + "__".join(pair),
                    batch=batch["id"], input_id=inp["id"], minus_cell_id=matches[0]["id"], plus_cell_id=matches[1]["id"],
                    mask_pair=pair, kind="signed_finite_raw_gray_probe" if signed else "asymmetric_finite_mask_change",
                    direction="second_minus_first", same_input_sha256=matches[0]["input"]["raw96"]["sha256"],
                    infinitesimal_phase_gradient=False, gray_amplitude_is_NOT_calibrated_phase=True,
                    historical_seen_development=True))
        sources.extend(x for x in local if x not in sources)
        batch_audit.append(dict(batch=batch["id"], cells=len(batch_cells), capture_header_complete=header_ok,
                                stage_coverage_complete=stage_coverage, frames_per_cell=plan["frames_per_cell"]))
    _tail(sources)
    return dict(cells=cells, probe_groups=probes, audit=dict(schema=SCHEMA, metadata_only=True,
        CCD_pixels_decoded=0, model_arrays_decoded=0, batches=batch_audit, cells=len(cells),
        historical_seen_NOT_fresh_holdout=True, no_hardware_or_training=True, sources=sources))


def _load_stage(cell, stage, sources):
    root = cell["legacy_root"]
    shape = (int(cell["roi"]["Height"]), int(cell["roi"]["Width"]))
    checks = dict(repeat_count_matches=len(stage["snapshots"]) == stage["expected_repeats"],
                  framebuffer_checked=stage["actual_framebuffer_checked"],
                  camera_active_matches=True, frame_ids_monotonic=True, timestamps_monotonic=True,
                  freshness_after_verified_display=True, recorded_frame_matches_metadata=True,
                  frame_dimensions_complete=True, within_original_max_ADU_gate=True)
    arrays, metadata, seen = [], [], set()
    last_id = last_tick = -1
    for snap in stage["snapshots"]:
        if snap["raw"]["path"] in seen:
            raise ValueError("duplicate real snapshot cannot count as independent repetition")
        seen.add(snap["raw"]["path"])
        meta = _json(root, snap["metadata"], sources)
        a = _array(root, snap["raw"], sources)
        if a.dtype != np.uint8 or a.shape != shape:
            raise ValueError("native uint8 CCD required, no silent resize/cast")
        frame = meta.get("frame", {})
        checks["camera_active_matches"] &= meta.get("camera_active") == cell["expected_active"]
        checks["recorded_frame_matches_metadata"] &= snap["recorded_frame"] is not None and frame == snap["recorded_frame"]
        checks["frame_dimensions_complete"] &= bool(frame.get("Width") == shape[1] and frame.get("Height") == shape[0]
            and frame.get("Stride") == shape[1] and frame.get("BufferSize") == shape[0] * shape[1]
            and frame.get("pixel_format") == "Mono8")
        fid, tick = frame.get("FrameID"), frame.get("TimeStamp")
        checks["frame_ids_monotonic"] &= isinstance(fid, int) and fid > last_id
        checks["timestamps_monotonic"] &= isinstance(tick, int) and tick > last_tick
        if isinstance(fid, int):
            last_id = fid
        if isinstance(tick, int):
            last_tick = tick
        started = frame.get("host_get_started_unix_ns")
        verified = stage.get("verified_unix_ns")
        checks["freshness_after_verified_display"] &= bool(isinstance(started, int) and isinstance(verified, int)
            and started >= verified + cell["freshness_settle_ns"] and meta.get("display_revision") == stage["revision"])
        checks["within_original_max_ADU_gate"] &= int(a.max()) < 240
        arrays.append(a)
        metadata.append(meta)
    stack = np.stack(arrays) if arrays else np.empty((0, *shape), dtype=np.uint8)
    association = bool(metadata) and all(m.get("frame_pattern_association_verified") is True for m in metadata)
    clock = bool(metadata) and all(m.get("frame", {}).get("camera_clock_relationship_verified") is True for m in metadata)
    return stack, dict(name=stage["name"], checks=checks, checks_passed=all(checks.values()),
                      frame_pattern_association_verified=association, camera_clock_relationship_verified=clock,
                      snapshots=metadata)


def load_historical_cell(cell):
    """Native arrays + source identities; missing temporal evidence is flagged.

    Hash/dtype/shape corruption raises. Missing freshness/association/repetition
    evidence is retained in metadata, never repaired, padded, or filtered.
    ``dark_stack`` combines bookends only for caller convenience; their exact
    boundary and individual metadata are retained in ``metadata.dark_bookends``.
    """
    if cell.get("schema") != SCHEMA:
        raise ValueError("unsupported historical cell schema")
    root, sources = cell["legacy_root"], []
    _tail(cell.get("loader_sources", []))
    sources.extend(deepcopy(cell.get("loader_sources", [])))
    for ref in cell["source_refs"]:
        _capture(root, ref, sources)
    real, timing = _load_stage(cell, cell["capture"], sources)
    models = {}
    for name, ref in cell["models"].items():
        a = _array(root, ref, sources)
        if a.shape != real.shape[1:] or not np.issubdtype(a.dtype, np.floating) or not np.isfinite(a).all():
            raise ValueError("native finite floating MODEL ADU required")
        if a.min() < 0 or a.max() > 255:
            raise ValueError("MODEL expected-code ADU outside declared Mono8 range")
        models[name] = a
    regions = _array(root, cell["regions"], sources)
    if (regions.shape != (8, *real.shape[1:]) or not np.isin(regions, [0, 1]).all()
            or not np.all(regions.sum(0) == 1)):
        raise ValueError("exact original binary fullROI8 partition required")
    head = _json(root, cell["head"], sources)
    w, b = np.asarray(head["weight_raw_ADU_to_rad"]), np.asarray(head["bias_rad"])
    if w.shape != (2, 8) or b.shape != (2,) or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("same original native8 diagnostic RAW18 head required")
    darks, dark_meta = [], []
    for stage in cell["dark_captures"]:
        a, m = _load_stage(cell, stage, sources)
        darks.append(a)
        dark_meta.append(m)
    dark = np.concatenate(darks) if darks and any(len(a) for a in darks) else None
    _tail(sources)
    metadata = dict(cell=deepcopy(cell), capture=timing, dark_bookends=dark_meta,
        dark_bookend_counts=[len(a) for a in darks],
        complete_freshness_evidence=bool(cell["capture_header_complete"] and cell["capture_stage_coverage_complete"]
            and timing["checks_passed"]),
        frame_pattern_association_verified=timing["frame_pattern_association_verified"],
        dark_metadata_matched=bool(dark_meta and all(m["checks_passed"] for m in dark_meta)),
        independent_hardware_qualification=False, historical_seen_development=True,
        no_rescale_registration_normalization_or_dark_subtraction=True,
        ADU_flux_NOT_absolute_diffraction_efficiency=True)
    metadata.update(background_kind="DMD_and_SLM_zero_optical_background_NOT_sensor_physical_dark",
                    sensor_physical_dark_measurement_verified=False)
    return dict(real_stack=real, models=models, dark_stack=dark, regions=regions,
                head={"weight_raw_ADU_to_rad": w.tolist(), "bias_rad": b.tolist()},
                metadata=metadata, sources=sources)
