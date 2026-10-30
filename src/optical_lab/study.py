"""Small, offline static-case assets shared by prediction, capture and analysis.

One case is one explicitly selected binary DMD history and one Z/A/B probe
family.  This does not generalize, overwrite, or reinterpret the sealed pilot.
No imports of a GUI, camera SDK, Torch, or MuJoCo occur here.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

from .probes import _npy_bytes, _png_bytes
from .storage import sha256, utc_now, write_json_new


SCHEMA = "optical-lab-static-case-assets-1"
DESIGN_SCHEMA = "optical-lab-static-study-design-1"
_ID = re.compile(r"[A-Z][A-Z0-9_]{1,63}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROLES = {"screening", "validation", "matched_null"}
_FIXED = {"canvas_shape_yx": [600, 600], "interior_shape_yx": [512, 512],
          "interior_origin_xy": [44, 44], "outside_gray": 0,
          "gray_domain": "raw_uint8_not_calibrated_phase",
          "canonical_coordinates": "x_right_y_down_no_flip_no_LUT"}


def _json(data):
    return json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def probe_arrays(spec):
    """Reconstruct exact constant-macro binary gratings (or all-zero null)."""
    if set(spec) != {*_FIXED, "macro_block_side_physical_px", "period_physical_px", "axis", "gray_pair", "kind"}:
        raise ValueError("exact declared probe fields required")
    if any(_canonical(spec[k]) != _canonical(v) for k, v in _FIXED.items()):
        raise ValueError("fixed canonical600 / interior512 geometry required")
    block, period, pair = (spec[k] for k in ("macro_block_side_physical_px", "period_physical_px", "gray_pair"))
    if (type(block) is not int or block not in (4, 8, 16, 32)
            or type(period) is not int or period not in (64, 128, 256)
            or period % (2 * block) or spec["axis"] not in ("x", "y")
            or not isinstance(pair, list) or len(pair) != 2
            or any(type(v) is not int for v in pair)):
        raise ValueError("integer aligned macro/physical period/axis/raw pair required")
    if spec["kind"] == "matched_null":
        if pair != [0, 0]:
            raise ValueError("matched null must use raw0/0")
    elif spec["kind"] != "binary_grating" or pair not in ([0, 32], [0, 64], [0, 128], [0, 192]):
        raise ValueError("unsupported raw gray probe pair")
    row = np.where(np.arange(512) % period < period // 2, pair[0], pair[1]).astype(np.uint8)
    interior = np.broadcast_to(row[None, :], (512, 512))
    if spec["axis"] == "y":
        interior = interior.T
    arrays = {name: np.zeros((600, 600), dtype=np.uint8) for name in ("Z", "A", "B")}
    arrays["A"][44:556, 44:556] = interior
    arrays["B"][44:556, 44:556] = pair[0] + pair[1] - interior
    for a in arrays.values():
        tiles = a[44:556, 44:556].reshape(512 // block, block, 512 // block, block)
        if not np.all(tiles == tiles[:, :1, :, :1]):
            raise ValueError("probe is not constant inside declared macro blocks")
    return arrays


def _design(data):
    if (data.get("schema") != DESIGN_SCHEMA or data.get("hardware_actions_enabled") is not False
            or not _ID.fullmatch(data.get("id", ""))):
        raise ValueError("explicit offline study design required")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("nonempty frozen case list required")
    ids = []
    for case in data["cases"]:
        if (not _ID.fullmatch(case.get("id", "")) or case.get("role") not in _ROLES
                or case.get("input_id") not in data["inputs"]):
            raise ValueError("declared unique case/input/split required")
        probe_arrays(case["probe_spec"])
        if (case["role"] == "matched_null") != (case["probe_spec"]["kind"] == "matched_null"):
            raise ValueError("null must have an explicitly separate null role")
        ids.append(case["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ID")
    return data


def _input(raw, dmd):
    if (raw.shape != (4, 1, 96, 96) or raw.dtype != np.uint8
            or not np.isin(raw, [0, 255]).all() or not np.any(raw)):
        raise ValueError("nonzero exact binary uint8 RAW96 history required")
    mosaic = np.block([[raw[0, 0], raw[1, 0]], [raw[2, 0], raw[3, 0]]])
    expected = np.pad(mosaic.repeat(3, 0).repeat(3, 1), ((12, 12), (12, 12)))
    if dmd.dtype != np.uint8 or dmd.shape != (600, 600) or not np.array_equal(dmd, expected):
        raise ValueError("DMD must match the chronological 3x RAW96 mosaic exactly")


def _component_count(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    count = 0
    while remaining:
        count += 1
        pending = [remaining.pop()]
        while pending:
            y, x = pending.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    point = (y + dy, x + dx)
                    if point in remaining:
                        remaining.remove(point)
                        pending.append(point)
    return count


def encode_history(raw, radius):
    """Explicit RAW96 disk dilation, not optical zoom or a physics change.

    Each frame is independently zero-padded. No roll, resampling, recentering,
    new state, or per-image optimization. Pixel centroids need not stay exact.
    """
    if type(radius) is not int or radius not in (0, 1, 2):
        raise ValueError("integer RAW96 disk radius0/1/2 required")
    if raw.shape != (4, 1, 96, 96) or raw.dtype != np.uint8 or not np.isin(raw, (0, 255)).all():
        raise ValueError("four exact binary uint8 RAW96 frames required")
    encoded = np.zeros_like(raw)
    rows = []
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
               if dy * dy + dx * dx <= radius * radius]
    for i, image in enumerate(raw[:, 0]):
        mask = image > 0
        if not mask.any() or _component_count(mask) != 1:
            raise ValueError("each source frame must have one nonempty 8-connected ball")
        grown = np.zeros_like(mask)
        padded = np.pad(mask, radius)
        for dy, dx in offsets:
            grown |= padded[radius + dy:radius + dy + 96, radius + dx:radius + dx + 96]
        if _component_count(grown) != 1:
            raise ValueError("encoded ball must remain one component")
        old_yx, new_yx = np.argwhere(mask), np.argwhere(grown)
        old_center, new_center = old_yx.mean(0), new_yx.mean(0)
        delta = new_center - old_center
        lo, hi = new_yx.min(0), new_yx.max(0)
        margin = int(min(*lo, *(95 - hi)))
        if margin < 2 or float(np.linalg.norm(delta)) > .15:
            raise ValueError("encoding would clip/approach edge or shift centroid beyond0.15 RAW pixels")
        encoded[i, 0] = grown.astype(np.uint8) * 255
        rows.append(dict(frame_index=i, source_on=int(mask.sum()), encoded_on=int(grown.sum()),
                         area_ratio=float(grown.sum() / mask.sum()),
                         source_centroid_xy=old_center[::-1].tolist(),
                         encoded_centroid_xy=new_center[::-1].tolist(),
                         centroid_delta_xy=delta[::-1].tolist(), centroid_delta_L2_raw_px=float(np.linalg.norm(delta)),
                         encoded_bbox_xyxy=[int(lo[1]), int(lo[0]), int(hi[1] + 1), int(hi[0] + 1)],
                         minimum_edge_margin_raw_px=margin, connected_components_8=1))
    return encoded, rows


def prepare_encoding(plan_path, output, *, expected_plan_sha256):
    """Freeze a tiny input bank + existing static-study design; no device calls."""
    plan_path = Path(plan_path).resolve(strict=True)
    if not _SHA.fullmatch(expected_plan_sha256 or "") or sha256(plan_path) != expected_plan_sha256:
        raise ValueError("external encoding plan SHA required")
    plan_bytes = plan_path.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != expected_plan_sha256:
        raise ValueError("captured encoding plan bytes differ from external SHA")
    plan = _json(plan_bytes)
    expected_cases = [("E00_BASELINE_BEFORE", 0), ("E01_DILATE_R1", 1),
                      ("E02_DILATE_R2", 2), ("E03_BASELINE_AFTER", 0)]
    if (plan.get("schema") != "optical-lab-encoding-plan-1" or plan.get("hardware_actions_enabled") is not False
            or _canonical(plan.get("case_order")) != _canonical([dict(id=name, radius_logical_px=r) for name, r in expected_cases])
            or _canonical(plan.get("algorithm")) != _canonical(dict(kind="integer_Euclidean_disk_dilation_per_RAW96_frame",
                border="zero_outside_each_frame_no_wrap", centroid_L2_limit_raw_px=.15,
                minimum_edge_margin_raw_px=2, component_connectivity=8))
            or type(plan.get("frames_per_state")) is not int or plan["frames_per_state"] != 8):
        raise ValueError("exact prospective encoding algorithm/order/limits required")
    source = plan["source_input"]
    if source["history_ages_s"] != [.18, .12, .06, 0.]:
        raise ValueError("unchanged four temporal inputs required")
    blobs = {}
    for key in ("raw", "dmd"):
        path = Path(source[key + "_path"])
        if not path.is_absolute() or path.is_symlink() or sha256(path) != source[key + "_sha256"]:
            raise ValueError("source input SHA changed: " + key)
        blobs[key] = path.read_bytes()
        if hashlib.sha256(blobs[key]).hexdigest() != source[key + "_sha256"]:
            raise ValueError("captured source input bytes differ from declared SHA: " + key)
    raw, dmd = [np.load(io.BytesIO(blobs[k]), allow_pickle=False) for k in ("raw", "dmd")]
    _input(raw, dmd)
    probe_arrays(plan["probe_spec"])
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError("new regular encoding bank directory required")
    root = Path(output).resolve()
    files = {"SOURCE_RAW96.npy": blobs["raw"], "SOURCE_DMD.npy": blobs["dmd"],
             "PLAN.json": plan_bytes, "generator_source.py": Path(__file__).read_bytes(),
             "png_encoder_source.py": Path(__file__).with_name("probes.py").read_bytes()}
    design = dict(schema=DESIGN_SCHEMA, id=plan["id"], hardware_actions_enabled=False,
                  inputs={}, cases=[], collection_order=[name for name, _ in expected_cases],
                  display=plan["display"], purpose=plan["purpose"],
                  encoding_plan=dict(path=str(plan_path), sha256=expected_plan_sha256))
    rows = []
    for name, radius in expected_cases:
        encoded, geometry = encode_history(raw, radius)
        panel = np.pad(np.block([[encoded[0, 0], encoded[1, 0]],
                                [encoded[2, 0], encoded[3, 0]]]).repeat(3, 0).repeat(3, 1), 12)
        _input(encoded, panel)
        raw_name, dmd_name = name + "_RAW96.npy", name + "_DMD.npy"
        files[raw_name] = blobs["raw"] if radius == 0 else _npy_bytes(encoded)
        files[dmd_name] = blobs["dmd"] if radius == 0 else _npy_bytes(panel)
        files[name + "_DMD.png"] = _png_bytes(panel)
        entry = dict(raw_path=str(root / raw_name), raw_sha256=hashlib.sha256(files[raw_name]).hexdigest(),
                     dmd_path=str(root / dmd_name), dmd_sha256=hashlib.sha256(files[dmd_name]).hexdigest(),
                     history_ages_s=source["history_ages_s"], selection_rule=plan["purpose"],
                     encoding=dict(radius_logical_px=radius, source_input=source, per_frame_geometry=geometry))
        design["inputs"][name] = entry
        design["cases"].append(dict(id=name, role="screening", input_id=name, probe_spec=plan["probe_spec"]))
        rows.append(dict(case_id=name, radius_logical_px=radius, **entry,
                         raw_on=int((encoded > 0).sum()), dmd_on=int((panel > 0).sum()),
                         dmd_canvas_fraction=float((panel > 0).mean()),
                         per_frame_geometry=geometry))
    _design(design)
    files["STUDY_DESIGN.json"] = json.dumps(design, ensure_ascii=False, indent=2, allow_nan=False).encode()
    result = dict(schema="optical-lab-encoding-inputs-1", completed=True, protocol=dict(path=str(plan_path), sha256=expected_plan_sha256),
                  source_input=source, rows=rows, hardware_accessed=False, physics_or_trajectory_changed=False,
                  exact_centroid_preservation_claimed=False, whole_mosaic_zoom=False, no_training=True,
                  unchanged_video_canvas=[600, 600], generated_study_design="STUDY_DESIGN.json")
    files["RESULT.json"] = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode()
    if plan_path.read_bytes() != plan_bytes or Path(__file__).read_bytes() != files["generator_source.py"]:
        raise ValueError("encoding plan/generator changed during preparation")
    for key in ("raw", "dmd"):
        if Path(source[key + "_path"]).read_bytes() != blobs[key]:
            raise ValueError("source input changed during preparation")
    root.mkdir(exist_ok=False)
    try:
        for name, blob in files.items():
            with (root / name).open("xb") as stream:
                stream.write(blob)
        if (plan_path.read_bytes() != plan_bytes or Path(__file__).read_bytes() != files["generator_source.py"]
                or Path(__file__).with_name("probes.py").read_bytes() != files["png_encoder_source.py"]
                or any(Path(source[k + "_path"]).read_bytes() != blobs[k] for k in ("raw", "dmd"))):
            raise ValueError("encoding source/implementation changed before sealing")
        write_json_new(root / "MANIFEST.json", dict(schema="optical-lab-encoding-inputs-1", completed=True,
            created_utc=utc_now(), file_sha256={name:hashlib.sha256(blob).hexdigest() for name, blob in files.items()},
            hardware_accessed=False, control_release_approved=False))
    except Exception as exc:
        write_json_new(root / "FAILED.json", dict(error=type(exc).__name__ + ": " + str(exc), hardware_accessed=False))
        raise
    return dict(completed=True, manifest_sha256=sha256(root / "MANIFEST.json"),
                rows=[dict(case_id=r["case_id"], radius_logical_px=r["radius_logical_px"], dmd_on=r["dmd_on"]) for r in rows],
                hardware_accessed=False)


def prepare_case(design_path, case_id, output):
    """Create a NEW self-contained asset directory, not a projection request."""
    design_path = Path(design_path).resolve(strict=True)
    design_bytes = design_path.read_bytes()
    design = _design(_json(design_bytes))
    matches = [c for c in design["cases"] if c["id"] == case_id]
    if len(matches) != 1:
        raise ValueError("case not in frozen study design")
    case = matches[0]
    source = design["inputs"][case["input_id"]]
    source_paths = {k: Path(source[k + "_path"]) for k in ("raw", "dmd")}
    for key, path in source_paths.items():
        if not path.is_absolute() or path.is_symlink() or sha256(path) != source[key + "_sha256"]:
            raise ValueError("explicit source input identity changed: " + key)
    if source["history_ages_s"] != [.18, .12, .06, 0.]:
        raise ValueError("fixed real-platform history ages required")
    raw, dmd = [np.load(source_paths[k], allow_pickle=False) for k in ("raw", "dmd")]
    _input(raw, dmd)
    # Preserve exact source NPY bytes as well as pixel identity.
    files = {"RAW96.npy": source_paths["raw"].read_bytes(), "DMD.npy": source_paths["dmd"].read_bytes(),
             "design_snapshot.json": design_bytes, "generator_source.py": Path(__file__).read_bytes(),
             "png_encoder_source.py": Path(__file__).with_name("probes.py").read_bytes()}
    arrays = probe_arrays(case["probe_spec"])
    assets = {}
    for name, array in arrays.items():
        stem = f"SLM_{name}_CANONICAL600_RAW_GRAY_NO_LUT"
        files[stem + ".npy"] = _npy_bytes(array)
        files[stem + ".png"] = _png_bytes(array)
        assets[name] = {"npy": stem + ".npy", "png": stem + ".png",
                        "pixel_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    output = Path(output)
    output.mkdir(exist_ok=False)
    output = output.resolve(strict=True)
    try:
        for name, data in files.items():
            with (output / name).open("xb") as stream:
                stream.write(data)
        fixed = {"id": case["input_id"], "raw_path": str(output / "RAW96.npy"),
                 "raw_sha256": hashlib.sha256(files["RAW96.npy"]).hexdigest(),
                 "dmd_path": str(output / "DMD.npy"), "dmd_sha256": hashlib.sha256(files["DMD.npy"]).hexdigest(),
                 "history_ages_s": source["history_ages_s"], "selection_rule": source["selection_rule"]}
        manifest = {"schema": SCHEMA, "created_utc": utc_now(), "asset_generation_complete": True,
                    "assets": assets, "fixed_input": fixed, "input_provenance": source,
                    "input_pixel_key_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                    "probe_spec": case["probe_spec"],
                    "design": {"path": str(design_path), "sha256": hashlib.sha256(design_bytes).hexdigest(),
                               "case_id": case_id, "role": case["role"], "snapshot": "design_snapshot.json"},
                    "file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                    "implementation": {"module": "optical_lab.study", "snapshot": "generator_source.py",
                                       "source_sha256": hashlib.sha256(files["generator_source.py"]).hexdigest()},
                    "captured_frames_at_asset_creation": 0, "hardware_accessed": False,
                    "projection_executed_at_asset_creation": False, "LUT_applied": False,
                    "control_release_approved": False, "hardware_release_approved": False}
        if design_path.read_bytes() != design_bytes or Path(__file__).read_bytes() != files["generator_source.py"]:
            raise ValueError("design/generator changed during preparation")
        for key, path in source_paths.items():
            if sha256(path) != source[key + "_sha256"]:
                raise ValueError("source input changed during preparation")
        write_json_new(output / "MANIFEST.json", manifest)
        return verify_case_assets(output, expected_manifest_sha256=sha256(output / "MANIFEST.json"))
    except Exception as exc:
        write_json_new(output / "FAILED.json", {"error": type(exc).__name__ + ": " + str(exc),
                                              "hardware_accessed": False})
        raise


def verify_case_assets(directory, *, expected_manifest_sha256):
    """Offline strict archive + exact array semantics; never executes a snapshot."""
    root = Path(directory)
    if root.is_symlink() or not _SHA.fullmatch(expected_manifest_sha256 or ""):
        raise ValueError("explicit external case manifest SHA required")
    root = root.resolve(strict=True)
    digest = sha256(root / "MANIFEST.json")
    if digest != expected_manifest_sha256:
        raise ValueError("case manifest SHA mismatch")
    manifest = _json((root / "MANIFEST.json").read_bytes())
    if manifest.get("schema") != SCHEMA or manifest.get("asset_generation_complete") is not True:
        raise ValueError("complete static case manifest required")
    if set(manifest.get("assets", {})) != {"Z", "A", "B"} or manifest.get("captured_frames_at_asset_creation") != 0:
        raise ValueError("exact offline Z/A/B case assets required")
    for key in ("hardware_accessed", "projection_executed_at_asset_creation", "LUT_applied",
                "control_release_approved", "hardware_release_approved"):
        if manifest.get(key) is not False:
            raise ValueError("offline case cannot claim hardware/release: " + key)
    names = {"RAW96.npy", "DMD.npy", "design_snapshot.json", "generator_source.py", "png_encoder_source.py"}
    names.update(f"SLM_{k}_CANONICAL600_RAW_GRAY_NO_LUT.{s}" for k in ("Z", "A", "B") for s in ("npy", "png"))
    hashes = manifest["file_sha256"]
    if set(hashes) != names or {p.name for p in root.iterdir()} != names | {"MANIFEST.json"}:
        raise ValueError("exact regular case artifact set required")
    captured = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("regular non-symlink case files required")
        captured[name] = path.read_bytes()
        if hashlib.sha256(captured[name]).hexdigest() != hashes[name]:
            raise ValueError("case file SHA mismatch: " + name)
    design = _design(_json(captured["design_snapshot.json"]))
    info = manifest["design"]
    if info["snapshot"] != "design_snapshot.json" or info["sha256"] != hashes["design_snapshot.json"]:
        raise ValueError("case design snapshot binding mismatch")
    cases = [c for c in design["cases"] if c["id"] == info["case_id"]]
    if len(cases) != 1:
        raise ValueError("case missing from design snapshot")
    case = cases[0]
    if case["role"] != info["role"] or _canonical(case["probe_spec"]) != _canonical(manifest["probe_spec"]):
        raise ValueError("case probe/split differs from frozen design")
    source = design["inputs"][case["input_id"]]
    if _canonical(manifest["input_provenance"]) != _canonical(source):
        raise ValueError("input provenance changed")
    fixed = manifest["fixed_input"]
    if fixed != {"id": case["input_id"], "raw_path": str(root / "RAW96.npy"), "raw_sha256": hashes["RAW96.npy"],
                 "dmd_path": str(root / "DMD.npy"), "dmd_sha256": hashes["DMD.npy"],
                 "history_ages_s": source["history_ages_s"], "selection_rule": source["selection_rule"]}:
        raise ValueError("case fixed input/file association mismatch")
    if hashes["RAW96.npy"] != source["raw_sha256"] or hashes["DMD.npy"] != source["dmd_sha256"]:
        raise ValueError("case input differs from declared source bytes")
    raw, dmd = [np.load(io.BytesIO(captured[k]), allow_pickle=False) for k in ("RAW96.npy", "DMD.npy")]
    _input(raw, dmd)
    if hashlib.sha256(raw.tobytes()).hexdigest() != manifest["input_pixel_key_sha256"]:
        raise ValueError("input pixel identity differs")
    for name, expected in probe_arrays(manifest["probe_spec"]).items():
        stem = f"SLM_{name}_CANONICAL600_RAW_GRAY_NO_LUT"
        value = np.load(io.BytesIO(captured[stem + ".npy"]), allow_pickle=False)
        if value.dtype != np.uint8 or not np.array_equal(value, expected):
            raise ValueError("case probe semantics differ: " + name)
        if captured[stem + ".png"] != _png_bytes(value):
            raise ValueError("fixed raw-gray PNG mismatch: " + name)
        if manifest["assets"][name] != {"npy": stem + ".npy", "png": stem + ".png",
                                     "pixel_sha256": hashlib.sha256(expected.tobytes()).hexdigest()}:
            raise ValueError("case probe asset table differs")
    if manifest["implementation"] != {"module": "optical_lab.study", "snapshot": "generator_source.py",
                                     "source_sha256": hashes["generator_source.py"]}:
        raise ValueError("archived generator identity differs")
    for name, data in captured.items():
        if (root / name).read_bytes() != data:
            raise ValueError("case file changed during verification")
    if sha256(root / "MANIFEST.json") != digest:
        raise ValueError("case manifest changed during verification")
    return {"verified": True, "manifest_sha256": digest, "manifest": manifest,
            "hardware_accessed": False, "control_release_approved": False}
