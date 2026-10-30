"""Explicit current-direction static capture through ONE existing GUI owner.

Default import is stdlib-only. Never owns an SDK, restarts a camera/window,
changes the SLM mode, or runs an actor. capture() is the explicit action API.
"""
from __future__ import annotations
import argparse
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from . import legacy_gui as old

LAB = Path(__file__).resolve().parents[2]
LEGACY = LAB.parent/"ballplate_fourframe96_deploy"
SCHEMA = "optical-lab-current-direction-plan-1"
RESULT_SCHEMA = "optical-lab-current-direction-capture-1"
INPUT_KEYS = ("REF", "VX_PLUS", "VX_MINUS", "VY_PLUS", "VY_MINUS")
STATE_IDS = ("BG0", "A_REF0", "A_VX_PLUS", "A_VX_MINUS", "A_REF1", "A_VY_MINUS", "A_VY_PLUS", "A_REF2",
             "BG1", "B_REF0", "B_VY_PLUS", "B_VY_MINUS", "B_REF1", "B_VX_MINUS", "B_VX_PLUS", "B_REF2", "BG2")


def states():
    return [dict(id=n, input_key=None if n.startswith("BG") else "REF" if "REF" in n else n[2:],
                 illuminated=not n.startswith("BG"), repeat=8) for n in STATE_IDS]


def _file(path, sha):
    p = Path(path)
    if (not p.is_absolute() or ".." in p.parts or p.is_symlink()
            or not p.resolve().is_relative_to(LAB.parent.resolve()) or not p.is_file()
            or not isinstance(sha, str) or old._SHA.fullmatch(sha) is None
            or old._sha(p.read_bytes()) != sha):
        raise ValueError("explicit frozen path/SHA required: " + str(path))
    return p.resolve()


def _archives(plan):
    for r in plan["frozen_archives"]:
        p = _file(r["manifest_path"], r["manifest_sha256"])
        m = old._read(p); files = m["file_sha256"]
        if not files or not isinstance(files, dict): raise ValueError("nonempty frozen archive file table required")
        for name, sha in files.items():
            rel = Path(name); f = p.parent/rel
            if rel.is_absolute() or ".." in rel.parts or not f.resolve().is_relative_to(p.parent):
                raise ValueError("frozen archive path escape")
            _file(f, sha)


def _sources(plan, path, sha):
    _file(path, sha)
    for name, digest in plan["source_sha256"].items(): _file(name, digest)
    _file(Path(plan["session"])/"runtime_sources.json", plan["runtime_sha256"])


def validate_plan(path, expected_sha):
    path = _file(Path(path).resolve(), expected_sha); blob = path.read_bytes()
    if old._sha(blob) != expected_sha: raise ValueError("PLAN changed while reading")
    p = json.loads(blob)
    if (p.get("schema") != SCHEMA or p.get("profile") != "current_direction_v1"
            or p.get("states") != states() or type(p.get("pid")) is not int or p["pid"] <= 0
            or not isinstance(p.get("process_start"), str) or not p["process_start"].isdigit()
            or type(p.get("maximum_seconds")) is not int or p["maximum_seconds"] != 360
            or p.get("cleanup_reserve_seconds") != 15
            or p.get("software_settle_wait_seconds") != 1.8
            or p.get("minimum_postcommit_frame_start_seconds") != 1.5
            or set(p.get("inputs", {})) != set(INPUT_KEYS)):
        raise ValueError("exact current_direction_v1 /17x8/360s plan required")
    if any(type(s["repeat"]) is not int or type(s["illuminated"]) is not bool for s in p["states"]):
        raise ValueError("integer8 repeats and boolean illumination required")
    if (p["settings"].get("slm") != "lens" or p["settings"].get("full_panel") is not False
            or p["settings"].get("focal_mm") != 400
            or p["roi"] != {k: int(p["expected_active"][k]) for k in ("OffsetX", "OffsetY", "Width", "Height")}
            or p["roi"]["Width"] != 552 or p["roi"]["Height"] != 554):
        raise ValueError("fixed current manual lens400 and native ROI contract required")
    auth = p.get("authorization", {})
    if not isinstance(auth.get("confirmed_utc"), str) or not auth.get("statement"):
        raise ValueError("explicit fresh low-power/protection/exclusive authorization required")
    if not Path(p["session"]).is_absolute() or not Path(p["session"]).resolve().is_relative_to(LAB/"experiments"):
        raise ValueError("explicit current lab owner session required")
    required_sources = (Path(__file__).resolve(), Path(old.__file__).resolve(),
                        LAB/"src/optical_lab/protocol.py", Path(p["session"])/"runtime_sources.json")
    if any(str(f) not in p["source_sha256"] for f in required_sources):
        raise ValueError("driver/helper/protocol/runtime source binding missing")
    runtime = old._read(_file(Path(p["session"])/"runtime_sources.json", p["runtime_sha256"]))
    if runtime["pid"] != p["pid"] or (len(runtime["sources"]), len(runtime["source_configs"])) != (78, 2):
        raise ValueError("current original runtime78+2 identity required")
    for group in ("sources", "source_configs"):
        for f, sha in runtime[group].items():
            if p["source_sha256"].get(f) != sha: raise ValueError("runtime binding absent/conflicting")
    declaration = Path(p["session"])/"spatial_preview_extension.json"
    if str(declaration) not in p["source_sha256"]: raise ValueError("owner extension declaration unbound")
    for f, sha in old._read(_file(declaration, p["source_sha256"][str(declaration)]))["new_source_sha256"].items():
        if p["source_sha256"].get(f) != sha: raise ValueError("owner extension source unbound")
    if not p.get("frozen_archives"): raise ValueError("assets and prospective prediction archives required")
    for r in p["frozen_archives"]:
        if p["source_sha256"].get(r["manifest_path"]) != r["manifest_sha256"]:
            raise ValueError("archive manifest absent from per-check source bindings")
    _sources(p, path, expected_sha); _archives(p)
    for record in p["inputs"].values():
        if set(record) != {"path", "sha256", "raw96_path", "raw96_sha256"}:
            raise ValueError("exact explicit DMD and RAW96 input record required")
        _file(record["path"], record["sha256"]); _file(record["raw96_path"], record["raw96_sha256"])
    _file(p["regions"]["path"], p["regions"]["sha256"])
    return p, blob


def _identity(plan):
    if old._start_ticks(plan["pid"]) != plan["process_start"]:
        raise RuntimeError("original process lifetime changed; no recovery")
    argv = [a.decode() for a in (Path("/proc")/str(plan["pid"])/"cmdline").read_bytes().split(b"\0") if a]
    positions = [i for i, value in enumerate(argv) if value == "--output"]
    if len(positions) != 1 or positions[0]+1 == len(argv) or Path(argv[positions[0]+1]).resolve() != Path(plan["session"]).resolve():
        raise RuntimeError("original process explicit session differs")


def _heartbeat(plan, clock):
    session = Path(plan["session"]); until = clock.monotonic()+2.5; rejected_publications = set()
    while True:
        h = old._read(session/"heartbeat.json"); age = clock.time_ns()-h["unix_ns"]
        complete = h.get("deployment_ui_revision") == 23 and "camera_active_readback" in h
        if h.get("pid") != plan["pid"] or h.get("hardware_fault"):
            raise RuntimeError("owner changed or hardware fault; no recovery")
        if any(h.get(k) is True for k in ("loop_active", "adaptation_active", "deployment_busy")) or h.get("spatial_calibration", {}).get("busy"):
            raise RuntimeError("competing operation; no recovery")
        if not h.get("live_camera"): raise RuntimeError("original camera is not running")
        fage = (h.get("latest_frame") or {}).get("age_seconds")
        if not isinstance(fage, (int, float)) or not math.isfinite(fage) or fage >= 1.5:
            raise RuntimeError("current camera frame not fresh")
        if (h.get("actual_roi") != plan["roi"] or h.get("optical_reference") != plan["optical_reference"]
                or not h.get("geometry_frozen")
                or any(h["manual_settings"].get(k) != v for k, v in plan["settings"].items() if k != "dmd")):
            raise RuntimeError("fixed ROI/model/SLM/reference geometry changed")
        if complete and (h["camera_active_readback"] != plan["expected_active"] or any(
                h.get(k) is not False for k in ("loop_active", "adaptation_active", "deployment_busy"))):
            raise RuntimeError("active/idle camera contract changed")
        _identity(plan)
        # The owner appends its v23 fields to the base heartbeat with the SAME
        # timestamp. Incompleteness alone must not poison that publication.
        # A truly future/negative-age publication, however, cannot be washed
        # clean by later clock progression or field changes under that stamp.
        if age < 0 or fage < 0: rejected_publications.add(h["unix_ns"])
        if complete and 0 <= age < 2_000_000_000 and fage >= 0 and h["unix_ns"] not in rejected_publications:
            return h
        if clock.monotonic() >= until:
            raise TimeoutError("no next complete fresh consistent heartbeat within2.5s: "+json.dumps(
                dict(header_age_ns=age, frame_age_seconds=fage, complete=complete, unix_ns=h["unix_ns"])))
        clock.sleep(.01)


def _arrays(plan, np):
    arrays = {}
    for key, record in plan["inputs"].items():
        a = np.load(io.BytesIO(_file(record["path"], record["sha256"]).read_bytes()), allow_pickle=False)
        raw = np.load(io.BytesIO(_file(record["raw96_path"], record["raw96_sha256"]).read_bytes()), allow_pickle=False)
        if (a.dtype != np.uint8 or a.shape != (600, 600) or not np.isin(a, (0, 255)).all()
                or raw.dtype != np.uint8 or raw.shape != (4, 1, 96, 96) or not np.isin(raw, (0, 255)).all()):
            raise ValueError("exact binary canonical600 and chronological RAW96 required")
        mosaic = np.block([[raw[0, 0], raw[1, 0]], [raw[2, 0], raw[3, 0]]])
        if not np.array_equal(a, np.pad(mosaic.repeat(3, 0).repeat(3, 1), 12)):
            raise ValueError("frozen RAW96/DMD packing differs")
        arrays[key] = a
    r = np.load(io.BytesIO(_file(plan["regions"]["path"], plan["regions"]["sha256"]).read_bytes()), allow_pickle=False)
    if (r.shape != (8, 554, 552) or not np.isfinite(r).all() or (r < 0).any() or (r > 1).any()
            or (r.sum(0) > 1.00001).any() or (r.sum((1, 2)) <= 0).any()):
        raise ValueError("eight fixed nonnegative matching native regions required")
    return arrays


def _coverage(plan):
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if isinstance(name, str) and name.endswith(".py"):
            p = Path(name).resolve()
            if p.is_relative_to(LEGACY.resolve()) and plan["source_sha256"].get(str(p)) != old._sha(p.read_bytes()):
                raise RuntimeError("imported old source absent from PLAN: " + str(p))


def _start_watchdog(plan_path, sha, root, plan, end, clock):
    return subprocess.Popen([sys.executable, "-m", "optical_lab.direction_capture", "--watchdog",
        "--plan", str(plan_path), "--sha256", sha, "--output", str(root), "--parent", str(os.getpid()),
        "--parent-start", old._start_ticks(os.getpid()), "--hard-deadline", str(time.time()+max(0, end-clock.monotonic()))])


def run_capture(plan, plan_path, plan_sha, output, client, *, clock=None):
    """One actual orchestration body, injectable only for no-hardware tests."""
    clock = clock or old.Clock(); root = Path(output); session = Path(plan["session"])
    root.mkdir(exist_ok=False); (root/"raw").mkdir()
    with (root/"PLAN.json").open("xb") as stream: stream.write(Path(plan_path).read_bytes())
    start = clock.monotonic(); end = start+360; work_end = end-15
    rows = [dict(**s, status="not_started", attempt_ids=[], records=[], valid_frame_count=0) for s in plan["states"]]
    result = dict(schema=RESULT_SCHEMA, profile="current_direction_v1", complete=False, error=None,
        states=rows, expected_states=17, expected_frames=136, frames_per_state=8, captured_valid_frames=0,
        no_SDK_no_new_windows=True, original_camera_stream_retained=True, camera_start_stop_calls=0,
        SLM_mode_or_file_changed=False, fixed_settings=plan["settings"], expected_active=plan["expected_active"],
        raw_snapshot_index="RAW_SNAPSHOT_INDEX.json", attempts_file="attempts.jsonl",
        BG_semantics="BOTH output panels zero/disarmed; lens SETTINGS retained; NOT physical dark",
        frame_pattern_association_verified=False, no_closed_loop=True, control_release_approved=False,
        archive_scope=old.ARCHIVE_SCOPE, snapshot_completion_rule=old.CORE_READY_RULE,
        cooperative_time_bound_seconds=360, cleanup_reserve_seconds=15, stopped_by_bound=False)
    baseline = set(session.glob("snapshot_*")); copied = {}; valid_folders = set()
    ui = watcher = lock = None; attempt_count = 0; last_id = last_tick = -1
    def event(name, **values): old._append(root/"events.jsonl", dict(event=name, unix_ns=clock.time_ns(), **values))
    def sources(): _sources(plan, Path(plan_path), plan_sha)
    def check():
        if clock.monotonic() >= work_end:
            result["stopped_by_bound"] = True; raise TimeoutError("stop to preserve15s cleanup; no replacement frame")
        if watcher is not None and watcher.poll() is not None: raise RuntimeError("independent watchdog exited")
        sources(); return _heartbeat(plan, clock)
    def operation(state_id, name, action):
        event("state_substep_intent", state_id=state_id, substep=name, public_action_limit=1)
        try: answer = action()
        except BaseException as exc:
            event("state_substep_failed", state_id=state_id, substep=name, error=repr(exc), no_action_retry=True); raise
        event("state_substep_ack", state_id=state_id, substep=name); return answer
    try:
        sources(); _archives(plan); lock = old._public_lock(); initial = _heartbeat(plan, clock)
        if initial["projection_armed"]: raise RuntimeError("root must publicly park before capture")
        if not 0 <= initial["latest_frame"]["frame_id"] < 43000:
            raise RuntimeError("wait for natural16bit FrameID wrap; never restart camera")
        before = client.inspect(session, parked=False)
        if before["pid"] != plan["pid"]: raise RuntimeError("inspected owner differs")
        old._new(root/"PREFLIGHT.json", before)
        old._new(root/"EXTERNAL_ACCESSIBILITY_COMPATIBILITY.json", old._install_accessibility_compatibility())
        _coverage(plan); ui = client.PilotUI(session, before, root, end)
        original_guard = ui.guard
        def guarded(*args, **kwargs):
            sources(); _heartbeat(plan, clock)
            if clock.monotonic() >= end: raise TimeoutError("360s bound")
            return original_guard(*args, **kwargs)
        ui.guard = guarded; ui.guard(black=True)
        old._new(root/"EXTERNAL_POINTER_COMPATIBILITY.json", client.install())
        arrays = _arrays(plan, client.np)
        watcher = _start_watchdog(plan_path, plan_sha, root, plan, end, clock)
        client.find_tab(ui)
        old._new(root/"ORIGINAL_SETTINGS.json", dict(settings=initial["manual_settings"],
            restored_armed=False, final_DMD_file_mode_may_remain=True, SLM_lens_settings_never_changed=True))
        loaded_key = None
        for spec, row in zip(plan["states"], rows):
            check()
            if work_end-clock.monotonic() < 20:
                result["stopped_by_bound"] = True; raise TimeoutError("less than20s before next full state")
            row["status"] = "partial"; event("state_intent", state_id=row["id"], input_key=spec["input_key"])
            operation(row["id"], "arm_false", lambda: ui.arm(False)); ui.guard(black=True)
            key = spec["input_key"]
            if key is not None and key != loaded_key:
                check(); ack = operation(row["id"], "load_DMD_"+key,
                    lambda: ui.load_live("dmd", Path(plan["inputs"][key]["path"])))
                if Path(ack["path"]).resolve() != Path(plan["inputs"][key]["path"]).resolve() or ack["sha256"] != plan["inputs"][key]["sha256"]:
                    raise RuntimeError("DMD file acknowledgement mismatch")
                ui.wait(lambda h: h["manual_settings"]["dmd"] == "file" and h["manual_settings"]["slm"] == "lens")
                row["file_load_acknowledgement"] = ack; loaded_key = key
            elif key is not None:
                row["file_load_skipped"] = dict(reason="same exact loaded key in this capture", asset=plan["inputs"][key])
            dmd = arrays[key] if key is not None else client.np.zeros((600, 600), client.np.uint8)
            setting = dict(plan["settings"], dmd="file")
            expected, cd, cs = client.expected_pixels(setting, dmd, None, spec["illuminated"], "lens")
            operation(row["id"], "arm_"+str(spec["illuminated"]), lambda: ui.arm(spec["illuminated"]))
            until = min(work_end, clock.monotonic()+5)
            while True:
                check(); h = ui.guard(black=not spec["illuminated"], expected=expected if spec["illuminated"] else None, transition=True)
                comparisons = {r: ui.pixels.compare(tuple(before["windows"][r]["rect"]), expected[r]) for r in ("dmd", "slm")}
                if h["last_verified_revision"] == h["display_revision"] and all(v["matches"] for v in comparisons.values()): break
                if clock.monotonic() >= until: raise TimeoutError("actual panel/revision not acknowledged")
                clock.sleep(.05)
            revision = h["display_revision"]; verified = clock.time_ns()
            row.update(display_revision=revision, verified_unix_ns=verified, display_readback=comparisons,
                expected_panel={r: dict(shape_yx=list(expected[r].shape), dtype="uint8_grayscale_equal_RGB",
                    pixel_sha256=old._sha(expected[r].tobytes(order="C"))) for r in expected})
            event("state_committed", state_id=row["id"], display_revision=revision, expected_panel=row["expected_panel"])
            settle = clock.monotonic()+1.8
            while clock.monotonic() < settle:
                check(); ui.guard(black=not spec["illuminated"], expected=expected if spec["illuminated"] else None); clock.sleep(.08)
            for repeat in range(8):
                check(); ui.guard(black=not spec["illuminated"], expected=expected if spec["illuminated"] else None)
                attempt_count += 1
                a = dict(id="A"+str(attempt_count).zfill(4), state_id=row["id"], input_key=key,
                    repeat_index=repeat, status="failed", error=None, raw=None, metadata=None, frame=None,
                    original_snapshot_folder=None, frame_pattern_association_verified=False,
                    snapshot_requested_unix_ns=clock.time_ns())
                row["attempt_ids"].append(a["id"]); event("frame_attempt_intent", id=a["id"], state_id=row["id"], repeat_index=repeat)
                old_folders = set(session.glob("snapshot_*"))
                try:
                    operation(row["id"], "snapshot_"+str(repeat), lambda: ui.button("snapshot"))
                    until = min(work_end, clock.monotonic()+4)
                    while True:
                        check(); ui.guard(black=not spec["illuminated"], expected=expected if spec["illuminated"] else None)
                        fresh = set(session.glob("snapshot_*"))-old_folders
                        if len(fresh) > 1: raise RuntimeError("ambiguous snapshot; no frame substitution")
                        if fresh:
                            folder = next(iter(fresh))
                            if all((folder/n).is_file() for n in old.NUMERIC_CORE):
                                try: old._read(folder/"metadata.json")
                                except (OSError, ValueError): pass
                                else: break
                        if clock.monotonic() >= until: raise TimeoutError("five-core snapshot not complete within4s")
                        clock.sleep(.04)
                    archive = old._copy_snapshot(folder, root, copied)
                    a.update(original_snapshot_folder=str(folder), raw=archive["files"]["CCD_raw.npy"],
                             metadata=archive["files"]["metadata.json"])
                    metadata = old._read(root/a["metadata"]["path"])
                    raw = client.np.load(io.BytesIO((root/a["raw"]["path"]).read_bytes()), allow_pickle=False)
                    f = metadata["frame"]; a.update(frame=f, host_get_started_unix_ns=f.get("host_get_started_unix_ns"),
                        camera_actual=metadata["camera_active"], display_revision=metadata["display_revision"])
                    if (raw.dtype != client.np.uint8 or raw.shape != (554, 552) or raw.max() >= 240
                            or metadata["camera_active"] != plan["expected_active"] or metadata["display_revision"] != revision
                            or any(type(f.get(k)) is not int for k in ("FrameID", "TimeStamp", "host_get_started_unix_ns"))
                            or f["host_get_started_unix_ns"] < verified+1_500_000_000
                            or not a["snapshot_requested_unix_ns"]-1_500_000_000 <= f["host_get_started_unix_ns"] <= clock.time_ns()
                            or f["FrameID"] <= last_id or f["TimeStamp"] <= last_tick or f.get("IsIncomplete") is True):
                        raise RuntimeError("native raw saturation/freshness/identity/revision/order gate failed")
                    for role, request in (("dmd", cd), ("slm", cs)):
                        with client.Image.open(root/archive["files"][role+"_canonical_requested.png"]["path"]) as im:
                            if not client.np.array_equal(client.np.asarray(im), request):
                                raise RuntimeError("saved canonical request differs: "+role)
                    last_id, last_tick = f["FrameID"], f["TimeStamp"]
                    a["status"] = "valid"; row["valid_frame_count"] += 1; result["captured_valid_frames"] += 1
                    valid_folders.add(str(folder)); row["records"].append(dict(a))
                except BaseException as exc:
                    a["error"] = repr(exc); raise
                finally: old._append(root/"attempts.jsonl", a)
            row["status"] = "complete"
            operation(row["id"], "state_end_arm_false", lambda: ui.arm(False)); ui.guard(black=True)
            old._new(root/(row["id"]+".json"), row)
        result["all_states_complete"] = True
    except BaseException as exc:
        result["error"] = repr(exc)
        if isinstance(exc, TimeoutError) and clock.monotonic() >= work_end: result["stopped_by_bound"] = True
    finally:
        try:
            if ui is not None:
                ui.button("allzero"); until = min(end, clock.monotonic()+5)
                while True:
                    h = ui.state(allow_fault=True)
                    if not h["projection_armed"] and all(ui.pixels.inspect(tuple(ui.before["windows"][r]["rect"]))["all_RGB_zero"] for r in ("dmd", "slm")):
                        result.update(actual_outputs_zero=True, camera_left_running=h["live_camera"]); break
                    if clock.monotonic() >= until: raise TimeoutError("cleanup actual allzero acknowledgement missing")
                    clock.sleep(.05)
        except BaseException as exc: result["cleanup_error"] = repr(exc)
        archive_errors = []
        for f in sorted((set(session.glob("snapshot_*"))-baseline)|{Path(p) for p in copied}):
            try: old._copy_snapshot(f, root, copied)
            except BaseException as exc: archive_errors.append(dict(folder=str(f), error=repr(exc)))
        result["archive_errors"] = archive_errors
        result["unassociated_snapshots"] = sorted(set(copied)-valid_folders)
        old._new(root/"RAW_SNAPSHOT_INDEX.json", dict(snapshots=list(copied.values()), errors=archive_errors,
            archive_scope=old.ARCHIVE_SCOPE, numeric_core_files=list(old.NUMERIC_CORE),
            snapshot_completion_rule=old.CORE_READY_RULE, invalid_or_unassociated_folders_are_NOT_valid_attempts=True))
        old._new(root/"FINISHED.json", dict(unix_ns=clock.time_ns(), actual_outputs_zero=result.get("actual_outputs_zero", False)))
        if watcher is not None:
            try:
                watcher.wait(timeout=max(.1, min(6, end-clock.monotonic())))
                w = old._read(root/"WATCHDOG_RESULT.json"); result["watchdog_result"] = w
                result["watchdog_verified"] = w.get("actual_all_zero") is True and not w.get("intervened") and not w.get("error")
            except BaseException as exc: result["watchdog_error"] = repr(exc)
        try:
            sources(); _archives(plan)
            for r in plan["inputs"].values(): _file(r["path"], r["sha256"]); _file(r["raw96_path"], r["raw96_sha256"])
            _file(plan["regions"]["path"], plan["regions"]["sha256"]); result["frozen_inputs_unchanged"] = True
        except BaseException as exc: result["integrity_error"] = repr(exc)
        for name, resource in (("pixels", None if ui is None else ui.pixels), ("lock", lock)):
            if resource is not None:
                try: resource.close()
                except BaseException as exc: result.setdefault("close_errors", []).append(dict(resource=name, error=repr(exc)))
        result["complete"] = bool(result.get("all_states_complete") and result["captured_valid_frames"] == 136
            and result.get("actual_outputs_zero") and result.get("camera_left_running") and result.get("watchdog_verified")
            and result.get("frozen_inputs_unchanged") and not result["unassociated_snapshots"]
            and not any(result.get(k) for k in ("error", "cleanup_error", "archive_errors", "integrity_error", "close_errors", "stopped_by_bound")))
        result["wall_seconds"] = clock.monotonic()-start
        old._new(root/"RESULT.json", result)
    return root/"RESULT.json"


def capture(plan_path, expected_sha, output):
    plan, _ = validate_plan(plan_path, expected_sha)
    out = Path(output).resolve()
    if out.parent != (LAB/"experiments").resolve(): raise ValueError("NEW direct lab experiments child required")
    if out.exists(): raise FileExistsError(out)
    client = old._load_clients(LEGACY); _coverage(plan)
    return run_capture(plan, Path(plan_path).resolve(), expected_sha, out, client)


def _watchdog(args):
    plan, _ = validate_plan(args.plan, args.sha256); root = Path(args.output)
    client = old._load_clients(LEGACY); _coverage(plan)
    before = old._read(root/"PREFLIGHT.json")
    result = dict(actual_all_zero=False, intervened=False, error=None, lease_limit_seconds=20)
    ui = None
    try:
        ui = client.PilotUI(Path(plan["session"]), before, root, time.monotonic()+360, renew=False)
        while True:
            h = ui.state(allow_fault=True); _identity(plan)
            if (root/"FINISHED.json").exists():
                if h["projection_armed"]: ui.button("allzero"); result["intervened"] = True
                break
            try: alive = old._start_ticks(args.parent) == args.parent_start
            except FileNotFoundError: alive = False
            lease = old._read(root/"LEASE.json")
            if not alive or not 0 <= time.time_ns()-lease["unix_ns"] <= 20_000_000_000 or time.time() >= args.hard_deadline:
                ui.button("allzero"); result["intervened"] = True; break
            time.sleep(.2)
        until = time.monotonic()+4
        while time.monotonic() < until:
            h = ui.state(allow_fault=True)
            if not h["projection_armed"] and all(ui.pixels.inspect(tuple(before["windows"][r]["rect"]))["all_RGB_zero"] for r in ("dmd", "slm")):
                result["actual_all_zero"] = True; break
            time.sleep(.1)
        if not result["actual_all_zero"]: raise RuntimeError("watchdog actual allzero not verified")
    except BaseException as exc: result["error"] = repr(exc)
    finally:
        if ui is not None:
            try: ui.pixels.close()
            except BaseException as exc: result["close_error"] = repr(exc)
        old._new(root/"WATCHDOG_RESULT.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--watchdog", action="store_true", required=True)
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--sha256", required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--parent", type=int, required=True)
    p.add_argument("--parent-start", required=True); p.add_argument("--hard-deadline", type=float, required=True)
    _watchdog(p.parse_args())
