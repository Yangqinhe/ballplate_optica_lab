"""Explicit bounded pilot transport through ONE already-running legacy GUI.

Import and legacy-pilot validation are standard-library only. Explicit new-case
validation lazily uses the offline NumPy asset verifier, never the old GUI.
Execution lazily uses
the old public accessibility/snapshot clients; it never opens an SDK, creates
an output window, changes exposure/ROI, or starts/stops the existing camera.
This is not an optical safety interlock or a frame/exposure association proof.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

from .protocol import bracketed_states, observability_states

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "optical-lab-existing-owner-pilot-plan-1"
RESULT_SCHEMA = "optical-lab-existing-owner-pilot-capture-1"
NUMERIC_CORE = ("CCD_raw.npy", "CCD_raw.png", "metadata.json",
                "dmd_canonical_requested.png", "slm_canonical_requested.png")
ARCHIVE_SCOPE = "five_synchronous_numeric_core_files_only_NOT_complete_snapshot"
CORE_READY_RULE = "all_five_core_files_exist_and_metadata_JSON_is_parseable"
AUXILIARY_EXCLUSION_REASON = ("display/comparison files are written after metadata, may be incomplete at copy time, "
    "and may describe a different camera frame; not used for pilot numeric evidence")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}\Z")
_DEAD_PATH = r"(?:'/org/a11y/atspi/accessible/[0-9]+'|\"/org/a11y/atspi/accessible/[0-9]+\"|/org/a11y/atspi/accessible/[0-9]+)"
_DEAD_MESSAGE = re.compile(r"(?:No such object path "+_DEAD_PATH+r"|Object(?: path)? "+_DEAD_PATH+r" does not exist)\.?\Z")
_CLIENTS = ("tools/static_learned_ccd_v28.py", "tools/inspect_recovery_ui_v20.py",
            "tools/static_singlebit_v109.py", "workbench/twin/capture.py",
            "workbench/unified/capture_static.py", "workbench/unified/pointer_control.py")


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _known_dead_accessible(error):
    """Only known read-side AT-SPI dead numeric-object errors, not all GErrors."""
    return (getattr(error,"domain",None)=="atspi_error" and type(getattr(error,"code",None)) is int
            and error.code==1 and isinstance(getattr(error,"message",None),str)
            and _DEAD_MESSAGE.fullmatch(error.message) is not None)


def _safe_accessibility_nodes(root, observations):
    """Fresh read-only discovery; operation/action exceptions are NOT caught."""
    pending=[root]
    while pending:
        node=pending.pop()
        try:
            node.clear_cache()
            node.get_role_name();node.get_name()
            children=[node.get_child_at_index(i) for i in range(node.get_child_count())]
        except Exception as exc:
            if not _known_dead_accessible(exc):raise
            observations["skipped_known_dead_reads"]+=1
            if len(observations["known_dead_messages"])<64:
                observations["known_dead_messages"].append(exc.message)
            continue
        observations["live_nodes_read"]+=1
        yield node
        pending.extend(reversed([child for child in children if child is not None]))


def _install_accessibility_compatibility(modules=None):
    """Explicit client-process namespace bindings only; NEVER target injection.

    Frozen helpers retain target PID/name/role/uniqueness/state/rectangle and
    single-action checks. Only their read-side discovery is made defunct-aware.
    """
    if modules is None:
        import optical_ui_v19
        import trigger_ui_v19
        from night_opt_20261020 import manual_io_v17
        modules=SimpleNamespace(optical=optical_ui_v19,trigger=trigger_ui_v19,manual=manual_io_v17)
    observations=dict(skipped_known_dead_reads=0,live_nodes_read=0,known_dead_messages=[])
    def nodes(root):return _safe_accessibility_nodes(root,observations)
    modules.optical.nodes=nodes;modules.trigger.nodes=nodes
    modules.manual.defunct=_known_dead_accessible
    return dict(external_client_only=True,target_process_injection=False,old_source_files_modified=False,
        read_only_namespaces=["optical_ui_v19.nodes","trigger_ui_v19.nodes","night_opt_20261020.manual_io_v17.defunct"],
        error_policy="only domain atspi_error/code1 plus exact known dead numeric-object path message",
        clear_cache_before_each_discovery_node=True,action_retries=0,
        target_identity_uniqueness_role_name_state_rectangle_guards_unchanged=True,observations=observations)


def dependency_sources(legacy_root):
    """AST-only conservative local-import closure; NEVER import old modules.

    Conditional local imports are included.  Actual loaded old .py modules
    are checked again before the first public park action, so a dynamic or
    unresolved dependency cannot silently proceed.
    """
    legacy = Path(legacy_root).resolve()
    handoff = legacy.parent / "ballplate_minframes_optical_ppo/SDON_Linux_Handoff_20261015/src"
    roots = (legacy, legacy/"tools", legacy/"live_manual_v17", handoff)
    todo = [legacy/p for p in _CLIENTS]
    seen = {}
    def resolve(parts):
        result = []
        for base in roots:
            p = base.joinpath(*parts)
            for candidate in (p.with_suffix(".py"), p/"__init__.py"):
                if candidate.is_file():
                    result.append(candidate.resolve())
                    for parent in candidate.parents:
                        if parent == base:
                            break
                        init = parent/"__init__.py"
                        if init.is_file():
                            result.append(init.resolve())
                    return result
        return result
    while todo:
        path = todo.pop()
        if str(path) in seen:
            continue
        # Initial entry files also import their package initializers. They
        # did not pass through resolve(), so bind these explicitly too.
        for parent in path.parents:
            if parent in roots:
                break
            initializer=parent/"__init__.py"
            if initializer.is_file() and initializer.resolve()!=path:
                todo.append(initializer.resolve())
        blob = path.read_bytes()
        seen[str(path)] = _sha(blob)
        tree = ast.parse(blob, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    todo.extend(resolve(alias.name.split(".")))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level-1):
                        base = base.parent
                    module = base.joinpath(*(node.module or "").split("."))
                    candidates = [module.with_suffix(".py"), module/"__init__.py"]
                    candidates += [module/(a.name+".py") for a in node.names if a.name != "*"]
                    todo.extend(p.resolve() for p in candidates if p.is_file())
                else:
                    parts = (node.module or "").split(".")
                    todo.extend(resolve(parts))
                    for alias in node.names:
                        if alias.name != "*":
                            todo.extend(resolve(parts+[alias.name]))
    return dict(sorted(seen.items()))


def _asset(record):
    path = Path(record["path"])
    if not path.is_absolute() or not isinstance(record["sha256"], str) or not _SHA.fullmatch(record["sha256"]):
        raise ValueError("explicit absolute asset path and SHA required")
    if _sha(path.read_bytes()) != record["sha256"]:
        raise ValueError("frozen asset changed: " + str(path))
    return path


def _fixed_states(frames_per_state=16):
    return bracketed_states(frames_per_state)


def _state_repetitions(plan):
    repeats=plan.get("frames_per_state",16)
    if plan.get("profile")=="observability_v1":
        expected=observability_states(plan.get("axis_family"))
        if (type(repeats) is not int or repeats!=8
                or json.dumps(plan["states"],sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False)
                or type(plan.get("total_planned_frames")) is not int or plan["total_planned_frames"]!=88):
            raise ValueError("exact observability11 INPUT/BG states, eight repeats and88 frames required")
        _observability_inputs(plan)
        return repeats
    expected=bracketed_states(repeats)
    if repeats != 16 and plan.get("profile") != "static_case_v1":
        raise ValueError("original pilot is fixed16; eight repeats require explicit new-case profile")
    if (plan["states"] != expected or any(type(s.get("repeat")) is not int for s in plan["states"])
            or type(plan.get("total_planned_frames")) is not int
            or plan["total_planned_frames"] != 17*repeats):
        raise ValueError("exact frozen BG / AB-BA-AB bracketed 17-state sequence and total required")
    return repeats


def _observability_inputs(plan):
    """Structural input/mask identity only; full bank semantics are external."""
    expected={s["input_key"] for s in observability_states(plan.get("axis_family"))}
    records=plan.get("inputs")
    if not isinstance(records,dict) or set(records)!=expected:
        raise ValueError("all five exact observability input identities required")
    canonical=lambda value:json.dumps(value,sort_keys=True,allow_nan=False)
    if canonical(plan["input"])!=canonical(records["REF"]):
        raise ValueError("backward-compatible input must equal the exact REF record")
    fields={"path","sha256","raw96","id","selection_rule","history_ages_s"}
    for record in records.values():
        if (not isinstance(record,dict) or set(record)!=fields or not isinstance(record["raw96"],dict)
                or set(record["raw96"])!={"path","sha256"}
                or record["history_ages_s"]!=[.18,.12,.06,0.]):
            raise ValueError("complete chronological DMD/RAW96 input records required")
    if (plan.get("mask_key") not in ("learned","lens","flat")
            or set(plan["masks"])!={"Z","A","B"}
            or any(canonical(plan["masks"][k])!=canonical(plan["masks"]["Z"]) for k in ("A","B"))):
        raise ValueError("one fixed observability mask must occupy all compatibility slots")
    if any(k in plan for k in ("pilot_manifest_sha256","case_asset_manifest","case_manifest_sha256","frozen_predictions")):
        raise ValueError("observability profile cannot reuse historical pilot/case/prediction semantics")
    return records


def _verify_observability_plan(plan):
    _observability_inputs(plan)
    helper=Path(__file__).with_name("observability.py")
    if plan["source_sha256"].get(str(helper))!=_sha(helper.read_bytes()):
        raise ValueError("observability bank verifier source must be frozen before capture")
    from .observability import verify_bank_and_plan
    checked=verify_bank_and_plan(plan)
    if not isinstance(checked,dict) or checked.get("verified") is not True:
        raise ValueError("complete observability bank/plan verification required")
    return checked


def _plan_assets(plan):
    inputs=list(_observability_inputs(plan).values()) if plan.get("profile")=="observability_v1" else [plan["input"]]
    records=inputs+list(plan["masks"].values())+[plan["hardware_contract"]]
    if plan.get("profile")=="observability_v1":records += [r["raw96"] for r in inputs]
    return records


def _manifest_checks(plan):
    if plan.get("profile")=="observability_v1":
        return _verify_observability_plan(plan)
    parents = {Path(m["path"]).resolve().parent for m in plan["masks"].values()}
    if plan.get("profile") == "static_case_v1":
        identity=plan["case_asset_manifest"]
        if ("pilot_manifest_sha256" in plan or identity["sha256"]!=plan["case_manifest_sha256"]
                or not Path(identity["path"]).is_absolute() or len(parents)!=1
                or next(iter(parents))!=Path(identity["path"]).resolve()):
            raise ValueError("explicit case asset identity cannot masquerade as the old pilot")
        expected=identity["sha256"]
    else:
        if plan.get("profile") is not None or any(k in plan for k in ("case_asset_manifest","case_manifest_sha256","case_spec","case_id","role","design","fixed_input","probe_spec")):
            raise ValueError("new case requires its explicit static_case_v1 profile")
        expected=plan["pilot_manifest_sha256"]
    if len(parents) != 1 or _sha((next(iter(parents))/"MANIFEST.json").read_bytes()) != expected:
        raise ValueError("frozen asset manifest mismatch")
    predictions = plan["frozen_predictions"]
    if _sha((Path(predictions["path"])/"MANIFEST.json").read_bytes()) != predictions["manifest_sha256"]:
        raise ValueError("frozen predictions manifest changed")


def _verify_case_plan(plan):
    """Explicit new profile only; lazy NumPy semantic check, never old imports."""
    from .study import verify_case_assets
    identity=plan["case_asset_manifest"]
    checked=verify_case_assets(identity["path"],expected_manifest_sha256=identity["sha256"])
    if checked.get("verified") is not True or checked["manifest_sha256"]!=plan["case_manifest_sha256"]:
        raise ValueError("new case assets did not verify against the external identity")
    manifest=checked["manifest"];design=manifest["design"];fixed=manifest["fixed_input"]
    expected={"case_id":design["case_id"],"role":design["role"],"design":design,
              "case_spec":manifest["probe_spec"],"probe_spec":manifest["probe_spec"],"fixed_input":fixed}
    canonical=lambda value:json.dumps(value,sort_keys=True,allow_nan=False)
    if manifest.get("schema")!="optical-lab-static-case-assets-1" or any(
            canonical(plan.get(key))!=canonical(value) for key,value in expected.items()):
        raise ValueError("case plan design/role/input/probe association differs")
    expected_input=dict(path=fixed["dmd_path"],sha256=fixed["dmd_sha256"],id=fixed["id"],
        raw96=dict(path=fixed["raw_path"],sha256=fixed["raw_sha256"]),
        selection_rule=fixed["selection_rule"],history_ages_s=fixed["history_ages_s"])
    if canonical(plan["input"])!=canonical(expected_input):
        raise ValueError("case DMD/RAW input is not the case's frozen input")
    root=Path(identity["path"]).resolve()
    for name in ("Z","A","B"):
        filename=manifest["assets"][name]["npy"]
        if plan["masks"][name]!={"path":str(root/filename),"sha256":manifest["file_sha256"][filename]}:
            raise ValueError("case mask is not associated with its Z/A/B role")
    required={str(root/"MANIFEST.json"):identity["sha256"],
        **{str(root/name):digest for name,digest in manifest["file_sha256"].items()},
        design["path"]:design["sha256"],str(Path(__file__).with_name("study.py")):_sha(Path(__file__).with_name("study.py").read_bytes())}
    if any(plan["source_sha256"].get(path)!=digest for path,digest in required.items()):
        raise ValueError("case source/design/complete artifact ledger is not frozen")
    _asset(design)
    return checked


def validate_plan(plan_path, expected_plan_sha256):
    """File identities only; no old code import, X11, accessibility, or SDK."""
    if not isinstance(expected_plan_sha256, str) or not _SHA.fullmatch(expected_plan_sha256):
        raise ValueError("external exact PLAN SHA256 required")
    path = Path(plan_path).resolve(strict=True)
    blob = path.read_bytes()
    if _sha(blob) != expected_plan_sha256:
        raise ValueError("PLAN SHA mismatch")
    plan = json.loads(blob)
    if (plan.get("schema") != SCHEMA or plan.get("expected_pid") != 2462026
            or plan.get("maximum_seconds") != 300 or plan.get("cleanup_reserve_seconds") != 12
            or plan.get("minimum_state_start_seconds") != 20 or plan.get("inter_block_wait_seconds") != 0
            or plan.get("software_settle_wait_seconds") != 1.8
            or plan.get("minimum_postcommit_frame_start_seconds") != 1.5
            or plan.get("no_training") is not True or plan.get("no_closed_loop") is not True
            or plan.get("real_actuators") is not False):
        raise ValueError("fixed original-owner offline-control-free 300s profile required")
    repeats=_state_repetitions(plan)
    observable=plan.get("profile")=="observability_v1"
    expected_count=11 if observable else 17
    if len(plan["states"]) != expected_count or len({s["id"] for s in plan["states"]}) != expected_count:
        raise ValueError("all predeclared states required")
    for state in plan["states"]:
        if (not _ID.fullmatch(state["id"]) or state["kind"] not in (("BG","INPUT") if observable else ("BG", "Z", "A", "B"))
                or type(state["repeat"]) is not int or state["repeat"] != repeats
                or (state["time_block"] is not None and type(state["time_block"]) is not int)):
            raise ValueError("unique safe state / fixed declared repeats required")
    if set(plan["masks"]) != {"Z", "A", "B"}:
        raise ValueError("exact frozen Z/A/B assets required")
    for record in _plan_assets(plan):
        _asset(record)
    _manifest_checks(plan)
    if plan.get("profile")=="static_case_v1":
        _verify_case_plan(plan)
    if set(plan["camera_expected"]) != {"active", "roi"}:
        raise ValueError("complete actual active-camera readback and ROI required")
    active = plan["camera_expected"]["active"]
    if any(active.get(k) != v for k, v in {"DeviceSerialNumber":"15328133", "PixelFormat":"Mono8",
            "TriggerMode":"On", "TriggerSelector":"FrameStart", "TriggerSource":"Line0",
            "TriggerActivation":"FallingEdge"}.items()):
        raise ValueError("original Mono8 Line0 camera identity required")
    if plan["camera_expected"]["roi"] != {"OffsetX":888,"OffsetY":416,"Width":552,"Height":554}:
        raise ValueError("native current ROI cannot change")
    if set(plan["manual_geometry_expected"]) != {"x","y","full_panel","square","focal_mm","period"}:
        raise ValueError("explicit unchanged six-field manual geometry required")
    if plan["manual_geometry_expected"]["full_panel"] is not False:
        raise ValueError("canonical600 window mode required; never generate a full-panel frame")
    if not plan["authorization"].get("confirmed_utc") or not plan["authorization"].get("statement"):
        raise ValueError("explicit current user confirmation record required")
    sources = plan["source_sha256"]
    own = str(Path(__file__).resolve())
    if own not in sources:
        raise ValueError("adapter source must be frozen before capture")
    shared=str(Path(__file__).with_name("protocol.py"))
    if sources.get(shared)!=_sha(Path(shared).read_bytes()):
        raise ValueError("shared static sequence source must be frozen before capture")
    legacy = Path(plan["legacy_root"]).resolve()
    runtime_path = Path(plan["session"])/"runtime_sources.json"
    runtime = _read(runtime_path)
    if runtime["pid"] != plan["expected_pid"] or (len(runtime["sources"]), len(runtime["source_configs"])) != (78, 2):
        raise ValueError("original78+2 owner source identity required")
    required = {**runtime["sources"], **runtime["source_configs"], **dependency_sources(legacy)}
    for p in (runtime_path, legacy/"workbench/current.json", PROJECT/"references/hardware/manifest.json"):
        if str(p.resolve()) not in sources:
            raise ValueError("missing frozen identity file: " + str(p))
    for p, sha in required.items():
        if sources.get(p) != sha:
            raise ValueError("missing/changed source closure: " + p)
    for p, sha in sources.items():
        _asset({"path":p,"sha256":sha})
    # Original reference verifier hashes only files, and never executes old code.
    from .references import verify
    if not verify("hardware", legacy_root=legacy)["verified"]:
        raise ValueError("original hardware reference identities changed")
    return plan, blob


def _load_clients(legacy):
    """Called ONLY after explicit execution and complete frozen-file preflight."""
    sys.path[:0] = [str(legacy/"tools"), str(legacy), str(legacy/"live_manual_v17")]
    from static_learned_ccd_v28 import inspect, expected_pixels, Input
    from workbench.twin.capture import PilotUI
    from workbench.unified.pointer_control import install
    from workbench.unified.capture_static import cached_assets
    from static_singlebit_v109 import find_tab
    from load_current_adaptation_bundle_v134 import one_control, press
    from select_verified_mode_v17 import window, inside, owned_popups
    from display_linux import read_roles, same_signals
    import numpy as np
    from PIL import Image
    return SimpleNamespace(**locals())


def _loaded_source_coverage(plan):
    workspace = Path(plan["legacy_root"]).resolve().parent
    rows = {}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not isinstance(name, str) or not name.endswith(".py"):
            continue
        path = Path(name).resolve()
        if path.is_relative_to(workspace) and not path.is_relative_to(PROJECT):
            actual = _sha(path.read_bytes())
            if plan["source_sha256"].get(str(path)) != actual:
                raise ValueError("imported old source absent from frozen PLAN: " + str(path))
            rows[str(path)] = actual
    return rows


class Clock:
    monotonic = staticmethod(time.monotonic)
    time_ns = staticmethod(time.time_ns)
    sleep = staticmethod(time.sleep)


def _start_ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def _heartbeat(session, plan, clock, start_ticks=None):
    # The frozen owner publishes about every two seconds, with a short base
    # -> v23-extension interval. Only wait for the next complete publication;
    # never accept a >=2s header, retry a frame/button, or wait out a fault.
    until = clock.monotonic()+2.5
    while True:
        h = _read(session/"heartbeat.json")
        age = clock.time_ns()-h["unix_ns"]
        complete = h.get("deployment_ui_revision")==23 and "camera_active_readback" in h
        diagnostics = dict(reason=None,heartbeat_age_ns=age,maximum_accepted_age_ns=2_000_000_000,
            deployment_ui_revision=h.get("deployment_ui_revision"),schema=h.get("schema"),
            complete_v23_publication=complete,pid=h.get("pid"),hardware_fault=h.get("hardware_fault"),
            frame_age_seconds=(h.get("latest_frame") or {}).get("age_seconds"))
        reason = None
        if h.get("pid")!=plan["expected_pid"]:reason="owner_pid_changed"
        elif age<0:reason="future_heartbeat"
        elif h.get("hardware_fault"):reason="hardware_fault"
        elif any(h.get(k) is True for k in ("loop_active","deployment_busy","adaptation_active")):reason="competing_operation"
        elif h.get("spatial_calibration",{}).get("busy"):reason="spatial_operation"
        elif not h.get("live_camera"):reason="camera_not_running"
        elif not h.get("latest_frame"):reason="latest_frame_missing"
        elif not -float("inf")<h["latest_frame"]["age_seconds"]<1:reason="reported_frame_not_fresh"
        elif not h.get("geometry_frozen"):reason="geometry_not_frozen"
        elif h.get("actual_roi")!=plan["camera_expected"]["roi"]:reason="ROI_changed"
        elif any(h.get("manual_settings",{}).get(k)!=v for k,v in plan["manual_geometry_expected"].items()):reason="manual_geometry_changed"
        elif "camera_active_readback" in h and h["camera_active_readback"]!=plan["camera_expected"]["active"]:reason="active_camera_changed"
        elif complete and any(h.get(k) is not False for k in ("loop_active","deployment_busy","adaptation_active")):reason="idle_flags_missing_or_invalid"
        if reason is None and start_ticks is not None and _start_ticks(h["pid"])!=start_ticks:reason="owner_process_lifetime_changed"
        if reason:
            diagnostics["reason"]=reason
            raise RuntimeError("fresh healthy idle original v23 stream required; no recovery: "+json.dumps(diagnostics,sort_keys=True))
        # The frozen tick samples `now` before reading the camera mailbox.
        # A newer worker packet can therefore make that publication's derived
        # frame age negative. Never clamp/accept it: await another complete
        # publication under this same read-only 2.5s bound. All other failures
        # above remain immediate (including a future heartbeat timestamp).
        negative_frame_age=h["latest_frame"]["age_seconds"]<0
        if complete and age<2_000_000_000 and not negative_frame_age:
            return h
        diagnostics["reason"]=("negative_frame_age_inconsistent_publication" if negative_frame_age else
            "incomplete_v23_publication" if not complete else "stale_publication_waiting_for_new_header")
        if clock.monotonic()>=until:
            raise RuntimeError("read-only heartbeat publication wait timed out; no recovery: "+json.dumps(diagnostics,sort_keys=True))
        clock.sleep(.01)


def _prepark_identity(client, session, plan, clock):
    h = _heartbeat(session, plan, clock)
    if not 0 <= h["latest_frame"]["frame_id"] < 45000:
        raise RuntimeError("wait for natural raw FrameID wrap; never restart camera")
    roles = client.read_roles()
    contract = _read(plan["hardware_contract"]["path"])
    client.same_signals(contract["display_roles"], roles)
    tree = subprocess.check_output(["xwininfo","-root","-tree"], text=True, timeout=3)
    candidates = []
    for line in tree.splitlines():
        if '"ballplate-realtime-manual"' not in line and '"实时光路对齐 v19 ' not in line:
            continue
        hit = re.search(r"0x[0-9a-fA-F]+", line)
        if hit:
            row = client.window(hit[0])
            if row["pid"] == h["pid"] and row["viewable"]:
                row["control"] = '"实时光路对齐 v19 ' in line
                candidates.append(row)
    main = [x for x in candidates if x["control"]]
    if len(main) != 1 or not client.inside(main[0]["rect"], [1080,600,2560,1440]):
        raise RuntimeError("unique visible PID-owned ordinary main window required")
    for role, video in roles.items():
        rectangle = video["desktop_position"] + video["desktop_size"]
        if len([x for x in candidates if not x["control"] and x["rect"] == rectangle]) != 1:
            raise RuntimeError("original output window identity differs: " + role)
    if client.owned_popups(h["pid"], main[0]["wid"]):
        raise RuntimeError("native popup/competing action already active")
    return h, main[0], _start_ticks(h["pid"])


def _source_checks(plan, plan_path, sha):
    if _sha(plan_path.read_bytes()) != sha:
        raise RuntimeError("PLAN changed during capture")
    for p, expected in plan["source_sha256"].items():
        if _sha(Path(p).read_bytes()) != expected:
            raise RuntimeError("frozen source changed: " + p)


def _copy_snapshot(folder, output, index):
    """Copy only five numeric-core files; retain invalid/partial core evidence.

    This is intentionally NOT an archive of the entire legacy snapshot.
    Its later display/comparison PNGs are neither read nor scored here.
    """
    if folder.is_symlink() or not folder.is_dir():
        raise RuntimeError("unexpected snapshot link/type")
    row = index.setdefault(str(folder), {"original_folder":str(folder),
        "archived_folder":"raw/"+folder.name, "files":{}, "association_verified":False,
        "archive_scope":ARCHIVE_SCOPE,"excluded_auxiliary_display_files":[],
        "auxiliary_exclusion_reason":AUXILIARY_EXCLUSION_REASON})
    for name in row["files"]:
        if not (folder/name).is_file() or (folder/name).is_symlink():
            raise RuntimeError("previously archived numeric-core file disappeared/type changed: "+str(folder/name))
    destination = output/row["archived_folder"]
    destination.mkdir(exist_ok=True)
    for source in sorted(folder.iterdir()):
        if source.is_symlink() or not source.is_file():
            raise RuntimeError("snapshot contains nonregular file")
        if source.name not in NUMERIC_CORE:
            if source.name not in row["excluded_auxiliary_display_files"]:
                row["excluded_auxiliary_display_files"].append(source.name)
                row["excluded_auxiliary_display_files"].sort()
            continue
        data = source.read_bytes(); sha = _sha(data)
        if source.name in row["files"]:
            if row["files"][source.name]["sha256"] != sha:
                raise RuntimeError("previously archived snapshot changed: " + str(source))
            continue
        target = destination/source.name
        with target.open("xb") as stream:
            stream.write(data)
        if _sha(target.read_bytes()) != sha or _sha(source.read_bytes()) != sha:
            raise RuntimeError("snapshot copy/source changed")
        row["files"][source.name] = {"path":str(target.relative_to(output)), "sha256":sha,
                                      "original_path":str(source)}
    row["all_five_numeric_core_files_archived"] = set(row["files"])==set(NUMERIC_CORE)
    return row


def _public_lock():
    import fcntl
    stream = open(f"/run/user/{os.getuid()}/ballplate_current_public_15328133.lock", "a")
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def run_capture(plan, plan_path, plan_sha, output, client, *, clock=None):
    """Narrow orchestration body; injectable public clients for CPU-only tests."""
    repeats=_state_repetitions(plan)
    observable=plan.get("profile")=="observability_v1"
    if observable:_manifest_checks(plan)  # also fail closed for direct orchestration callers, BEFORE any park
    clock = clock or Clock()
    session = Path(plan["session"]); root = Path(output)
    root.mkdir(exist_ok=False); (root/"raw").mkdir()
    with (root/"PLAN.json").open("xb") as stream:
        stream.write(plan_path.read_bytes())
    started = clock.monotonic(); end = started+300; work_end = end-12
    states = [dict(id=s["id"], name=s["id"], kind=s["kind"], time_block=s["time_block"],
                   repeat_expected=s["repeat"], status="not_started", attempt_ids=[], valid_frame_count=0) for s in plan["states"]]
    if observable:
        for spec,row in zip(plan["states"],states):row.update(input_key=spec["input_key"],mask_key=plan["mask_key"])
    result = dict(schema=RESULT_SCHEMA, complete=False, error=None, states=states, attempts_file="attempts.jsonl",
        expected_states=len(states), expected_frames=plan["total_planned_frames"], frames_per_state=repeats,
        captured_valid_frames=0, frame_pattern_association_proven=False,
        hardware_connected_by_this_adapter=False, original_camera_stream_retained=True,
        original_camera_start_stop_calls=0, camera_or_controller_parameters_written=False,
        no_new_output_windows=True, no_training=True, no_closed_loop=True,
        production_qualified=False, hardware_release_approved=False, control_release_approved=False,
        inter_block_wait_seconds=0, time_blocks_are_NOT_independent_delayed_replicates=True,
        cooperative_time_bound_seconds=300, raw_snapshot_index="RAW_SNAPSHOT_INDEX.json", stopped_by_bound=False,
        actual_outputs_zero=False, camera_left_running=None, public_park_requested=False,
        archive_scope=ARCHIVE_SCOPE,snapshot_completion_rule=CORE_READY_RULE,
        excluded_auxiliary_display_files_reason=AUXILIARY_EXCLUSION_REASON,
        heartbeat_read_only_publication_wait_max_seconds=2.5,heartbeat_accepted_age_strictly_less_than_seconds=2)
    ui = watcher = lock = None; initial = None; ticks = None; accessibility=None
    baseline = set(session.glob("snapshot_*")); copied = {}; valid_folders = set(); attempts = 0; last_id = last_tick = -1
    def event(event_name, **values):
        _append(root/"events.jsonl", dict(event=event_name, unix_ns=clock.time_ns(), **values))
    def operation(state_id, name, action):
        event("state_substep_intent",state_id=state_id,substep=name,public_method_call_limit=1)
        try:
            value=action()
        except BaseException as exc:
            event("state_substep_failed",state_id=state_id,substep=name,error=repr(exc),
                  action_may_have_been_sent=True,action_will_NOT_be_retried=True)
            raise
        event("state_substep_ack",state_id=state_id,substep=name)
        return value
    def check_work():
        if clock.monotonic() >= work_end:
            result["stopped_by_bound"] = True
            raise TimeoutError("stop capture to preserve12s cleanup reserve; no shortened or replacement state")
        if watcher is not None and watcher.poll() is not None:
            raise RuntimeError("independent watchdog exited; no new projection")
        _source_checks(plan, plan_path, plan_sha)
        return _heartbeat(session, plan, clock, ticks)
    try:
        lock = _public_lock()
        initial, main, ticks = _prepark_identity(client, session, plan, clock)
        accessibility=_install_accessibility_compatibility()
        _new(root/"EXTERNAL_ACCESSIBILITY_COMPATIBILITY.json",accessibility)
        _new(root/"INITIAL_OWNER.json", dict(heartbeat=initial, main=main, start_ticks=ticks,
            initial_projection_armed=initial["projection_armed"], read_only_before_public_park=True))
        button, _ = client.one_control(initial["pid"], "push button", "两路全0，CCD保持（Esc）", main["rect"])
        client.press(button)
        result["public_park_requested"] = True
        until = min(end, clock.monotonic()+5)
        while _heartbeat(session, plan, clock, ticks)["projection_armed"]:
            if clock.monotonic() >= until:
                raise TimeoutError("public all-zero park acknowledgement missing")
            clock.sleep(.08)
        before = client.inspect(session, parked=False)
        _new(root/"PREFLIGHT.json", before)
        ui = client.PilotUI(session, before, root, end)
        ui.guard(black=True)
        _new(root/"EXTERNAL_POINTER_COMPATIBILITY.json", client.install())
        _new(root/"IMPORTED_OLD_SOURCE_COVERAGE.json", _loaded_source_coverage(plan))
        parent_start = _start_ticks(os.getpid())
        watcher = subprocess.Popen([sys.executable, str(Path(plan["legacy_root"])/"tools/static_learned_ccd_v28.py"),
            "--mode","watchdog","--session",str(session),"--output",str(root),"--parent",str(os.getpid()),
            "--parent-start",parent_start,"--hard-deadline",str(time.time()+max(0,end-clock.monotonic()))])
        client.find_tab(ui)
        try:
            original_cache = dict(status="verified_event_and_file_SHA_only",
                                  assets=client.cached_assets(session,initial))
        except Exception as exc:
            original_cache = dict(status="unknown",error=repr(exc),assets=None)
        _new(root/"ORIGINAL_SETTINGS.json", dict(manual_settings=initial["manual_settings"],
            original_cached_file_identities=original_cache,restored=False,
            reason="initial lens/fiducial modes need not be supported by public selector; final remains disarmed file/zero"))
        np = client.np
        dmd = np.load(plan["input"]["path"], allow_pickle=False)
        input_arrays=({k:np.load(v["path"],allow_pickle=False) for k,v in plan["inputs"].items()} if observable else None)
        masks = {k:np.load(v["path"], allow_pickle=False) for k,v in plan["masks"].items()}
        if dmd.dtype != np.uint8 or dmd.shape != (600,600) or not np.isin(dmd,(0,255)).all():
            raise ValueError("frozen canonical600 binary DMD required")
        if observable and any(a.dtype!=np.uint8 or a.shape!=(600,600) or not np.isin(a,(0,255)).all() for a in input_arrays.values()):
            raise ValueError("every observability DMD must be exact canonical600 binary uint8")
        for name, value in masks.items():
            if value.dtype != np.uint8 or value.shape != (600,600) or not np.array_equal(value,value[::4,::4].repeat(4,0).repeat(4,1)):
                raise ValueError("canonical600 exact4-compatible raw gray required: "+name)
        # New pilot verifier proves macro16/P128/raw0,64 semantics; it does not
        # authorize a panel or pretend a phase LUT was measured.
        if observable:
            _verify_observability_plan(plan)
        elif plan.get("profile")=="static_case_v1":
            _verify_case_plan(plan)
        else:
            from .probes import verify_pilot_assets
            verify_pilot_assets(Path(plan["masks"]["Z"]["path"]).parent,
                                expected_manifest_sha256=plan["pilot_manifest_sha256"])
        check_work(); operation(None,"preload_DMD",lambda:ui.load_live("dmd",Path(plan["input"]["path"])))
        # Native file loading commits the owner's file mode itself. Load Z
        # while disarmed before the first BG, so the original strict pointer
        # menu selector is never needed for an initial lens->file transition.
        # In particular, do not change keyboard locks or accept CapsLock in
        # the frozen pointer guard to work around a rejected menu click.
        check_work(); ui.guard(black=True)
        preload = operation(None,"preload_SLM_fixed" if observable else "preload_SLM_Z",lambda:ui.load_live("slm",Path(plan["masks"]["Z"]["path"])))
        ui.wait(lambda h:h["manual_settings"]["dmd"]=="file" and h["manual_settings"]["slm"]=="file")
        check_work(); ui.guard(black=True)
        event("fixed_observability_mask_preloaded_disarmed" if observable else "canonical_Z_preloaded_disarmed",asset=plan["masks"]["Z"],file_load_acknowledgement=preload,
              mode_commit_verified=True,no_projection=True,keyboard_lock_state_unchanged=True)
        loaded_mask = "Z"
        loaded_input="REF"
        for spec, row in zip(plan["states"], states):
            check_work()
            if work_end-clock.monotonic() < 20:
                result["stopped_by_bound"] = True
                raise TimeoutError("less than20s active budget; next full declared state not started")
            row["status"] = "partial"; event("state_intent",state_id=row["id"],kind=row["kind"],repeat_expected=spec["repeat"])
            operation(row["id"],"arm_false",lambda:ui.arm(False))
            ui.wait(lambda h:not h["projection_armed"]); ui.guard(black=True)
            illuminated = spec["kind"] != "BG"
            if observable:
                key=spec["input_key"]
                if loaded_input!=key:
                    check_work();ui.guard(black=True)
                    operation(row["id"],"load_DMD_"+key,lambda:ui.load_live("dmd",Path(plan["inputs"][key]["path"])))
                    loaded_input=key
                dmd=input_arrays[key]
                row.update(input_asset=plan["inputs"][key],fixed_mask_asset=plan["masks"]["Z"])
                phase=masks["Z"]
            else:
                phase = masks["Z"] if not illuminated else masks[spec["kind"]]
            if not observable and illuminated and loaded_mask != spec["kind"]:
                check_work(); operation(row["id"],"load_SLM_"+spec["kind"],
                    lambda:ui.load_live("slm",Path(plan["masks"][spec["kind"]]["path"])))
                loaded_mask = spec["kind"]
            operation(row["id"],"DMD_file_mode",lambda:ui.mode("dmd","file"))
            operation(row["id"],"SLM_file_mode",lambda:ui.mode("slm","file"))
            settings = dict(initial["manual_settings"],dmd="file",slm="file")
            expected, cd, cs = client.expected_pixels(settings,dmd,phase,illuminated,"file")
            operation(row["id"],"arm_"+str(illuminated).lower(),lambda:ui.arm(illuminated))
            until = min(work_end, clock.monotonic()+5)
            while True:
                check_work(); h = ui.guard(black=not illuminated,expected=expected if illuminated else None,transition=True)
                checks = {r:ui.pixels.compare(tuple(before["windows"][r]["rect"]),expected[r]) for r in ("dmd","slm")}
                if h["last_verified_revision"] == h["display_revision"] and all(x["matches"] for x in checks.values()):
                    break
                if clock.monotonic() >= until:
                    raise TimeoutError("actual framebuffer revision/readback missing")
                clock.sleep(.08)
            revision = h["display_revision"]; verified_ns = clock.time_ns()
            panels = {r:dict(shape_yx=list(expected[r].shape),dtype="uint8_grayscale_equal_RGB",
                             pixel_sha256=_sha(expected[r].tobytes(order="C"))) for r in ("dmd","slm")}
            row.update(display_revision=revision,verified_unix_ns=verified_ns,display_readback=checks,
                       expected_panel=panels)
            event("state_committed",state_id=row["id"],display_revision=revision,display_readback=checks,
                  expected_panel=panels)
            settle = clock.monotonic()+1.8
            while clock.monotonic() < settle:
                check_work(); ui.guard(black=not illuminated,expected=expected if illuminated else None); clock.sleep(.08)
            for repeat in range(spec["repeat"]):
                check_work(); ui.guard(black=not illuminated,expected=expected if illuminated else None)
                attempts += 1
                attempt = dict(id="A"+str(attempts).zfill(4), state_id=row["id"],kind=row["kind"],
                    time_block=row["time_block"],repeat_index=repeat,status="failed",error=None,raw=None,metadata=None,
                    frame=None,camera_actual=None,display_revision=None,original_snapshot_folder=None,
                    frame_pattern_association_verified=False,SDK_complete_flag_in_metadata=None)
                if observable:attempt.update(input_key=spec["input_key"],input_asset=plan["inputs"][spec["input_key"]],mask_key=plan["mask_key"],fixed_mask_asset=plan["masks"]["Z"])
                row["attempt_ids"].append(attempt["id"])
                event("frame_attempt_intent",**{k:attempt[k] for k in ("id","state_id","repeat_index")})
                old = set(session.glob("snapshot_*"))
                try:
                    ui.button("snapshot"); until = min(work_end,clock.monotonic()+4)
                    while True:
                        check_work(); ui.guard(black=not illuminated,expected=expected if illuminated else None)
                        new = set(session.glob("snapshot_*"))-old
                        if len(new)>1:
                            raise RuntimeError("ambiguous public snapshot; no frame substitution")
                        if new and all((next(iter(new))/name).is_file() for name in NUMERIC_CORE):
                            folder = next(iter(new))
                            # The frozen GUI writes these five core files before
                            # metadata, then writes optional display PNGs after
                            # it. This is a numeric-core barrier only; never
                            # pretend the entire folder is finished.
                            try:
                                _read(folder/"metadata.json")
                            except (OSError, ValueError):
                                pass
                            else:
                                break
                        if clock.monotonic()>=until:
                            raise TimeoutError("public raw snapshot not saved")
                        clock.sleep(.04)
                    event("numeric_core_ready",state_id=row["id"],attempt_id=attempt["id"],
                          folder=str(folder),completion_rule=CORE_READY_RULE,archive_scope=ARCHIVE_SCOPE)
                    archive = _copy_snapshot(folder,root,copied)
                    attempt["original_snapshot_folder"] = str(folder)
                    attempt["raw"] = archive["files"].get("CCD_raw.npy")
                    attempt["metadata"] = archive["files"].get("metadata.json")
                    meta = _read(root/attempt["metadata"]["path"])
                    raw = np.load(io.BytesIO((root/attempt["raw"]["path"]).read_bytes()),allow_pickle=False)
                    frame = meta["frame"]
                    attempt.update(frame=frame,camera_actual=meta["camera_active"],display_revision=meta["display_revision"],
                        host_get_started_unix_ns=frame.get("host_get_started_unix_ns"),
                        SDK_complete_flag_in_metadata=frame.get("IsIncomplete"))
                    if (raw.dtype!=np.uint8 or raw.shape!=(554,552) or raw.max()>=240
                            or meta["camera_active"]!=plan["camera_expected"]["active"]
                            or meta["display_revision"]!=revision
                            or frame["host_get_started_unix_ns"]<verified_ns+1_500_000_000
                            or frame["TimeStamp"]<=last_tick or frame["FrameID"]<=last_id
                            or frame.get("IsIncomplete") is True):
                        raise RuntimeError("native layout/saturation/camera/freshness/revision/order gate failed")
                    for role, array in (("dmd",cd),("slm",cs)):
                        png = root/archive["files"][role+"_canonical_requested.png"]["path"]
                        with client.Image.open(png) as image:
                            if not np.array_equal(np.asarray(image),array):
                                raise RuntimeError("saved canonical request differs: "+role)
                    last_tick=frame["TimeStamp"];last_id=frame["FrameID"]
                    attempt["status"]="valid";row["valid_frame_count"]+=1;result["captured_valid_frames"]+=1
                    valid_folders.add(str(folder))
                except BaseException as exc:
                    attempt["error"]=repr(exc)
                    raise
                finally:
                    _append(root/"attempts.jsonl",attempt)
            row["status"]="complete"
            operation(row["id"],"state_end_arm_false",lambda:ui.arm(False))
            ui.wait(lambda h:not h["projection_armed"]);ui.guard(black=True)
            _new(root/(row["id"]+".json"),row)
        result["all_capture_states_complete"] = True
    except BaseException as exc:
        result["error"] = repr(exc)
        if isinstance(exc,TimeoutError) and clock.monotonic()>=work_end:
            result["stopped_by_bound"] = True
    finally:
        try:
            if ui is not None:
                ui.button("allzero");ui.wait(lambda h:not h["projection_armed"]);ui.guard(black=True)
                result["actual_outputs_zero"] = True
                result["camera_left_running"] = ui.state()["live_camera"]
        except BaseException as exc:
            result.update(cleanup_error=repr(exc),actual_outputs_zero=False)
        # Preserve the numeric core of every new snapshot, even if a button
        # failed before its attempt acquired the path. Auxiliary display files
        # are explicitly excluded, not treated as immutable numeric evidence.
        raw_errors=[]
        for folder in sorted((set(session.glob("snapshot_*"))-baseline)|{Path(p) for p in copied}):
            try:
                _copy_snapshot(folder,root,copied)
            except BaseException as exc:
                raw_errors.append(dict(folder=str(folder),error=repr(exc)))
        _new(root/"RAW_SNAPSHOT_INDEX.json",dict(snapshots=list(copied.values()),errors=raw_errors,
            invalid_or_unassociated_folders_are_NOT_valid_attempts=True,archive_scope=ARCHIVE_SCOPE,
            numeric_core_files=list(NUMERIC_CORE),snapshot_completion_rule=CORE_READY_RULE,
            excluded_auxiliary_display_files=[dict(original_folder=r["original_folder"],
                files=r["excluded_auxiliary_display_files"]) for r in copied.values()],
            auxiliary_exclusion_reason=AUXILIARY_EXCLUSION_REASON,complete_legacy_snapshot_archive=False))
        if raw_errors:result["archive_error"]=raw_errors
        _new(root/"FINISHED.json",dict(parent_capture_finished=True,actual_outputs_zero=result.get("actual_outputs_zero",False),
            unix_ns=clock.time_ns(),complete_is_declared_only_in_RESULT_after_watchdog=True))
        if watcher is not None:
            try:
                watcher.wait(timeout=max(.1,min(6,end-clock.monotonic())))
                watchdog=_read(root/"WATCHDOG_RESULT.json")
                result["watchdog_result"]=watchdog
                result["watchdog_verified"]=watchdog.get("actual_all_zero") is True and not watchdog.get("error") and not watchdog.get("intervened")
            except BaseException as exc:
                result.update(watchdog_error=repr(exc),watchdog_verified=False)
        else:result["watchdog_verified"]=False
        try:
            _source_checks(plan,plan_path,plan_sha)
            for record in _plan_assets(plan):_asset(record)
            _manifest_checks(plan)
            result["frozen_inputs_unchanged"]=True
        except BaseException as exc:
            result.update(integrity_error=repr(exc),frozen_inputs_unchanged=False)
        result["archived_snapshot_count"]=len(copied)
        result["snapshots_without_valid_attempt"] = sorted(set(copied)-valid_folders)
        for name, resource in (("pixel_reader", None if ui is None else ui.pixels), ("public_lock",lock)):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    result.setdefault("resource_cleanup_errors",[]).append(dict(resource=name,error=repr(exc)))
        result["complete"]=bool(result.get("all_capture_states_complete")
            and result["captured_valid_frames"]==plan["total_planned_frames"]
            and result.get("actual_outputs_zero") and result.get("camera_left_running")
            and result.get("watchdog_verified") and result.get("frozen_inputs_unchanged")
            and not result["snapshots_without_valid_attempt"]
            and not any(result.get(k) for k in ("error","cleanup_error","archive_error","integrity_error","stopped_by_bound","resource_cleanup_errors")))
        result["wall_seconds"]=clock.monotonic()-started
        if accessibility is not None:
            result["external_accessibility_compatibility"]=accessibility
        if plan.get("profile")=="static_case_v1":
            result.update(profile=plan["profile"],case_asset_manifest=plan["case_asset_manifest"],
                case_id=plan["case_id"],role=plan["role"],design=plan["design"],
                fixed_input=plan["fixed_input"],probe_spec=plan["probe_spec"])
        elif observable:
            result.update(profile="observability_v1",observability_bank=plan["observability_bank"],
                axis_family=plan["axis_family"],mask_key=plan["mask_key"],inputs=plan["inputs"],
                batch_id=plan["batch_id"],batch_spec=plan["batch_spec"],
                prospective_predictions=plan["prospective_predictions"],
                fixed_mask_asset=plan["masks"]["Z"],single_mask_fixed_for_batch=True,
                historical_SLM_six_contrast_semantics_used=False)
        result["original_manual_modes_restored"]=False
        result["final_projection_armed"]=False if result.get("actual_outputs_zero") else None
        _new(root/"RESULT.json",result)
    return root/"RESULT.json"


def execute(plan_path, output, expected_plan_sha256, *, execute=False, low_power_confirmed=False):
    plan, blob = validate_plan(plan_path,expected_plan_sha256)
    if not execute:
        return dict(validated=True,executed=False,old_code_imported=False,hardware_accessed=False,
                    states=11 if plan.get("profile")=="observability_v1" else 17,
                    expected_frames=plan["total_planned_frames"],plan_sha256=_sha(blob))
    if not low_power_confirmed:
        raise ValueError("explicit current low-power confirmation required")
    output=Path(output).resolve()
    if output.parent!=(PROJECT/"experiments").resolve():
        raise ValueError("one NEW direct child of this project's experiments required")
    if output.exists():raise FileExistsError(output)
    client=_load_clients(Path(plan["legacy_root"]).resolve())
    _loaded_source_coverage(plan)  # fail BEFORE the first public park action
    return run_capture(plan,Path(plan_path).resolve(),expected_plan_sha256,output,client)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True);p.add_argument("--plan-sha256",required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--execute",action="store_true")
    p.add_argument("--low-power-confirmed",action="store_true");a=p.parse_args(argv)
    result=execute(a.plan,a.output,a.plan_sha256,execute=a.execute,low_power_confirmed=a.low_power_confirmed)
    if isinstance(result,dict):print(json.dumps(result,ensure_ascii=False));return 0
    row=_read(result);print(json.dumps({k:v for k,v in row.items() if k!="states"},ensure_ascii=False))
    return 0 if row["complete"] else 1


if __name__=="__main__":
    raise SystemExit(main())
