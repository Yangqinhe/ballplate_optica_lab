"""Read-only restart evidence, no GUI/SDK/launch; retained geometry stays pending.
Only explicit prepare_restart lazily imports the old CPU contract validator.
"""
from __future__ import annotations
import hashlib, json, math, sys, time
import importlib.util
from pathlib import Path
LAB = Path(__file__).resolve().parents[2]
RESUME = {"line0_reference_retained_pending", "explicit_alignment_resume_pending"}
MANUAL_GEOMETRY_SCHEMA = "optical-lab-manual-photometry-geometry-retention-1"
REFERENCE_FILES = {"frozen_geometry.json", "capture.json", "model_reference.json"}

def _sha(data): return hashlib.sha256(data).hexdigest()

def _manual_photometry_changes(source_active, expected_active):
    """Only manual exposure/gain values may differ; all other keys stay exact."""
    if (not isinstance(source_active, dict) or not isinstance(expected_active, dict)
            or set(source_active) != set(expected_active)):
        raise ValueError("complete identical camera parameter keys required")
    changed = {k for k in source_active if source_active[k] != expected_active[k]}
    if not changed <= {"ExposureTime", "Gain"}:
        raise ValueError("only manual ExposureTime/Gain differences are permitted")
    for active in (source_active, expected_active):
        if active.get("ExposureAuto") != "Off" or active.get("GainAuto") != "Off":
            raise ValueError("manual ExposureAuto/GainAuto Off required")
        # Gain is a camera-readback value here, not a requested SDK write.
        # Do not guess a device's upper limit: the live control must query it.
        for key, low, high in (("ExposureTime", 50., 4000.), ("Gain", 0., math.inf)):
            raw = active.get(key)
            if isinstance(raw, bool):
                raise ValueError("finite bounded manual camera values required")
            try:
                value = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError("finite bounded manual camera values required") from error
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError("finite bounded manual camera values required")
    return {key: dict(source=source_active[key], expected=expected_active[key])
            for key in sorted(changed)}


def _read_explicit_reference(folder, session, heartbeat, read, *,
                             retained_geometry_with_manual_photometry_only=False):
    """Read an explicitly named current-session zero reference, never search."""
    if type(retained_geometry_with_manual_photometry_only) is not bool:
        raise ValueError("manual-photometry retention opt-in must be bool")
    folder, session = Path(folder).resolve(), Path(session).resolve()
    if (folder.parent.parent != session or not folder.parent.name.startswith("live_")
            or not folder.name.startswith("zero_reference_")):
        raise ValueError("explicit reference must belong to this owner's live session")
    reg = json.loads(read(folder / "frozen_geometry.json"))
    capture = json.loads(read(folder / "capture.json"))
    model = json.loads(read(folder / "model_reference.json"))
    active = capture["active"]
    if (reg.get("reference_folder") != str(folder) or reg.get("frozen") is not True
            or model != heartbeat["optical_reference"]
            or {k: int(active[k]) for k in ("OffsetX", "OffsetY", "Width", "Height")}
               != heartbeat["actual_roi"]):
        raise ValueError("explicit current reference differs from live owner ROI/model/active")
    if retained_geometry_with_manual_photometry_only:
        # Exposure/gain edits may have invalidated the live geometry flag.
        # Only an explicit bool is accepted; the original source MUST remain
        # frozen and only a NONEMPTY manual photometry change permits this
        # separate alignment-only prior. No current calibration is certified.
        if (type(heartbeat.get("geometry_frozen")) is not bool
                or not _manual_photometry_changes(active, heartbeat.get("camera_active_readback"))):
            raise ValueError("explicit changed manual photometry and current bool geometry status required")
    elif (heartbeat.get("geometry_frozen") is not True
          or active != heartbeat.get("camera_active_readback")):
        raise ValueError("explicit current reference differs from live owner ROI/model/active")
    return reg, active


def validate_manual_geometry_sidecar(path, sha256):
    """Read SHA-bound alignment-only evidence; never import Qt, SDK or a driver.

    The source registration/signature and camera capture remain untouched.
    This is not a resume/deployment contract and must not become registration
    accepted by quantitative jobs. The caller must separately check the NEW
    live camera/model/ROI before using its matrix for manual comparison only.
    """
    path = Path(path).resolve()
    data = path.read_bytes()
    if not isinstance(sha256, str) or len(sha256) != 64 or _sha(data) != sha256:
        raise ValueError("explicit manual geometry sidecar SHA differs")
    value = json.loads(data)
    if (value.get("schema") != MANUAL_GEOMETRY_SCHEMA
            or value.get("manual_alignment_only") is not True
            or any(value.get(k) is not False for k in
                   ("photometry_calibrated", "newly_fitted", "deployment_or_control_release"))):
        raise ValueError("explicit unqualified manual alignment prior required")
    source, expected = value["source_active"], value["expected_active"]
    changes = _manual_photometry_changes(source, expected)
    if not changes or value.get("manual_photometry_changes") != changes:
        raise ValueError("explicit nonempty manual photometry change evidence required")
    requested = value["requested_exposure_us"]
    if (isinstance(requested, bool) or not isinstance(requested, (float, int))
            or not math.isfinite(requested) or not 50 <= requested <= 4000
            or abs(float(expected["ExposureTime"]) - requested) > 50):
        raise ValueError("manual prior exposure differs from explicit new request")
    folder = Path(value["source_reference_folder"]).resolve()
    hashes = value["source_reference_sha256"]
    if (str(folder) != value["source_reference_folder"]
            or not isinstance(hashes, dict) or set(hashes) != REFERENCE_FILES):
        raise ValueError("three explicit original reference files required")
    evidence = {}
    for name in sorted(REFERENCE_FILES):
        blob = (folder / name).read_bytes()
        if _sha(blob) != hashes[name]:
            raise ValueError("original reference SHA differs: " + name)
        evidence[name] = json.loads(blob)
    registration = value["registration"]
    roi = value["roi"]
    if (registration != evidence["frozen_geometry.json"]
            or registration.get("frozen") is not True
            or registration.get("reference_folder") != str(folder)
            or source != evidence["capture.json"]["active"]
            or value["optical_config"] != evidence["model_reference.json"]
            or roi != {k: int(source[k]) for k in ("OffsetX", "OffsetY", "Width", "Height")}
            or roi != {k: int(expected[k]) for k in ("OffsetX", "OffsetY", "Width", "Height")}):
        raise ValueError("manual prior must retain original registration/capture/model/ROI unchanged")
    if (not isinstance(value.get("settings"), dict)
            or "file" in (value["settings"].get("dmd"), value["settings"].get("slm"))):
        raise ValueError("file caches cannot be restored through a manual geometry prior")
    return value

def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
def _validate(contract, legacy):
    path = legacy / "live_deploy_v20/contract.py"
    prior = sys.modules.get("diffraction_model")
    expected = legacy / "live_manual_v17/diffraction_model.py"
    if prior is not None and Path(prior.__file__).resolve() != expected:
        raise ValueError("a different legacy diffraction_model is already imported")
    spec = importlib.util.spec_from_file_location("_optical_lab_restart_contract", path)
    module = importlib.util.module_from_spec(spec)
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(legacy / "live_manual_v17"))
        spec.loader.exec_module(module)
        if Path(sys.modules["diffraction_model"].__file__).resolve() != expected:
            raise ValueError("wrong CPU reference module")
        module.validate(contract)
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode

def prepare_restart(session, output, *, requested_exposure_us,
                    source_gui_sha256=None, legacy_root=None, owner_extension_sources=None,
                    current_reference_folder=None, current_reference_sha256=None,
                    retained_geometry_with_manual_photometry_only=False):
    """Freeze an explicitly identified, stopped original v23 owner into NEW output.
    Output must be a direct child of this lab's experiments directory. The owner
    must remain alive; this function never stops it. No newest-report/reference
    search, file-cache restoration, SDK readback, fault clearing or auto-start.
    owner_extension_sources maps every declared original extension source path
    to an explicit archived source path (not the possibly edited current file).
    An explicit startup reference may remain pending, never newly revalidated.
    A user-recaptured current-session reference needs its explicit directory and
    three SHA256 identities; it also remains pending until new active readback.
    An explicit manual-photometry opt-in retains an old matrix only in a
    separate alignment-only sidecar; the legacy restart contract gets NO
    registration/expected_active and never represents the new photometry as
    calibrated. Original reference files, signature and baseline stay intact.
    """
    session, output = Path(session).resolve(), Path(output).resolve()
    legacy = (Path(legacy_root).resolve() if legacy_root else
              LAB.parent / "ballplate_fourframe96_deploy")
    if output.parent != (LAB / "experiments").resolve():
        raise ValueError("NEW direct experiments child required")
    if type(retained_geometry_with_manual_photometry_only) is not bool:
        raise ValueError("manual-photometry retention opt-in must be bool")
    if retained_geometry_with_manual_photometry_only and current_reference_folder is None:
        raise ValueError("manual-photometry retention needs an explicit current reference")
    if (isinstance(requested_exposure_us, bool) or not isinstance(requested_exposure_us, (int, float))
            or not math.isfinite(requested_exposure_us) or not 50 <= requested_exposure_us <= 4000):
        raise ValueError("explicit unique UI exposure in 50..4000 us required")
    output.mkdir(exist_ok=False)
    blobs, immutable = {}, {}
    def read(path, expected=None, stable=True):
        path = Path(path).resolve(); data = path.read_bytes()
        if expected is not None and _sha(data) != expected:
            raise ValueError("source/evidence SHA mismatch: " + str(path))
        if stable:
            if str(path) in immutable and immutable[str(path)] != _sha(data):
                raise ValueError("evidence changed while preparing: " + str(path))
            immutable[str(path)] = _sha(data)
        return data
    def archive(name, data):
        with (output / name).open("xb") as stream:
            stream.write(data)
        blobs[str(output / name)] = _sha(data)
        return output / name
    try:
        hb = read(session / "heartbeat.json", stable=False)
        ev = read(session / "events.jsonl", stable=False)
        rt = read(session / "runtime_sources.json")
        h, runtime = json.loads(hb), json.loads(rt)
        events = [(i + 1, line, json.loads(line)) for i, line in enumerate(ev.splitlines())]
        if (not 0 <= time.time_ns() - h["unix_ns"] < 2_000_000_000
                or h["pid"] != runtime["pid"] or h.get("deployment_ui_revision") != 23
                or any(h.get(k) is not False for k in ("busy", "live_camera", "projection_armed", "loop_active"))
                or h.get("adaptation_active") is not False or h.get("deployment_busy") is not False):
            raise ValueError("fresh complete stopped/disarmed original v23 heartbeat required")
        if "file" in (h["manual_settings"]["dmd"], h["manual_settings"]["slm"]):
            raise ValueError("file caches are not silently restored; original owner untouched")
        gui = legacy / "live_loop_v23/direct_alignment_gui.py"
        sources = dict(runtime["sources"])
        for name, sha in runtime["source_configs"].items():
            if name in sources and sources[name] != sha:
                raise ValueError("conflicting runtime source identities")
            sources[name] = sha
        if str(gui) not in sources or (source_gui_sha256 is not None and sources[str(gui)] != source_gui_sha256):
            raise ValueError("original GUI source identity differs")
        for name, sha in sources.items():
            read(name, sha)
        cmd = read(Path("/proc") / str(h["pid"]) / "cmdline", stable=False)
        argv = [x.decode() for x in cmd.split(b"\0") if x]
        cwd = (Path("/proc") / str(h["pid"]) / "cwd").resolve()
        def argument(flag):
            positions = [i for i, x in enumerate(argv) if x == flag]
            if len(positions) != 1 or positions[0] + 1 == len(argv):
                raise ValueError("unique explicit owner argument required: " + flag)
            p = Path(argv[positions[0] + 1]); return (cwd / p).resolve()
        script_owner = len(argv) > 1 and (cwd / argv[1]).resolve() == gui
        extension_owner = argv[1:3] == ["-m", "optical_lab.preview"] and argv.count("-m") == 1
        if argument("--output") != session or not (script_owner or extension_owner):
            raise ValueError("process is not the explicitly bound original owner")
        if extension_owner:
            if argument("--legacy-root") != legacy:
                raise ValueError("extension owner legacy root differs")
            declaration = read(session / "spatial_preview_extension.json")
            declared = json.loads(declaration)["new_source_sha256"]
            if (not isinstance(owner_extension_sources, dict) or not declared
                    or set(owner_extension_sources) != set(declared)):
                raise ValueError("complete explicit archived extension-source mapping required")
            archive("spatial_preview_extension.json", declaration)
            mapping = {}
            for i, (original, sha) in enumerate(declared.items()):
                archived = Path(owner_extension_sources[original]).resolve()
                if (not Path(original).is_absolute() or archived == Path(original).resolve()
                        or not archived.is_relative_to((LAB / "experiments").resolve())):
                    raise ValueError("explicit experiment source archive, not current source, required")
                copied = archive("extension_%02d_%s" % (i, Path(original).name), read(archived, sha))
                mapping[original] = dict(archived_path=str(archived), sha256=sha, copied_path=str(copied))
            _write(output / "owner_extension_sources.json", mapping)
            blobs[str(output / "owner_extension_sources.json")] = _sha((output / "owner_extension_sources.json").read_bytes())
        elif owner_extension_sources is not None:
            raise ValueError("extension mapping supplied for a non-extension owner")
        archive("heartbeat.json", hb); archive("events.jsonl", ev)
        runtime_path = archive("runtime_sources.json", rt); archive("owner_cmdline.bin", cmd)
        archive("preparer_source.py", read(__file__))
        rechecks = [x for x in events if x[2].get("state") == "original_baseline_recheck_finished"]
        if not rechecks:
            raise ValueError("no explicit completed baseline recheck; no report guessing")
        line_no, line, event = rechecks[-1]
        result = event["result"]; report = result["camera_report"]
        baseline = result["original_context"]["target"]
        if (result.get("unlocked") is not True or report.get("passed") is not True
                or report.get("target") != baseline or report.get("parameter_writes") != 0
                or report.get("acquisition_started") is not False or report.get("connections") != 2
                or len(report.get("snapshots", [])) != 10 or report.get("error") or report.get("errors")
                or any(x.get("differences") or x.get("values") != baseline for x in report["snapshots"])):
            raise ValueError("latest explicit baseline recheck is not strict and complete")
        resumes = [x for x in events if x[2].get("state") in RESUME]
        candidates = [("event_line_" + str(n), e["contract"]) for n, _, e in reversed(resumes)]
        startup_contract = None
        if "--restart-state" in argv:
            path = argument("--restart-state"); data = read(path)
            archive("original_restart_state.json", data)
            startup_contract = json.loads(data)
            candidates.insert(0, (str(path), startup_contract))
        bound = None
        for identity, candidate in candidates:
            if candidate.get("original_fault_before") != baseline:
                continue
            for name, sha in candidate.get("evidence_sha256", {}).items():
                data = read(name, sha)
                if Path(name).suffix == ".json":
                    old = json.loads(data)
                    if (old.get("camera_before") or old.get("target")) == baseline:
                        bound = identity, name, data; break
            if bound:
                break
        if bound is None:
            raise ValueError("recheck target is not bound to explicit original restart/resume evidence")
        archive("original_baseline_source.json", bound[2])
        baseline_path = output / "baseline_evidence.json"
        _write(baseline_path, dict(target=baseline, derived_evidence_NOT_new_measurement=True,
            original_contract=bound[0], original_source=bound[1], recheck_event_line=line_no,
            recheck_event_line_sha256=_sha(line)))
        blobs[str(baseline_path)] = _sha(baseline_path.read_bytes())
        registration = active = None
        manual_geometry_path = None
        reference_mode = "none"
        if current_reference_folder is not None:
            names = REFERENCE_FILES
            if not isinstance(current_reference_sha256, dict) or set(current_reference_sha256) != names:
                raise ValueError("all three explicit current-reference SHA256 identities required")
            folder = Path(current_reference_folder).resolve()
            registration, active = _read_explicit_reference(folder, session, h,
                lambda path: read(path, current_reference_sha256[path.name]),
                retained_geometry_with_manual_photometry_only=retained_geometry_with_manual_photometry_only)
            archive("explicit_current_reference.json", json.dumps(dict(
                folder=str(folder), sha256=current_reference_sha256,
                retained_as_pending_NOT_new_calibration=True), sort_keys=True).encode())
            reference_mode = "explicit_current_session_zero_reference_retained_as_pending"
        elif current_reference_sha256 is not None:
            raise ValueError("current-reference hashes require an explicit directory")
        elif h["geometry_frozen"]:
            matches = {}
            for n, _, e in resumes:
                c = e["contract"]; reg = c.get("registration")
                if not reg or c["optical_config"] != h["optical_reference"] or c["roi"] != h["actual_roi"]:
                    continue
                if any(k > n and v.get("state") == "retained_geometry_revalidated"
                       and v.get("signature") == reg["signature"] and v.get("active") == c["expected_active"]
                       for k, _, v in events):
                    matches[json.dumps(reg, sort_keys=True)] = (reg, c["expected_active"])
            if len(matches) != 1:
                raise ValueError("current frozen reference is not uniquely explicit and revalidated")
            registration, active = next(iter(matches.values()))
            if active != h.get("camera_active_readback"):
                raise ValueError("last active readback differs from the retained reference")
            reference_mode = "explicit_revalidated_reference_retained_as_pending"
        elif startup_contract is not None and startup_contract.get("registration") is not None:
            if (startup_contract.get("schema") != "v20-retained-line0-v1"
                    or startup_contract["roi"] != h["actual_roi"]
                    or startup_contract["optical_config"] != h["optical_reference"]
                    or startup_contract["original_fault_before"] != baseline):
                raise ValueError("explicit pending reference differs from current ROI/model/baseline")
            _validate(startup_contract, legacy)
            registration, active = startup_contract["registration"], startup_contract["expected_active"]
            if h.get("camera_active_readback") is not None and active != h["camera_active_readback"]:
                raise ValueError("last active readback differs from explicit pending reference")
            reference_mode = "explicit_startup_pending_reference_NOT_revalidated"
        if registration is not None:
            for name in ("frozen_geometry.json", "capture.json", "model_reference.json"):
                path = Path(registration["reference_folder"]) / name
                data = read(path); archive("reference_" + name, data); blobs[str(path)] = _sha(data)
        if retained_geometry_with_manual_photometry_only and active != h["camera_active_readback"]:
            changes = _manual_photometry_changes(active, h["camera_active_readback"])
            manual_geometry_path = output / "geometry_for_manual_alignment.json"
            _write(manual_geometry_path, dict(schema=MANUAL_GEOMETRY_SCHEMA,
                registration=registration, source_active=active,
                expected_active=h["camera_active_readback"], optical_config=h["optical_reference"],
                roi=h["actual_roi"], settings=h["manual_settings"],
                requested_exposure_us=float(requested_exposure_us),
                source_reference_folder=str(Path(current_reference_folder).resolve()),
                source_reference_sha256=current_reference_sha256,
                manual_photometry_changes=changes, manual_alignment_only=True,
                current_geometry_frozen_at_prepare=h["geometry_frozen"],
                current_frozen_geometry_certified=False,
                photometry_calibrated=False, newly_fitted=False, deployment_or_control_release=False))
            blobs[str(manual_geometry_path)] = _sha(manual_geometry_path.read_bytes())
            validate_manual_geometry_sidecar(manual_geometry_path, blobs[str(manual_geometry_path)])
            # Never forge a new camera capture, registration signature or a
            # validated deployment contract for the changed photometry.
            registration = active = None
            reference_mode = "manual_photometry_changed_geometry_retained_for_alignment_ONLY"
        roles = [e["video"] for _, _, e in events if e.get("state") == "created"]
        if len(roles) != 1:
            raise ValueError("unique explicit original display-role event required")
        contract = dict(schema="v20-retained-line0-v1", saved_unix_ns=time.time_ns(),
            original_session=str(session), original_pid=h["pid"], runtime_path=str(runtime_path),
            settings=h["manual_settings"], optical_config=h["optical_reference"], roi=h["actual_roi"],
            requested_exposure_us=float(requested_exposure_us), fault=h["hardware_fault"],
            original_fault_before=baseline, original_last_camera_report=report, registration=registration,
            expected_active=active, display_roles=roles[0], evidence_sha256=dict(blobs),
            no_auto_capture=True, no_auto_projection=True, release_approved=False)
        _validate(contract, legacy)
        for name, sha in immutable.items():
            if _sha(Path(name).read_bytes()) != sha:
                raise ValueError("immutable input changed before sealing: " + name)
        final = json.loads((session / "heartbeat.json").read_bytes())
        keys = ("pid", "deployment_ui_revision", "manual_settings", "optical_reference", "actual_roi",
                "hardware_fault", "geometry_frozen", "camera_active_readback", "busy", "live_camera",
                "projection_armed", "loop_active", "adaptation_active", "deployment_busy")
        if (not 0 <= time.time_ns() - final["unix_ns"] < 2_000_000_000
                or any(final.get(k) != h.get(k) for k in keys)):
            raise ValueError("fresh original owner settings changed before sealing")
        contract_path = output / "restart_contract.json"
        _write(contract_path, contract)
        recovery_path = output / "recovery_record.json"
        _write(recovery_path, dict(schema="manual-alignment-recovery-v1", baseline_source=str(contract_path),
            baseline_source_sha256=_sha(contract_path.read_bytes()), camera_baseline=baseline,
            original_fault=h["hardware_fault"], old_pid=h["pid"], manual_stop_requires_recheck=True,
            planned_normal_restart=True, no_forced_termination=True))
        _write(output / "RESULT.json", dict(complete=True, hardware_accessed=False, gui_actions=False,
            fault_preserved=h["hardware_fault"], reference_is_pending_prior_NOT_new_calibration=True,
            reference_retention_mode=reference_mode, original_geometry_frozen=h["geometry_frozen"],
            original_input_sha256=immutable, restart_contract=str(output / "restart_contract.json"),
            manual_geometry_sidecar=None if manual_geometry_path is None else str(manual_geometry_path)))
        files = {p.name: _sha(p.read_bytes()) for p in output.iterdir() if p.is_file()}
        _write(output / "MANIFEST.json", dict(schema="optical-lab-preview-restart-1", file_sha256=files))
        return dict(output=str(output), contract=str(contract_path), recovery_record=str(recovery_path),
            manual_geometry_sidecar=None if manual_geometry_path is None else str(manual_geometry_path),
            manual_geometry_sha256=None if manual_geometry_path is None else blobs[str(manual_geometry_path)],
            contract_sha256=files["restart_contract.json"], recovery_record_sha256=files["recovery_record.json"],
            manifest_sha256=_sha((output / "MANIFEST.json").read_bytes()))
    except Exception as exc:
        _write(output / "FAILED.json", dict(complete=False, error=repr(exc), hardware_accessed=False))
        raise
