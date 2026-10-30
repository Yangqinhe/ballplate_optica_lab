"""Single research entry point. Subcommand imports remain isolated."""
import argparse
import json
from pathlib import Path

from .storage import PROJECT, verify_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Optical platform research: assess before retrain. No live device commands.")
    commands = parser.add_subparsers(dest="command", required=True)
    references = commands.add_parser("references", help="read-only identity check of two preserved projects")
    references.add_argument("--profile", choices=("simulation", "hardware", "all"), default="all")
    references.add_argument("--details", action="store_true", help="include the full per-file identity ledger")
    historical = commands.add_parser("assess", help="recompute ALL declared historical real/model cells")
    historical.add_argument("--sources", type=Path, default=PROJECT/"configs/historical_sources.json")
    historical.add_argument("--protocol", type=Path, default=PROJECT/"configs/assessment_protocol.json")
    historical.add_argument("--output", required=True, type=Path, help="NEW child of an existing experiments directory")
    view = commands.add_parser("view", help="locate offline evidence viewer; never open hardware")
    view.add_argument("--run", type=Path, required=True)
    view.add_argument("--open", action="store_true", help="explicitly open local HTML in a browser; default prints path only")
    check = commands.add_parser("verify-run", help="verify a completed experiment archive, not hardware qualification")
    check.add_argument("--run", type=Path, required=True)
    pilot = commands.add_parser("prepare-pilot", help="generate fixed offline SLM Z/A/B assets; never project or capture")
    pilot.add_argument("--protocol", type=Path, default=PROJECT/"configs/assessment_protocol.json")
    pilot.add_argument("--output", type=Path, required=True)
    pilot_check = commands.add_parser("verify-pilot", help="verify offline probe files, not E002 measurements")
    pilot_check.add_argument("--run", type=Path, required=True)
    pilot_check.add_argument("--manifest-sha256")
    case = commands.add_parser("prepare-case", help="generate one explicit offline study case; never touches devices")
    case.add_argument("--design", type=Path, required=True)
    case.add_argument("--case-id", required=True)
    case.add_argument("--output", type=Path, required=True)
    case_check = commands.add_parser("verify-case", help="verify self-contained case archive and exact raw probe semantics")
    case_check.add_argument("--run", type=Path, required=True)
    case_check.add_argument("--manifest-sha256", required=True)
    encoding = commands.add_parser("prepare-encoding", help="offline RAW96 ball-thickness input bank; no optical zoom or devices")
    encoding.add_argument("--plan", type=Path, required=True)
    encoding.add_argument("--plan-sha256", required=True)
    encoding.add_argument("--output", type=Path, required=True)
    simulate = commands.add_parser("predict-pilot", help="freeze CPU-only model predictions before any new measurement")
    simulate.add_argument("--pilot", type=Path, required=True)
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--execute", action="store_true", help="explicitly run the frozen numerical models, not any device")
    simulate.add_argument("--case-manifest-sha256", help="explicit new static-case profile instead of the original fixed pilot")
    inputs = commands.add_parser("prepare-inputs", help="freeze and optionally render new CPU MuJoCo pose histories; not hardware")
    inputs.add_argument("--plan", type=Path, required=True)
    inputs.add_argument("--plan-sha256", required=True)
    inputs.add_argument("--output", type=Path, required=True)
    inputs.add_argument("--execute", action="store_true", help="explicitly render via isolated CPU OSMesa; no GUI or old dataset loader")
    input_report = commands.add_parser("assess-inputs", help="offline provenance audit and common-scale input history figure")
    input_report.add_argument("--bank", type=Path, required=True)
    input_report.add_argument("--manifest-sha256", required=True)
    input_report.add_argument("--output", type=Path, required=True)
    capture = commands.add_parser("prepare-capture", help="freeze original-window static plan; no UI calls or projection")
    capture.add_argument("--pilot", type=Path, required=True)
    capture.add_argument("--predictions", type=Path, required=True)
    capture.add_argument("--prediction-manifest-sha256", required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--authority-statement", required=True, help="current user's specific unchanged-condition confirmation")
    capture.add_argument("--case-manifest-sha256", help="explicit new static-case profile instead of the original fixed pilot")
    capture.add_argument("--frames-per-state", type=int, choices=(8, 16), default=16,
                         help="stationary CCD repeats only; optical network still has four temporal input images")
    pilot_report = commands.add_parser("assess-pilot", help="offline native-ADU analysis of a real static pilot, including partial failures")
    pilot_report.add_argument("--capture", type=Path, required=True)
    pilot_report.add_argument("--output", type=Path, required=True)
    pilot_report.add_argument("--capture-manifest-sha256")
    encoding_report = commands.add_parser("assess-encoding", help="offline four-slot input-thickness comparison, including missing/failed slots")
    encoding_report.add_argument("--index", type=Path, required=True)
    encoding_report.add_argument("--index-sha256", required=True)
    encoding_report.add_argument("--output", type=Path, required=True)
    obs_bank = commands.add_parser("prepare-observability-bank", help="offline local histories and immutable task masks; never project")
    obs_bank.add_argument("--design", type=Path, required=True)
    obs_bank.add_argument("--design-sha256", required=True)
    obs_bank.add_argument("--cpu-bank", type=Path, required=True)
    obs_bank.add_argument("--cpu-manifest-sha256", required=True)
    obs_bank.add_argument("--output", type=Path, required=True)
    obs_plan = commands.add_parser("prepare-observability-plan", help="read-only original-owner plan; no UI actions or hardware initialization")
    obs_plan.add_argument("--bank", type=Path, required=True)
    obs_plan.add_argument("--bank-sha256", required=True)
    obs_plan.add_argument("--batch-id", required=True)
    obs_plan.add_argument("--predictions", type=Path, required=True)
    obs_plan.add_argument("--prediction-manifest-sha256", required=True)
    obs_plan.add_argument("--authority-statement", required=True)
    obs_plan.add_argument("--output", type=Path, required=True)
    obs_report = commands.add_parser("assess-observability", help="offline single-frame local input-response analysis; not a control release")
    obs_report.add_argument("--index", type=Path, required=True)
    obs_report.add_argument("--index-sha256", required=True)
    obs_report.add_argument("--output", type=Path, required=True)
    obs_predict = commands.add_parser("predict-observability", help="freeze CPU M0/M1 and four-parameter sensitivity; never hardware or fitting")
    obs_predict.add_argument("--bank", type=Path, required=True)
    obs_predict.add_argument("--bank-sha256", required=True)
    obs_predict.add_argument("--output", type=Path, required=True)
    obs_predict.add_argument("--execute", action="store_true", help="explicit bounded CPU propagation; default validates only")
    physics_report = commands.add_parser("assess-physical-sensitivity", help="sealed704-frame model/real sensitivity audit; no parameter fitting")
    physics_report.add_argument("--report", type=Path, required=True)
    physics_report.add_argument("--report-manifest-sha256", required=True)
    physics_report.add_argument("--predictions", type=Path, required=True)
    physics_report.add_argument("--prediction-manifest-sha256", required=True)
    physics_report.add_argument("--output", type=Path, required=True)
    lowpass = commands.add_parser("assess-lowpass", help="offline spatial-only CCD low-pass comparison; raw data and live policy unchanged")
    lowpass.add_argument("--index", type=Path, required=True)
    lowpass.add_argument("--index-sha256", required=True)
    lowpass.add_argument("--output", type=Path, required=True)
    preprocessing = commands.add_parser("assess-readout", help="offline paired fixed-intensity preprocessing and Linear18 refit; never deploys")
    preprocessing.add_argument("--head", type=Path, required=True)
    preprocessing.add_argument("--head-sha256", required=True)
    preprocessing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "references":
        from .references import verify
        profiles = ("simulation", "hardware") if args.profile == "all" else (args.profile,)
        reports = {profile:verify(profile) for profile in profiles}
        shown = reports if args.details else {name:{k:r[k] for k in
            ("profile", "checked_count", "verified", "scope", "old_code_imported", "hardware_accessed")}
            for name,r in reports.items()}
        print(json.dumps(shown, ensure_ascii=False, indent=2))
        return 0 if all(r["verified"] is True for r in reports.values()) else 1
    if args.command == "assess":
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        from .assessment import assess
        r = assess(args.sources, args.protocol, args.output)
        print(json.dumps({k:r[k] for k in ("complete", "cell_count", "real_signal_frames", "probe_pair_count", "platform_modulation_efficiency_acceptable")}, ensure_ascii=False))
        return 0
    if args.command == "verify-run":
        print(json.dumps(verify_experiment(args.run), ensure_ascii=False))
        return 0
    if args.command == "prepare-pilot":
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        from .probes import prepare_pilot
        result = prepare_pilot(args.protocol, args.output)
        print(json.dumps({"status":result["status"], "pilot_spec":result["pilot_spec"],
                          "captured_frames":result["unreleased"]["captured_frames"],
                          "projection_release_approved":False}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify-pilot":
        from .probes import verify_pilot_assets
        print(json.dumps(verify_pilot_assets(args.run, expected_manifest_sha256=args.manifest_sha256), ensure_ascii=False))
        return 0
    if args.command in ("prepare-case", "verify-case"):
        from .study import prepare_case, verify_case_assets
        if args.command == "prepare-case":
            if args.output.resolve().parent != (PROJECT/"experiments").resolve():
                parser.error("output must be one NEW direct child of this project's experiments/")
            result = prepare_case(args.design, args.case_id, args.output)
        else:
            result = verify_case_assets(args.run, expected_manifest_sha256=args.manifest_sha256)
        print(json.dumps({"verified": result["verified"], "manifest_sha256": result["manifest_sha256"],
                          "case_id": result["manifest"]["design"]["case_id"],
                          "role": result["manifest"]["design"]["role"], "hardware_accessed": False}, ensure_ascii=False))
        return 0
    if args.command == "predict-pilot":
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        from .simulate import prepare_predictions
        result = prepare_predictions(args.pilot, args.output, execute=args.execute,
                                     case_manifest_sha256=args.case_manifest_sha256)
        print(json.dumps({k: result[k] for k in ("completed", "propagation_executed", "metadata_validated", "backend", "output_created", "wall_seconds", "G2_model_qualified", "unreleased") if k in result}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-encoding":
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        from .study import prepare_encoding
        print(json.dumps(prepare_encoding(args.plan, args.output, expected_plan_sha256=args.plan_sha256),
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-inputs":
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        from .simulate import prepare_inputs
        result = prepare_inputs(args.plan, args.output, expected_plan_sha256=args.plan_sha256, execute=args.execute)
        print(json.dumps({k: result[k] for k in ("completed", "metadata_validated", "output_created", "backend",
                                                "wall_seconds", "eligible_input_count", "hardware_accessed") if k in result}, ensure_ascii=False))
        return 0
    if args.command == "assess-inputs":
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        from .assessment import assess_inputs
        result = assess_inputs(args.bank, args.output, expected_manifest_sha256=args.manifest_sha256)
        print(json.dumps({k: result[k] for k in ("schema", "audit_completed", "row_count", "all_rows_eligible",
                                                "hardware_accessed", "control_release_approved") if k in result}, ensure_ascii=False))
        return 0
    if args.command == "prepare-capture":
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        from .capture import prepare_capture
        result = prepare_capture(args.pilot, args.predictions, args.output,
                                 authority_statement=args.authority_statement,
                                 prediction_manifest_sha256=args.prediction_manifest_sha256,
                                 case_manifest_sha256=args.case_manifest_sha256,
                                 frames_per_state=args.frames_per_state)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "assess-pilot":
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        from .assessment import assess_pilot
        result = assess_pilot(args.capture, args.output,
                              expected_capture_manifest_sha256=args.capture_manifest_sha256)
        print(json.dumps({k: result[k] for k in ("schema", "planned_capture_and_analysis_complete", "planned_frames", "valid_frames", "available_primary_contrasts", "gate_states", "hardware_release_approved") if k in result}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "assess-encoding":
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        from .assessment import assess_encoding_study
        result = assess_encoding_study(args.index, args.output, expected_index_sha256=args.index_sha256)
        print(json.dumps({key: result[key] for key in ("schema", "analysis_complete", "all_four_complete", "hardware_accessed",
                                                      "control_release_approved") if key in result}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "assess-lowpass":
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        from .filtering import assess_lowpass
        result = assess_lowpass(args.index, args.output, expected_index_sha256=args.index_sha256)
        print(json.dumps({key: result[key] for key in ("schema", "analysis_complete", "hardware_accessed",
                         "control_release_approved") if key in result}, ensure_ascii=False))
        return 0
    if args.command == "assess-readout":
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        from .readout_assessment import assess_preprocessing
        result = assess_preprocessing(args.head, args.output, expected_head_sha256=args.head_sha256)
        print(json.dumps({key: result[key] for key in ("schema", "analysis_complete", "hardware_accessed",
                         "control_release_approved") if key in result}, ensure_ascii=False))
        return 0
    if args.command in ("prepare-observability-bank", "prepare-observability-plan", "assess-observability", "predict-observability", "assess-physical-sensitivity"):
        if args.output.resolve().parent != (PROJECT/"experiments").resolve():
            parser.error("output must be one NEW direct child of this project's experiments/")
        if args.command == "prepare-observability-bank":
            from .observability import prepare_bank
            result = prepare_bank(args.design, args.cpu_bank, args.output,
                design_sha256=args.design_sha256, cpu_manifest_sha256=args.cpu_manifest_sha256)
        elif args.command == "prepare-observability-plan":
            from .observability import prepare_plan
            result = prepare_plan(args.bank, args.output, bank_sha256=args.bank_sha256,
                batch_id=args.batch_id, authority_statement=args.authority_statement,
                predictions_root=args.predictions, predictions_sha256=args.prediction_manifest_sha256)
        elif args.command == "predict-observability":
            from .identification import prepare_sensitivity
            result = prepare_sensitivity(args.bank, args.output, expected_bank_sha256=args.bank_sha256, execute=args.execute)
        elif args.command == "assess-physical-sensitivity":
            from .identification_assessment import assess_observed
            result = assess_observed(args.report, args.output, report_manifest_sha256=args.report_manifest_sha256,
                predictions_dir=args.predictions, predictions_manifest_sha256=args.prediction_manifest_sha256)
        else:
            from .observability_assessment import assess
            result = assess(args.index, args.output, expected_index_sha256=args.index_sha256)
        shown = result if args.command in ("prepare-observability-bank", "prepare-observability-plan") else {
            key: result[key] for key in ("schema", "complete", "analysis_complete", "validated", "propagation_executed",
                "output_created", "hardware_accessed", "nominal_prediction_count", "sensitivity_command_count",
                "planned_frames", "valid_frames", "control_release_approved") if key in result}
        print(json.dumps(shown, ensure_ascii=False, indent=2))
        return 0
    verify_experiment(args.run)
    target = args.run.resolve()/"dashboard.html"
    if not target.is_file():
        parser.error("completed report dashboard not found: "+str(target))
    print(target)
    if args.open:
        import webbrowser
        webbrowser.open(target.as_uri())
    return 0
