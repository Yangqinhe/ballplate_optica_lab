"""Retrospective matched-raw Linear18 study. No GUI, SDK, rollout or deployment.

Only explicit assess_preprocessing imports the SHA-checked legacy NumPy fitter.
The old train/select/test partitions have already been seen; none is a new test.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from . import readout, reporting, storage

LAB = Path(__file__).resolve().parents[2]
WORKSPACE = LAB.parent
LEGACY = WORKSPACE / "ballplate_fourframe96_deploy"
CORE_SHA256 = "ec6d37c77314af399b617aeebe9afc62e888f036dffc7cb5951c39aca374aa79"
NEG200_PHASE_SHA256 = "3c62987e51fd0e2bc7bc912d3b146253286c180c608d2c848136a9b05bb45b88"
SCHEMA = "optical-lab-retrospective-readout-assessment-1"
ARMS = ("raw", "requested_float", "requested_uint8")
REPRODUCTION_RTOL, REPRODUCTION_ATOL = 1e-8, 1e-10
RAW8_REDUCTION_ATOL = 1e-6  # ADU sum; only floating-point summation-order audit.


def _load_core(core_path, source):
    """Execute only captured, checked CPU core bytes; no legacy entry point."""
    if hashlib.sha256(source).hexdigest() != CORE_SHA256:
        raise ValueError("legacy NumPy core source differs from the frozen source")
    module = ModuleType("_optical_lab_matched_raw18_core")
    module.__file__ = str(core_path)
    exec(compile(source, str(core_path), "exec"), module.__dict__)
    return module


def _reproduction(fitted, original):
    differences = {}
    for name in ("weight_raw_ADU_to_rad", "bias_rad"):
        a, b = np.asarray(fitted[name], float), np.asarray(original[name], float)
        if a.shape != b.shape or not np.allclose(a, b, rtol=REPRODUCTION_RTOL, atol=REPRODUCTION_ATOL):
            raise ValueError("raw refit does not reproduce original " + name)
        differences[name + "_max_abs_difference"] = float(np.max(np.abs(a-b)))
    if fitted["selected_ridge"] != original["selected_ridge"]:
        raise ValueError("raw refit selected ridge differs from original")
    differences["metrics_max_abs_difference"] = 0.
    for split in ("train", "select", "test"):
        if set(fitted["metrics"][split]) != set(original["metrics"][split]):
            raise ValueError("raw refit metric definition changed")
        for key, value in fitted["metrics"][split].items():
            other = original["metrics"][split][key]
            if not np.isclose(value, other, rtol=REPRODUCTION_RTOL, atol=REPRODUCTION_ATOL):
                raise ValueError("raw refit does not reproduce original metric " + split + "/" + key)
            differences["metrics_max_abs_difference"] = max(
                differences["metrics_max_abs_difference"], abs(float(value)-float(other)))
    return dict(passed=True, rtol=REPRODUCTION_RTOL, atol=REPRODUCTION_ATOL, **differences)


def _scores(core, actions, labels, groups, splits):
    return dict(
        by_split={s: core.metrics(actions[splits == s], labels[splits == s])
                  for s in ("train", "select", "test")},
        by_trajectory=[dict(trajectory=int(g), split=str(splits[groups == g][0]),
            unique_inputs=int(np.sum(groups == g)), frames=int(np.sum(groups == g)*actions.shape[1]),
            **core.metrics(actions[groups == g], labels[groups == g])) for g in np.unique(groups)])


def _figures(out, results, actions, direct, labels, groups, splits, examples, example_index):
    plt = reporting.plotting()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = ("#222222", "#0072B2", "#D55E00")
    for row, scope in enumerate(("by_split", "by_trajectory")):
        names = ["train", "select", "test"] if row == 0 else [str(int(g)) for g in np.unique(groups)]
        for col, (key, unit) in enumerate((("MAE_rad", "Action MAE / rad"),
                ("within_input_action_RMS_rad", "Within-input SINGLE-frame action RMS / rad"))):
            ax = axes[row, col]; x = np.arange(len(names))
            for j, arm in enumerate(ARMS):
                scores = results[arm][scope]
                vals = [scores[n][key] for n in names] if row == 0 else [r[key] for r in scores]
                ax.bar(x+(j-1)*.25, vals, .25, label=arm, color=colors[j])
            ax.set_xticks(x, names, rotation=0 if row == 0 else 65, fontsize=7)
            ax.set_ylabel(unit); ax.set_xlabel("Existing split" if row == 0 else "ALL existing trajectories")
            ax.grid(axis="y", alpha=.2); ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("HISTORICAL MATCHED RAW / RETROSPECTIVE, NOT NEW HOLDOUT OR CONTROL")
    fig.tight_layout(); reporting.save_figure(fig, out/"figures/01_action_error_and_jitter", 180); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    all_actions = np.concatenate([a[example_index].ravel() for a in (*actions.values(), *direct.values())])
    bound = max(.05, float(np.max(np.abs(all_actions)))*1.05, float(np.max(np.abs(labels[example_index])))*1.05)
    for j, arm in enumerate(ARMS):
        im = axes[0, j].imshow(examples[arm], cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(arm + " | repeat 0"); axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        for axis in range(2):
            axes[1, j].plot(actions[arm][example_index, :, axis], label="own refit " + "xy"[axis])
            axes[1, j].plot(direct[arm][example_index, :, axis], ":", label="OLD head direct " + "xy"[axis])
            axes[1, j].axhline(labels[example_index, axis], ls="--", lw=.7, color=colors[axis+1])
        axes[1, j].set_ylim(-bound, bound); axes[1, j].set_xlabel("Individual repeated frame (no averaging)")
        axes[1, j].set_ylabel("UNCLIPPED head action / rad"); axes[1, j].grid(alpha=.2)
    axes[1, 0].legend(frameon=False, fontsize=6)
    fig.colorbar(im, ax=list(axes[0]), label="Raw ADU or transformed code; FIXED 0..255", fraction=.02)
    fig.suptitle("Predeclared example: FIRST catalog TEST input, repeat 0 | NOT REALTIME / NOT NEW DATA")
    fig.subplots_adjust(hspace=.35, top=.86, bottom=.12, wspace=.35)
    reporting.save_figure(fig, out/"figures/02_paired_example", 180); plt.close(fig)


def assess_preprocessing(head_path, output, *, expected_head_sha256):
    """Three fixed arms, original trajectory splits, no current-hardware claim.

    Hashes original dependencies (including fixture files) but never decodes a
    Torch fixture or constructs optics/Env. Only the matched old 320 CCD raws
    are decoded. The legacy audit and this reader independently recompute raw8.
    """
    out = Path(output).resolve()
    if out.parent != (LAB/"experiments").resolve():
        raise ValueError("NEW direct lab experiments child required")
    out.mkdir(exist_ok=False)
    for name in ("provenance", "derived", "figures"):
        (out/name).mkdir()
    ledger = {}
    def safe(path):
        p = Path(path)
        if not p.is_absolute() or ".." in p.parts or not p.resolve().is_relative_to(WORKSPACE.resolve()):
            raise ValueError("explicit source/asset path escapes workspace")
        return p.resolve()
    def remember(path, digest, expected=None):
        p = safe(path); name = str(p)
        if (expected is not None and digest != expected) or (name in ledger and ledger[name] != digest):
            raise ValueError("input SHA mismatch or changed: " + name)
        ledger[name] = digest
        return p
    def pin(path, expected=None):
        p = safe(path); return remember(p, storage.sha256(p), expected)
    def read(path, expected=None):
        p = safe(path); data = p.read_bytes()
        remember(p, hashlib.sha256(data).hexdigest(), expected)
        return data
    def load(path, expected=None): return json.loads(read(path, expected))
    def snapshot(name, data):
        with (out/"provenance"/name).open("xb") as stream: stream.write(data)
    def finish_manifest(complete):
        files = {str(p.relative_to(out)): storage.sha256(p) for p in sorted(out.rglob("*"))
                 if p.is_file() and p != out/"MANIFEST.json"}
        storage.write_json_new(out/"MANIFEST.json", dict(schema=SCHEMA+"-archive", complete=complete,
            file_sha256=files, manifest_excludes_itself=True, deployment_eligible=False))
    try:
        if (not isinstance(expected_head_sha256, str) or len(expected_head_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_head_sha256)):
            raise ValueError("explicit lowercase 64-character original HEAD SHA256 required")
        hp = safe(head_path); head = load(hp, expected_head_sha256)
        if (head.get("schema") != "real-single-frame-linear18-v22" or head.get("parameter_count") != 18
                or head.get("CCD_history") != 0 or head.get("real_hardware_features") is not True
                or head.get("phase_mode") != "learned"):
            raise ValueError("explicit original matched real raw18 HEAD required")
        bp = safe(head["bundle"]); data = load(bp, head["bundle_sha256"])
        capture = load(hp.parent/"capture.json", head["capture_sha256"])
        protocol = load(hp.parent/"PROSPECTIVE_PROTOCOL.json", head["prospective_protocol_sha256"])
        if (protocol["bundle"] != str(bp) or protocol["bundle_sha256"] != head["bundle_sha256"]
                or protocol["repeats"] != 8 or len(data["cases"]) != 40
                or len(capture["inputs"]) != 40 or data.get("input_frame_stride") != 3):
            raise ValueError("matched 40 inputs x 8 frames / stride3 protocol required")
        for mapping in (head["source_sha256"], data["bound_sha256"], protocol["sources"]):
            for path, sha in mapping.items(): pin(path, sha)
        contract = load(data["contract"])
        for path, sha in contract["evidence_sha256"].items(): pin(path, sha)
        runtime = load(contract["runtime_path"])
        for mapping in (runtime["sources"], runtime["source_configs"]):
            for path, sha in mapping.items(): pin(path, sha)
        for key in ("contract", "native_weights", "SLM_RAW_GRAY_NO_LUT", "model_xml"):
            path = safe(data[key])
            if str(path) not in data["bound_sha256"]: raise ValueError("unbound bundle asset " + key)
        for case in data["cases"]:
            for key in ("raw96", "dmd"):
                path = safe(case[key])
                if str(path) not in data["bound_sha256"]:
                    raise ValueError("unbound chronological input asset " + key)
        pin(data["native_weights"], head["native_weights_sha256"])
        pin(data["SLM_RAW_GRAY_NO_LUT"], head["phase_sha256"])
        seen = set(); records = []
        for row, case in zip(capture["inputs"], data["cases"]):
            if any(row[k] != case[k] for k in ("id", "trajectory", "split")) or len(row["frames"]) != 8:
                raise ValueError("input trajectory/split/order/repeat mismatch")
            for repeat, record in enumerate(row["frames"]):
                rel = Path(record["raw_relative"])
                p = (hp.parent/rel).resolve()
                if (rel.is_absolute() or ".." in rel.parts or not p.is_relative_to(hp.parent)
                        or str(p) in seen): raise ValueError("raw path escape or duplicate raw frame")
                seen.add(str(p)); pin(p, record["raw_sha256"])
                records.append(dict(input_id=case["id"], trajectory=case["trajectory"], split=case["split"],
                    repeat_index=repeat, path=str(p), sha256=record["raw_sha256"], frame=record["frame"]))
        core_path = safe(LEGACY/"live_adapt_v22/core.py")
        if head["source_sha256"].get(str(core_path)) != CORE_SHA256:
            raise ValueError("HEAD does not bind the frozen CPU fitter")
        source_files = [Path(__file__).resolve(), Path(readout.__file__).resolve(),
                        Path(reporting.__file__).resolve(), Path(storage.__file__).resolve(), core_path,
                        LEGACY/"live_deploy_v20/contract.py", LEGACY/"live_deploy_v20/protocol.py",
                        LEGACY/"live_deploy_v20/__init__.py", LEGACY/"live_manual_v17/diffraction_model.py"]
        for i, path in enumerate(source_files): snapshot("source_%02d_%s" % (i, path.name), read(path))
        for name, path in (("HEAD_original.json", hp), ("catalog_original.json", bp),
                ("capture_original.json", hp.parent/"capture.json"),
                ("capture_protocol_original.json", hp.parent/"PROSPECTIVE_PROTOCOL.json"),
                ("contract_original.json", Path(data["contract"]))): snapshot(name, read(path))
        config = readout.validate_config(dict(readout.CONFIG_REQUESTED))
        byte_config = dict(config, output="display_uint8")
        storage.write_json_new(out/"PROTOCOL.json", dict(schema=SCHEMA+"-protocol", arms=list(ARMS),
            requested_float=config, requested_uint8=byte_config, historical_inputs=40, repeated_frames_per_input=8,
            original_split_preserved=True, retrospective_seen_TEST_not_new_holdout=True,
            each_arm_ridge_selected_on_original_SELECT_only=True, test_based_arm_selection=False,
            paired_example_rule="first catalog test input; repeat 0, never chosen by error",
            raw_refit_reproduction_tolerance=dict(rtol=REPRODUCTION_RTOL, atol=REPRODUCTION_ATOL),
            independent_raw8_reduction_atol_ADU_sum=RAW8_REDUCTION_ATOL,
            no_GUI_SDK_Env_or_optical_propagation=True, source_and_asset_sha256=dict(ledger)))
        old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
        modules_before = set(sys.modules)
        try:
            for name, expected in (("diffraction_model", LEGACY/"live_manual_v17/diffraction_model.py"),
                    ("live_deploy_v20", LEGACY/"live_deploy_v20/__init__.py"),
                    ("live_deploy_v20.contract", LEGACY/"live_deploy_v20/contract.py"),
                    ("live_deploy_v20.protocol", LEGACY/"live_deploy_v20/protocol.py")):
                prior = sys.modules.get(name)
                if prior is not None and Path(prior.__file__).resolve() != expected.resolve():
                    raise ValueError("foreign legacy module already loaded: " + name)
            sys.dont_write_bytecode = True
            sys.path[:0] = [str(LEGACY), str(LEGACY/"live_manual_v17")]
            core = _load_core(core_path, read(core_path, CORE_SHA256))
            bundle = core.Bundle(bp)
            original_x, audited, _ = core.audit_capture(bundle, hp.parent)
        finally:
            sys.path[:] = old_path; sys.dont_write_bytecode = old_bytecode
        for name in set(sys.modules)-modules_before:
            module = sys.modules[name]; f = getattr(module, "__file__", None)
            if isinstance(f, str) and Path(f).is_file() and Path(f).resolve().is_relative_to(LEGACY.resolve()):
                if str(Path(f).resolve()) not in ledger: raise ValueError("unbound imported legacy source " + f)
                pin(f, ledger[str(Path(f).resolve())])
            if name.startswith(("PyQt", "PySide", "PySpin")) or name == "spin_c":
                raise ValueError("unexpected GUI/SDK import during offline assessment")
        if audited != capture or original_x.shape != (40, 8, 8):
            raise ValueError("legacy audit report/features changed")
        covered_pixels = int(np.count_nonzero(np.any(bundle.weights > 0, axis=0)))
        roi_pixels = int(np.prod(bundle.weights.shape[1:]))
        readout_identity = dict(path=data["native_weights"], sha256=head["native_weights_sha256"],
            shape=list(bundle.weights.shape), covered_pixels=covered_pixels, ROI_pixels=roi_pixels,
            coverage_fraction=covered_pixels/roi_pixels,
            full_ROI_coverage=covered_pixels == roi_pixels,
            channel_weight_sums=bundle.weights.sum((1, 2), dtype=np.float64).tolist(),
            current_BROAD8_identity_NOT_assumed=True,
            fractional_weights_preserved=True, region_mean_or_area_normalization=False)
        phase_description = ("historical quadratic_neg200mm prior (exact known SHA); learned is a legacy alias"
            if head["phase_sha256"] == NEG200_PHASE_SHA256 else
            "historical frozen mask identified ONLY by the bound path/SHA; learned is a legacy interface alias")
        labels = np.asarray([c["teacher_action_rad"] for c in bundle.cases], np.float64)
        groups = np.asarray([c["trajectory"] for c in bundle.cases])
        splits = np.asarray([c["split"] for c in bundle.cases])
        example_index = int(np.flatnonzero(splits == "test")[0]); examples = {}
        features = {name: np.empty((40, 8, 8), np.float64) for name in ARMS}
        clipping = []
        for i, record in enumerate(records):
            raw = np.load(io.BytesIO(read(record["path"], record["sha256"])), allow_pickle=False)
            if raw.dtype != np.uint8 or raw.shape != bundle.weights.shape[1:] or raw.max() >= 240:
                raise ValueError("original complete native uint8/headroom contract changed")
            arrays = dict(raw=raw, requested_float=readout.preprocess_intensity(raw, config),
                          requested_uint8=readout.preprocess_intensity(raw, byte_config))
            q, repeat = divmod(i, 8)
            for name in ARMS:
                features[name][q, repeat] = readout.integrate_pixels(arrays[name], bundle.weights)
            clipping.append(dict(input_id=record["input_id"], repeat_index=repeat,
                software_lower_rail_fraction=float(np.mean(raw <= config["black_ADU"])),
                software_upper_rail_fraction=float(np.mean(raw >= config["black_ADU"]+255/config["gain"])),
                upper_rail_raw_ADU=config["black_ADU"]+255/config["gain"],
                actual_ADC_255_fraction=float(np.mean(raw == 255)),
                uint8_rounding_to_zero_fraction=float(np.mean(arrays["requested_uint8"] == 0))))
            if q == example_index and repeat == 0: examples = {k: v.copy() for k, v in arrays.items()}
        raw8_difference = float(np.max(np.abs(features["raw"]-original_x)))
        if not np.allclose(features["raw"], original_x, rtol=0, atol=RAW8_REDUCTION_ATOL):
            raise ValueError("independent raw8 recomputation disagrees with legacy audit")
        fitted, actions, direct, scores = {}, {}, {}, {}
        for name in ARMS:
            fitted[name] = core.fit_head(features[name], labels, groups, splits, bundle.data["rules"])
            actions[name] = core.predict(fitted[name], features[name])
            direct[name] = core.predict(head, features[name])
            scores[name] = _scores(core, actions[name], labels, groups, splits)
        reproduced = _reproduction(fitted["raw"], head)
        gain_head = readout.linear_gain_head(head, config["gain"])
        gain_actions = core.predict(gain_head, config["gain"]*features["raw"])
        raw_old_actions = core.predict(head, features["raw"])
        gain_error = float(np.max(np.abs(gain_actions-raw_old_actions)))
        if not np.allclose(gain_actions, raw_old_actions, rtol=1e-10, atol=1e-10):
            raise ValueError("pure linear gain compensation failed")
        for name in ARMS:
            own = fitted[name]
            storage.write_json_new(out/"derived"/(name+"_HEAD_RESEARCH.json"), dict(
                schema="optical-lab-preprocessed-linear18-research-1", arm=name,
                feature_contract="raw_native_ADU_sums" if name == "raw" else "transformed_code_sums",
                preprocessing=None if name == "raw" else (config if name == "requested_float" else byte_config),
                weight_feature_to_rad=own["weight_raw_ADU_to_rad"], bias_rad=own["bias_rad"],
                selected_ridge=own["selected_ridge"], parameter_count=18,
                source_original_HEAD_sha256=expected_head_sha256, current_hardware_validated=False,
                deployment_eligible=False, release_approved=False, not_accepted_by_legacy_HEAD_loader=True))
        arrays = {name+"_features8": features[name] for name in ARMS}
        arrays.update({name+"_unclipped_actions_rad": actions[name] for name in ARMS})
        arrays.update({name+"_OLD_HEAD_direct_unclipped_actions_rad": direct[name] for name in ARMS})
        arrays.update(labels_rad=labels, trajectories=groups, splits=splits, pure_gain_compensated_actions_rad=gain_actions)
        with (out/"derived/all_features_and_actions.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        storage.write_json_new(out/"derived/frames.json", records)
        storage.write_json_new(out/"derived/software_clipping.json", clipping)
        result = dict(schema=SCHEMA, complete=True, analysis_complete=True,
            historical_unique_inputs=40, historical_raw_frames=320,
            retrospective_seen_TEST_not_new_holdout=True, repeated_frames_not_independent_trajectories=True,
            no_temporal_average_or_new_optical_model=True, current_alignment_not_validated=True,
            current_ROI_not_substituted_for_historical_ROI=True, source_HEAD=str(hp), source_HEAD_sha256=expected_head_sha256,
            historical_ROI=contract["roi"], historical_settings=contract["settings"],
            historical_readout=readout_identity,
            historical_phase=dict(path=data["SLM_RAW_GRAY_NO_LUT"], sha256=head["phase_sha256"],
                original_phase_mode_alias=head["phase_mode"],
                description=phase_description, not_new_learned_phase_advantage_or_current_SLM_validation=True),
            independent_raw8_reduction=dict(max_abs_difference_ADU_sum=raw8_difference,
                atol_ADU_sum=RAW8_REDUCTION_ATOL, rtol=0., changes_no_physical_acceptance_gate=True),
            raw_refit_reproduction=reproduced,
            arms=scores, OLD_HEAD_direct_negative_control={name: _scores(core, direct[name], labels, groups, splits) for name in ARMS},
            pure_linear_gain_counterfactual=dict(max_abs_action_difference_rad=gain_error,
                requested_clip_not_supported=True, photon_SNR_unchanged=True),
            software_rails=dict(lower_fraction=float(np.mean([r["software_lower_rail_fraction"] for r in clipping])),
                upper_fraction=float(np.mean([r["software_upper_rail_fraction"] for r in clipping])),
                no_pixels_or_inputs_filtered_out=True, threshold_not_measured_dark_calibration=True),
            no_test_based_arm_winner_selected=True, hardware_accessed=False, deployment_eligible=False,
            production_qualified=False, release_approved=False, control_release_approved=False,
            software=dict(numpy=np.__version__))
        _figures(out, scores, actions, direct, labels, groups, splits, examples, example_index)
        lines = ["# E011 历史匹配CCD前处理／18参数头对照", "",
            "40个历史输入×8个单帧，共320帧。三臂使用原轨迹划分；原TEST已见，不能称新holdout。",
            "原始文件只读且逐SHA核验，未接GUI／SDK，未运行仿真闭环、改SLM或当前部署。历史ROI与当前重对齐光路不能混用。", "",
            "历史mask身份：`"+data["SLM_RAW_GRAY_NO_LUT"]+"`，SHA `"+head["phase_sha256"]+"`；"+phase_description+"。这不是学习相位优势或当前新SLM验证。", "",
            "历史ROI：`"+json.dumps(contract["roi"], sort_keys=True)+"`，SLM位置`("+
                str(contract["settings"]["x"])+","+str(contract["settings"]["y"])+")`；不能套用当前重新对齐的ROI／位置。", "",
            "历史八区：`"+data["native_weights"]+"`，SHA `"+head["native_weights_sha256"]+"`；覆盖"+
                str(covered_pixels)+"/"+str(roi_pixels)+"像素（"+format(100*covered_pixels/roi_pixels,".3f")+
                "%），不是默认全ROI积分，也不借用当前BROAD8身份；原分数权重与八区顺序不变。", "",
            "固定请求：9.5×、黑偏置10.5 ADU、γ1；float保留小数，uint8另行舍入。两者都有软件上下截断，不是提高光子SNR。", "",
            "| Arm | 已见TEST MAE/rad | 已见TEST MSE/rad² | 同图单帧动作RMS/rad |", "|---|---:|---:|---:|"]
        for name in ARMS:
            m = scores[name]["by_split"]["test"]
            lines.append(f"| {name} | {m['MAE_rad']:.8g} | {m['MSE_rad2']:.8g} | {m['within_input_action_RMS_rad']:.8g} |")
        lines += ["", "raw重拟合已按预声明容差复现原权重／偏置／指标。所有320×8特征、未限幅动作、旧头误接新特征负对照与软件截断率均保存；没有按TEST挑新默认。",
            "纯增益并补偿W的动作应不变；这不适用于阈值／截断／γ。像素变干净或积分数值变大不是可部署／稳定性或SNR证据。", "",
            "![全部轨迹误差与单帧抖动](figures/01_action_error_and_jitter.png)", "",
            "![预声明的配对示例](figures/02_paired_example.png)", "",
            "新头使用独立研究schema，无旧pilot资格。最终所有原数据、模型、源码及配置均作SHA尾检。", ""]
        with (out/"总结.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(lines))
        for path, sha in ledger.items():
            if storage.sha256(path) != sha: raise ValueError("input/source changed before sealing: " + path)
        storage.write_json_new(out/"INPUT_SHA256.json", ledger)
        storage.write_json_new(out/"RESULT.json", result)
        finish_manifest(True)
        return result
    except Exception as exc:
        storage.write_json_new(out/"FAILED.json", dict(complete=False, error=repr(exc), hardware_accessed=False,
            deployment_eligible=False, input_sha256=ledger))
        finish_manifest(False)
        raise
