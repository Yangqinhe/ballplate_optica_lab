"""Sealed CAL-only physical sensitivity audit: no fitting or device access.

Single-frame repeat covariance is descriptive (eight serial samples), never
divided by N or inverted. Model rank, response/noise ratios and residual
alignment do not establish physical identifiability or release a deployment.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from . import observability_assessment as observed
from .storage import sha256, write_json_new
from .reporting import plotting, save_figure

SCHEMA = "optical-lab-observed-physical-sensitivity-audit-1"
PARAMETERS = ("effective_input_defocus_m", "phase_response_multiplier",
              "global_ADU_gain_multiplier", "global_latent_ADU_bias")
STEPS = np.array([.0005, .03, .05, .005])
MASKS = ("learned", "lens", "flat")
GRID = ("REF", "X_PLUS", "X_MINUS", "Y_PLUS", "Y_MINUS", "VX_PLUS", "VX_MINUS", "VY_PLUS", "VY_MINUS")
FLOOR = 1.0  # integrated-ADU numerical reporting floor; NOT a physical criterion.


def _vector(value):
    v = np.asarray(value, float)
    if v.shape != (8,) or not np.isfinite(v).all():
        raise ValueError("finite native-eight vector required")
    return v


def _covariance(value):
    c = np.asarray(value, float)
    if (c.shape != (8, 8) or not np.isfinite(c).all()
            or not np.allclose(c, c.T, rtol=0, atol=1e-7)
            or np.linalg.eigvalsh((c+c.T)/2).min() < -1e-6):
        raise ValueError("finite symmetric nonnegative sample covariance required")
    return c


def _ratio(numerator, denominator):
    return float(numerator/denominator) if denominator > FLOOR else None


def vector_error(actual, predicted):
    a, p = _vector(actual), _vector(predicted)
    an, pn = np.linalg.norm(a), np.linalg.norm(p)
    return dict(L2_ADU_sum=float(np.linalg.norm(a-p)),
        relative_L2_to_prediction=_ratio(np.linalg.norm(a-p), pn),
        cosine=(float(np.dot(a, p)/(an*pn)) if min(an, pn) > FLOOR else None),
        actual_L2_ADU_sum=float(an), predicted_L2_ADU_sum=float(pn))


def sensitivity_scale(effect, covariance, references):
    """Project onto a fixed MODEL direction, without empirical whitening."""
    d, c = _vector(effect), _covariance(covariance)
    length = float(np.linalg.norm(d)); trace_scale = float(np.sqrt(max(0., np.trace(c))))
    if length <= FLOOR:
        return dict(step_effect_L2_ADU_sum=length, projected_single_frame_scale_ADU=None,
            single_frame_full8_scale_ADU=trace_scale, projected_REF_span_ADU=None,
            step_to_projected_single_frame=None, step_to_REF_span=None,
            effect_is_too_small_for_direction_reporting=True)
    unit = d/length
    sigma = float(np.sqrt(max(0., unit@c@unit)))
    ref = np.asarray([_vector(x) for x in references])
    span = float(np.ptp(ref@unit)) if len(ref) else None
    return dict(step_effect_L2_ADU_sum=length, projected_single_frame_scale_ADU=sigma,
        single_frame_full8_scale_ADU=trace_scale, projected_REF_span_ADU=span,
        step_to_projected_single_frame=_ratio(length, sigma),
        step_to_REF_span=None if span is None else _ratio(length, span),
        effect_is_too_small_for_direction_reporting=False)


def _audit_observations(reader, root, manifest_sha):
    # Read the external seal and header before opening any derived arrays.
    r = reader.js(root/"RESULT.json", role="observability_complete_header")
    if (r.get("schema") != observed.SCHEMA or r.get("analysis_complete") is not True
            or r.get("all_planned_captures_complete") is not True
            or (r.get("planned_batches"), r.get("planned_frames"), r.get("valid_frames")) != (8, 704, 704)
            or len(r.get("batches", [])) != 8 or len(r.get("conditions", [])) != 4
            or any(c.get("complete_pair") is not True for c in r["conditions"])):
        raise ValueError("sealed all-eight/704 COMPLETE observability analysis required; no partial audit")
    reader.archive(root, manifest_sha)
    if len({b["batch_id"] for b in r["batches"]}) != 8:
        raise ValueError("eight unique complete batch identities required")
    keys = set()
    for b in r["batches"]:
        key = (b["radius"], b["mask_key"], b["axis_family"]); keys.add(key)
        expected = observed.observability_states(b["axis_family"])
        if (b.get("status") != "COMPLETE" or b.get("valid_frames") != 88
                or b.get("planned_frames") != 88 or len(b["states"]) != 11
                or [s["id"] for s in b["states"]] != [s["id"] for s in expected]):
            raise ValueError("complete unique11-state/88-frame batch required")
        for s, spec in zip(b["states"], expected):
            if (s.get("numeric_eligible") is not True or s.get("valid_frames") != 8
                    or s.get("kind") != spec["kind"] or s.get("input_key") != spec["input_key"]):
                raise ValueError("all declared stationary states must be eight-frame eligible")
            _vector(s["native8_mean_ADU"]); _covariance(s["native8_single_frame_covariance_ADU2"])
    expected = {(r, m, f) for r, m in ((0, "learned"), (2, "learned"), (2, "lens"), (2, "flat")) for f in ("POS", "VEL")}
    if keys != expected:
        raise ValueError("exact predeclared encoding/mask/POS/VEL batch matrix required")
    return r


def _predictions(reader, root, manifest_sha, obs):
    reader.archive(root, manifest_sha)
    p = reader.js(root/"RESULT.json")
    if (p.get("schema") != "optical-lab-global-physics-sensitivity-1" or p.get("complete") is not True
            or p.get("hardware_accessed") is not False or p.get("bank_sha256") != obs["bank"]["sha256"]
            or p.get("nominal_prediction_count") != 108 or p.get("sensitivity_command_count") != 27
            or obs["predictions"] != dict(path=str(root), sha256=manifest_sha)):
        raise ValueError("exact completed frozen prediction/bank identity required")
    means = {}
    for row in p["rows"]:
        key = (row["model"], row["input_id"], row["mask_id"])
        if key in means:
            raise ValueError("duplicate model/input/mask prediction")
        ref = row["files"]["native8"]
        means[key] = _vector(reader.npy(reader.local(root, ref["path"]), ref["sha256"]))
    expected = {(m, f"R{r}_{i}", k) for m in ("M0_REFERENCE", "M1_RADIAL") for r in (0, 2) for i in GRID for k in MASKS}
    if set(means) != expected:
        raise ValueError("exact independent108 M0/M1 nominal identities required")
    jac = {}
    for row in p["sensitivities"]:
        key = (row["input_id"], row["mask_id"])
        if key in jac or row.get("parent") != "M0_REFERENCE" or row.get("zero_perturbation_exact_parent") is not True:
            raise ValueError("unique exact-parent R2 sensitivity required")
        ref = row["jacobian8"]; j = reader.npy(reader.local(root, ref["path"]), ref["sha256"])
        if j.shape != (8, 4) or not np.isfinite(j).all():
            raise ValueError("finite physical J8x4 required")
        jac[key] = np.asarray(j, float)
    if set(jac) != {(f"R2_{i}", k) for i in GRID for k in MASKS}:
        raise ValueError("exact27 R2 commands required; no extrapolated R0 sensitivity")
    if p["global_physical_parameter_sensitivity"]["parameter_names"] != list(PARAMETERS) or not np.array_equal(p["global_physical_parameter_sensitivity"]["finite_difference_steps"], STEPS):
        raise ValueError("fixed physical parameter units/steps required")
    return means, jac


def assess_observed(observability_report, output, report_manifest_sha256,
                    predictions_dir, predictions_manifest_sha256):
    """Consume complete SEALED offline reports only; never open capture RAWs."""
    root = Path(observability_report).resolve(strict=True); pred = Path(predictions_dir).resolve(strict=True)
    out = Path(output).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError("NEW audit directory required")
    if any(out.is_relative_to(p) for p in (root, pred)):
        raise ValueError("output cannot modify an input archive")
    reader = observed._Reader(); created = False
    try:
        # Seal first: header is accepted only under the externally fixed bytes.
        reader.js(root/"MANIFEST.json", report_manifest_sha256, "external_observation_seal")
        obs = _audit_observations(reader, root, report_manifest_sha256)
        means, jac = _predictions(reader, pred, predictions_manifest_sha256, obs)
        out.mkdir(); created = True; (out/"provenance").mkdir(); (out/"figures").mkdir()
        for n, path in enumerate((Path(__file__), Path(observed.__file__), Path(__file__).with_name("protocol.py"), Path(__file__).with_name("storage.py"), Path(__file__).with_name("reporting.py"))):
            blob = reader.read(path, role="audit_implementation"); (out/"provenance"/f"{n:02d}_{path.name}").write_bytes(blob)
        states, contrasts, sensitivity = [], [], []
        for batch in obs["batches"]:
            saved = {s["id"]: s for s in batch["states"]}; mask = batch["mask_key"]; radius = batch["radius"]
            refs = [s["native8_mean_ADU"] for s in batch["states"] if s["id"].startswith("REF")]
            for s in batch["states"]:
                mean = _vector(s["native8_mean_ADU"]); cov = _covariance(s["native8_single_frame_covariance_ADU2"])
                item = dict(batch_id=batch["batch_id"], state_id=s["id"], mask_id=mask, radius=radius,
                    native8_mean_ADU=mean.tolist(), native8_single_frame_covariance_ADU2=cov.tolist(),
                    single_frame_repeat_scale_ADU=float(np.sqrt(max(0., np.trace(cov)))), models={})
                if s["kind"] == "BG":
                    item["model_comparison_NA_reason"] = "optical all-zero BG has no saved nominal prediction; not a physical shutter dark"
                else:
                    ik = f"R{radius}_{s['input_key']}"
                    item["input_id"] = ik
                    for model in ("M0_REFERENCE", "M1_RADIAL"):
                        item["models"][model] = vector_error(mean, means[(model, ik, mask)])
                    if radius == 2:
                        for j, param in enumerate(PARAMETERS):
                            sensitivity.append(dict(batch_id=batch["batch_id"], state_id=s["id"], input_id=ik, mask_id=mask,
                                kind="absolute", parameter=param, step=float(STEPS[j]),
                                **sensitivity_scale(jac[(ik, mask)][:, j]*STEPS[j], cov, refs)))
                states.append(item)
            axes = ("X", "Y") if batch["axis_family"] == "POS" else ("VX", "VY")
            for ax, axis in enumerate(axes):
                for sign, suffix, a, b in ((1, "PLUS", 2*ax, 2*ax+1), (-1, "MINUS", 2*ax+1, 2*ax+2)):
                    s = saved[axis+"_"+suffix]; before, after = saved["REF"+str(a)], saved["REF"+str(b)]
                    delta = observed.bracketed_response(s["native8_mean_ADU"], before["native8_mean_ADU"], after["native8_mean_ADU"],
                        s["effective_host_time_relative_ns"], before["effective_host_time_relative_ns"], after["effective_host_time_relative_ns"])
                    if delta is None:
                        raise ValueError("complete observation has invalid host bracket chronology")
                    ik = f"R{radius}_{axis}_{suffix}"; rk = f"R{radius}_REF"; weights = delta["weights"]
                    covs = [_covariance(q["native8_single_frame_covariance_ADU2"]) for q in (s, before, after)]
                    repeat_bound = sum(w*np.sqrt(max(0., np.trace(c))) for w, c in zip((1., *weights), covs))
                    row = dict(batch_id=batch["batch_id"], state_id=s["id"], input_id=ik, mask_id=mask, radius=radius,
                        observed_REF_corrected_native8=delta["response"].tolist(), reference_weights=weights,
                        conservative_single_frame_triangle_repeat_scale_ADU=float(repeat_bound), models={})
                    for model in ("M0_REFERENCE", "M1_RADIAL"):
                        row["models"][model] = vector_error(delta["response"], means[(model, ik, mask)]-means[(model, rk, mask)])
                    contrasts.append(row)
                    if radius == 2:
                        for j, param in enumerate(PARAMETERS):
                            effect = (jac[(ik, mask)]-jac[(rk, mask)])[:, j]*STEPS[j]
                            scale = sensitivity_scale(effect, covs[0], refs)
                            if np.linalg.norm(effect) > FLOOR:
                                unit = effect/np.linalg.norm(effect)
                                triangle = sum(w*np.sqrt(max(0., unit@c@unit)) for w, c in zip((1., *weights), covs))
                            else:
                                triangle = None
                            scale.update(projected_single_frame_triangle_scale_ADU=None if triangle is None else float(triangle),
                                step_to_projected_single_frame_triangle=None if triangle is None else _ratio(np.linalg.norm(effect), triangle))
                            sensitivity.append(dict(batch_id=batch["batch_id"], state_id=s["id"], input_id=ik, mask_id=mask,
                                kind="REF_contrast", parameter=param, step=float(STEPS[j]), **scale))
        if len(states) != 88 or len(contrasts) != 32 or len(sensitivity) != (54+24)*4:
            raise ValueError("complete88 states/32 contrasts/312 sensitivity comparisons required")
        summary = []
        for kind in ("absolute", "REF_contrast"):
            for param in PARAMETERS:
                rows = [s for s in sensitivity if s["kind"] == kind and s["parameter"] == param]
                field = "step_to_projected_single_frame" if kind == "absolute" else "step_to_projected_single_frame_triangle"
                # Equal unique command weight; repeated REF visits do not create new inputs.
                grouped = {}
                for s in rows:
                    if s.get(field) is not None: grouped.setdefault((s["input_id"], s["mask_id"]), []).append(s[field])
                ratios = [float(np.median(x)) for x in grouped.values()]
                summary.append(dict(kind=kind, parameter=param, unique_commands_with_defined_ratio=len(ratios),
                    median_step_to_single_frame_ratio=None if not ratios else float(np.median(ratios)),
                    fraction_ratio_above_one=None if not ratios else float(np.mean(np.array(ratios)>1)),
                    above_one_is_descriptive_NOT_a_fit_or_qualification_gate=True))
        plt = plotting(); fig, axes = plt.subplots(3, 1, figsize=(13, 10))
        rows = [s for s in states if s["models"]]
        for model, color in (("M0_REFERENCE", "#0072B2"), ("M1_RADIAL", "#D55E00")):
            axes[0].plot([s["models"][model]["L2_ADU_sum"] for s in rows], ".", color=color, label=model)
            axes[1].plot([s["models"][model]["relative_L2_to_prediction"] for s in contrasts], ".", color=color)
        axes[0].plot([s["single_frame_repeat_scale_ADU"] for s in rows], color="#222222", label="single-frame repeat scale")
        axes[0].set_ylabel("Absolute native8 residual / ADU sum\nraw baseline retained, NOT optical efficiency")
        axes[1].set_ylabel("REF-contrast relative L2\nsmall denominator is NA")
        for kind, marker in (("absolute", "o"), ("REF_contrast", "x")):
            vals = [next(s for s in summary if s["kind"]==kind and s["parameter"]==p)["median_step_to_single_frame_ratio"] for p in PARAMETERS]
            axes[2].plot(range(4), [np.nan if v is None else v for v in vals], marker+"-", label=kind)
        axes[2].set_xticks(range(4), ["z .5mm", "phase .03", "gain .05", "bias .005ADU"])
        axes[2].set_ylabel("Median model step / projected single-frame scale\nNOT noise whitening / NOT confidence")
        axes[0].legend(frameon=False); axes[2].legend(frameon=False)
        fig.suptitle("E005 sealed CAL replay / fixed models / no fitted parameters / unqualified")
        fig.tight_layout(); save_figure(fig, out/"figures/01_observed_and_model_sensitivity", 180); plt.close(fig)
        result = dict(schema=SCHEMA, analysis_complete=True, observability_report=dict(path=str(root), sha256=report_manifest_sha256),
            predictions=dict(path=str(pred), sha256=predictions_manifest_sha256), states=states, contrasts=contrasts,
            physical_step_comparisons=sensitivity, summary=summary, planned_frames=704, valid_frames=704,
            parameters_fitted=False, new_SLM_trained=False, new_head_trained=False, hardware_accessed=False,
            covariance_inverse_used=False, sample_count_division_used=False, host_time_is_NOT_verified_exposure=True,
            reporting_norm_floor_ADU_sum=FLOOR, reporting_floor_is_NOT_a_physical_release_criterion=True,
            interpretation=["absolute raw8 is dominated by retained background; REF contrasts are complementary, not replacement actor inputs",
                "triangle repeat scales assume no independence; covariance estimates are only eight serial samples",
                "bias contrast usually cancels; optical all-zero BG is not shutter dark, so four-parameter fitting is not automatically justified",
                "M0/M1 are previously calibrated unqualified hypotheses, not error-free ideal models",
                "fixed global physical parameters only; no per-image gain, affine, online error network or parameter estimate"],
            recommendation="仅供前瞻小范围共享参数研究拟合的人工审阅：先排除参考漂移；优先z/phase/gain，bias无独立暗场锚定时固定；以未参与拟合的输入/掩膜验证。比值或模型秩不能自动放行。",
            all_data_CAL_development_NOT_blind_validation=True, human_review_required_before_any_fit=True,
            production_qualified=False, control_release_approved=False, physics_identification_qualification="UNKNOWN")
        (out/"总结.md").write_text("# 实测物参可辨识性审计（未拟合）\n\n仅使用完整704帧分析归档和先冻结预测；不重开CCD原始帧、不访问设备。保存88个状态的8区均值/协方差、32个夹REF差分和312项物参步长对比。\n\n绝对8积分保留大背景，不能只凭其相对误差小称模型吻合。主差分用同host时钟夹REF（非已验证曝光时刻）；单帧协方差不除N、不求逆。差分重复尺度用三项标准差的三角上界，不假定共享REF独立。\n\nz/phase/gain/bias步长只是研究扰动，不是实测置信区间。尤其bias在REF差分中抵消，光学全0也不等于物理遮光暗场；模型rank4不是四参数可拟合的证明。参考漂移与参数方向相似也不能证明因果。\n\n"+result["recommendation"]+"\n\n全部是CAL开发数据，没有新训练/估参/SLM/head/部署放行。\n\n![统一离线诊断](figures/01_observed_and_model_sensitivity.png)\n", encoding="utf8")
        reader.tail(); write_json_new(out/"source_sha256.json", list(reader.sources.values())); write_json_new(out/"RESULT.json", result)
        write_json_new(out/"MANIFEST.json", dict(schema=SCHEMA, analysis_complete=True,
            file_sha256={str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file()}, manifest_excludes_itself=True))
        return result
    except Exception as exc:
        if created: write_json_new(out/"FAILED.json", dict(schema=SCHEMA, analysis_complete=False, error=repr(exc), partial_files_preserved=True))
        raise
