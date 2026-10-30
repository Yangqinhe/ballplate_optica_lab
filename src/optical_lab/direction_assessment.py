"""Pure fixed-8-feature analysis of the predeclared two-visit velocity study.

No capture, source discovery, model inference, head fitting, SDK or GUI. The
caller must bind sealed raw bytes, current ROI/readout and fixed manual lens.
Means, REF interpolation and unit-total normalization are OFFLINE diagnostics,
not actor inputs. Eight repeats are one episode, not eight independent states.
"""
from pathlib import Path

import numpy as np


STATE_IDS = (
    "BG0", "A_REF0", "A_VX_PLUS", "A_VX_MINUS", "A_REF1",
    "A_VY_MINUS", "A_VY_PLUS", "A_REF2", "BG1", "B_REF0",
    "B_VY_PLUS", "B_VY_MINUS", "B_REF1", "B_VX_MINUS", "B_VX_PLUS",
    "B_REF2", "BG2",
)
REPEATS = 8
VELOCITY_STEP_M_S = .12
TARGET_TAU_M_S = .02
CRITERIA = dict(minimum_effect_over_single_frame_difference_scale=3.,
                minimum_cross_visit_cosine=.8, maximum_relative_cross_visit_change=.5,
                minimum_effect_over_REF_curvature=3.)
PAIRS = {
    "VX": {"A": ("A_VX_PLUS", "A_VX_MINUS", "A_REF0", "A_REF1"),
           "B": ("B_VX_PLUS", "B_VX_MINUS", "B_REF1", "B_REF2")},
    "VY": {"A": ("A_VY_PLUS", "A_VY_MINUS", "A_REF1", "A_REF2"),
           "B": ("B_VY_PLUS", "B_VY_MINUS", "B_REF0", "B_REF1")},
}


def _ratio(value, denominator):
    return float(value / denominator) if denominator > 0 else None


def _cosine(a, b):
    return _ratio(float(np.dot(a, b)), float(np.linalg.norm(a)*np.linalg.norm(b)))


def bracketed_delta(value, before, after, time, before_time, after_time):
    """Same-clock linear REF interpolation, strictly inside, never extrapolate."""
    if (not all(np.isfinite(t) for t in (time, before_time, after_time))
            or not before_time < time < after_time):
        return None
    weight = float((time-before_time)/(after_time-before_time))
    delta = np.asarray(value, np.float64)-(1-weight)*np.asarray(before, np.float64)-weight*np.asarray(after, np.float64)
    return dict(delta=delta.tolist(), weights=[1-weight, weight])


def _prepare(states):
    if [s.get("id") for s in states] != list(STATE_IDS):
        raise ValueError("all17 predeclared state slots in exact order required, including partial/empty slots")
    prepared = {}
    origin = last_time = None
    for state in states:
        values = state.get("features8")
        values = np.empty((0, 8), np.float64) if values is None or np.size(values) == 0 else np.asarray(values)
        times = np.asarray(state.get("host_times", []))
        if (values.ndim != 2 or values.shape[1] != 8 or values.shape[0] > REPEATS
                or values.dtype.kind not in "uif" or not np.isfinite(values).all()
                or times.ndim != 1 or len(times) != len(values) or times.dtype.kind not in "uif"
                or not np.isfinite(times).all()):
            raise ValueError("at most8 finite numeric feature8 rows with matching same-clock host times required")
        native_times = times.tolist()
        if native_times:
            if origin is None:
                origin = native_times[0]
            if (any(b <= a for a, b in zip(native_times, native_times[1:]))
                    or (last_time is not None and native_times[0] <= last_time)):
                raise ValueError("host times must advance strictly within and across all predeclared episodes")
            last_time = native_times[-1]
        # Subtract Python integers before float conversion: do not lose ns at epoch scale.
        relative = np.asarray([t-origin for t in native_times], np.float64)
        x = values.astype(np.float64, copy=True)
        mean = x.mean(0) if len(x) else None
        covariance = np.cov(x, rowvar=False, ddof=1) if len(x) > 1 else None
        if mean is not None and not np.isfinite(mean).all():
            raise ValueError("nonfinite feature mean")
        if covariance is not None and not np.isfinite(covariance).all():
            raise ValueError("nonfinite feature covariance")
        prepared[state["id"]] = dict(id=state["id"], x=x, count=len(x), complete=len(x) == REPEATS,
            mean=mean, covariance=covariance, time=float(relative.mean()) if len(x) else None,
            first_host_time=native_times[0] if len(x) else None,
            last_host_time=native_times[-1] if len(x) else None)
    return prepared, origin


def _public_state(row):
    return dict(id=row["id"], status="COMPLETE" if row["complete"] else "PARTIAL_OR_NOT_EXECUTED",
        expected_frames=REPEATS, frames=row["count"], mean8=None if row["mean"] is None else row["mean"].tolist(),
        covariance8_ddof1=None if row["covariance"] is None else row["covariance"].tolist(),
        single_frame_sqrt_trace_cov=None if row["covariance"] is None else float(np.sqrt(max(0., np.trace(row["covariance"])))),
        mean_host_time_offset=row["time"], first_host_time=row["first_host_time"], last_host_time=row["last_host_time"])


def _null_curvature(rows, visit):
    selected = [rows[visit+"_REF"+str(i)] for i in range(3)]
    if not all(row["complete"] for row in selected):
        return dict(complete=False, vector8=None, norm=None)
    before, middle, after = selected
    result = bracketed_delta(middle["mean"], before["mean"], after["mean"],
                             middle["time"], before["time"], after["time"])
    if result is None:
        return dict(complete=False, vector8=None, norm=None)
    return dict(complete=True, vector8=result["delta"], norm=float(np.linalg.norm(result["delta"])),
                interpolation_weights=result["weights"], offline_only=True)


def _visit(rows, ids):
    plus, minus, before, after = (rows[key] for key in ids)
    if not all(row["complete"] for row in (plus, minus, before, after)):
        return dict(complete=False, raw=None, bracketed_REF=None, reason="at least one required episode has fewer than8 frames")
    difference = plus["mean"]-minus["mean"]
    noise = float(np.sqrt(max(0., np.trace(plus["covariance"]+minus["covariance"]))))
    response = []
    for row in (plus, minus):
        result = bracketed_delta(row["mean"], before["mean"], after["mean"],
                                 row["time"], before["time"], after["time"])
        if result is None:
            return dict(complete=False, raw=None, bracketed_REF=None, reason="REF bracket invalid; no extrapolation")
        response.append(result)
    corrected = np.asarray(response[0]["delta"])-np.asarray(response[1]["delta"])
    def summary(vector):
        magnitude = float(np.linalg.norm(vector))
        return dict(vector8=vector.tolist(), norm=magnitude,
            effect_over_single_frame_difference_scale=_ratio(magnitude, noise))
    return dict(complete=True, raw=summary(difference), bracketed_REF=summary(corrected),
        single_frame_difference_scale=noise, sample_mean_standard_error_NOT_used=True,
        REF_weights_plus=response[0]["weights"], REF_weights_minus=response[1]["weights"],
        REF_correction_is_offline_NOT_actor=True)


def _cross_visit(a, b):
    if not a["complete"] or not b["complete"]:
        return dict(complete=False, raw=None, bracketed_REF=None)
    result = dict(complete=True)
    for name in ("raw", "bracketed_REF"):
        av, bv = np.asarray(a[name]["vector8"]), np.asarray(b[name]["vector8"])
        result[name] = dict(cosine=_cosine(av, bv),
            relative_change=_ratio(float(np.linalg.norm(bv-av)), (a[name]["norm"]+b[name]["norm"])/2.))
    return result


def _classification(rows, ids_a, ids_b, method):
    selected = [rows[key] for key in (*ids_a[:2], *ids_b[:2])]
    if not all(row["complete"] for row in selected):
        return dict(complete=False, accuracy=None, reason="incomplete A/B probe episodes")
    samples = [row["x"] for row in selected]
    if method == "sum8":
        samples = [x.sum(1, keepdims=True) for x in samples]
    elif method == "unit_total8":
        if any((x < 0).any() or (x.sum(1) <= 0).any() for x in samples):
            return dict(complete=False, accuracy=None, reason="unit-total diagnostic requires nonnegative rows with positive totals")
        samples = [x/x.sum(1, keepdims=True) for x in samples]
    plus, minus = samples[0].mean(0), samples[1].mean(0)
    direction = plus-minus
    tested = np.concatenate(samples[2:])
    labels = np.repeat([1, -1], REPEATS)
    scores = (tested-(plus+minus)/2.) @ direction
    predictions = np.sign(scores).astype(int)
    identifiable = bool(np.linalg.norm(direction) > 0)
    return dict(complete=True, identifiable=identifiable,
        accuracy=float(np.mean(predictions == labels)) if identifiable else None,
        correct=int(np.count_nonzero(predictions == labels)), tested_frames=len(labels),
        test_episodes=2, independent_test_samples_NOT_claimed=True,
        tie_count=int(np.count_nonzero(predictions == 0)), predictions=predictions.tolist(),
        expected_labels=labels.tolist(), scores=scores.tolist(),
        prototype_plus=plus.tolist(), prototype_minus=minus.tolist(),
        fit_visit="A", evaluated_visit="B", rule="predeclared Euclidean nearest centroid; ties not correct",
        no_REF_or_BG_subtraction=True, normalization_is_diagnostic_NOT_actor=method == "unit_total8")


def assess_feature_states(states):
    """Analyze exact17 slots of {id, features8:r×8, host_times:r}, with r≤8.

    Host times share one caller-declared clock/unit. Integer Unix-ns values are
    supported without epoch float rounding. Partial rows stay visible; pairs
    lacking any required full episode are NA, not shortened-state results.
    """
    rows, origin = _prepare(states)
    nulls = {visit: _null_curvature(rows, visit) for visit in ("A", "B")}
    null_norm = max(q["norm"] for q in nulls.values()) if all(q["complete"] for q in nulls.values()) else None
    axes = {}
    for axis, pair in PAIRS.items():
        visits = {v: _visit(rows, ids) for v, ids in pair.items()}
        cross = _cross_visit(visits["A"], visits["B"])
        criteria = {}
        for name in ("raw", "bracketed_REF"):
            if not cross["complete"]:
                criteria[name] = None
                continue
            ratios = [visits[v][name]["effect_over_single_frame_difference_scale"] for v in ("A", "B")]
            c = cross[name]
            criteria[name] = dict(
                both_visits_noise_margin=None if any(x is None for x in ratios) else all(x >= 3. for x in ratios),
                cross_visit_cosine=None if c["cosine"] is None else c["cosine"] >= .8,
                cross_visit_relative_change=None if c["relative_change"] is None else c["relative_change"] <= .5,
                both_visits_REF_curvature_margin=None if null_norm is None else all(
                    visits[v][name]["norm"] > 0 and visits[v][name]["norm"] >= 3.*null_norm for v in ("A", "B")))
        axes[axis] = dict(visits=visits, cross_visit=cross, reported_criteria=criteria,
            cross_time={method: _classification(rows, pair["A"], pair["B"], method)
                        for method in ("raw8", "sum8", "unit_total8")})
    jacobian = dict(available=False, matrix8x2=None, scaled_target_singular_values=None,
                    primary_raw_matrix8x2=None, primary_raw_scaled_target_singular_values=None)
    if all(axes[axis]["cross_visit"]["complete"] for axis in PAIRS):
        columns = [sum(np.asarray(axes[axis]["visits"][v]["bracketed_REF"]["vector8"])
                       for v in ("A", "B"))/2./(2*VELOCITY_STEP_M_S) for axis in PAIRS]
        matrix = np.column_stack(columns)
        singular = np.linalg.svd(matrix*TARGET_TAU_M_S, compute_uv=False)
        raw_columns = [sum(np.asarray(axes[axis]["visits"][v]["raw"]["vector8"])
                           for v in ("A", "B"))/2./(2*VELOCITY_STEP_M_S) for axis in PAIRS]
        raw_matrix = np.column_stack(raw_columns)
        jacobian.update(available=True, matrix8x2=matrix.tolist(), scaled_target_singular_values=singular.tolist(),
            primary_raw_matrix8x2=raw_matrix.tolist(),
            primary_raw_scaled_target_singular_values=np.linalg.svd(raw_matrix*TARGET_TAU_M_S, compute_uv=False).tolist())
    jacobian.update(axis_order=list(PAIRS), finite_difference_step_m_s=VELOCITY_STEP_M_S,
        target_tau_m_s=[TARGET_TAU_M_S]*2, target_is_linear_extrapolation_NOT_verified_20mm_s=True,
        covariance_inverse_or_noise_whitening=False, offline_REF_interpolation=True,
        matrix8x2_is_offline_REF_diagnostic=True)
    def drift(keys):
        reference = rows[keys[0]]["mean"]
        return [dict(id=key, frames=rows[key]["count"],
            delta_from_first8=None if reference is None or rows[key]["mean"] is None else (rows[key]["mean"]-reference).tolist(),
            delta_norm=None if reference is None or rows[key]["mean"] is None else float(np.linalg.norm(rows[key]["mean"]-reference)))
                for key in keys]
    return dict(schema="optical-lab-direction-fixed8-statistics-1", expected_states=17, expected_frames=136,
        available_frames=sum(row["count"] for row in rows.values()),
        complete_states=sum(row["complete"] for row in rows.values()), data_complete=all(row["complete"] for row in rows.values()),
        states=[_public_state(row) for row in rows.values()], host_time_origin=origin,
        host_times_are_NOT_verified_exposure_times=True,
        fixed_readout_and_current_lens_identity_must_be_bound_by_caller=True,
        time_visits=2, statistically_independent_visits_proven=False, declared_signal_command_count=5,
        repeated_REF_episodes=6, probe_episodes=8, background_episodes=3,
        raw8_is_main_branch=True, BG_NOT_used_for_signal_subtraction=True,
        BG_is_both_outputs_zero_NOT_physical_dark_or_same_lens_photon_background=True,
        REF_drift=drift([key for key in STATE_IDS if "REF" in key]), BG_drift=drift(["BG0", "BG1", "BG2"]),
        REF_curvature_null=nulls, axes=axes, velocity_jacobian=jacobian,
        predeclared_criteria=dict(CRITERIA), criteria_are_descriptive_NOT_significance=True,
        qualification="UNKNOWN", release_approved=False,
        model_comparison="caller must bind frozen predictions; arbitrary ASM intensity is NOT ADU and must not be clipped as ADU")


def plot_direction_report(report, output_prefix):
    """One fixed CPU PNG/PDF summary at180dpi; never starts a UI or overwrites."""
    from .reporting import plotting, save_figure
    prefix = Path(output_prefix)
    if not prefix.parent.is_dir() or any(Path(str(prefix)+suffix).exists() for suffix in (".png", ".pdf")):
        raise FileExistsError("existing parent and NEW figure pair required")
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for ax, axis in zip(axes[0], PAIRS):
        for visit, style in (("A", "o-"), ("B", "s--")):
            row = report["axes"][axis]["visits"][visit]
            if row["complete"]:
                ax.plot(range(8), row["raw"]["vector8"], style, label=visit+" raw difference")
                ax.plot(range(8), row["bracketed_REF"]["vector8"], style, alpha=.5, label=visit+" offline REF corrected")
        ax.axhline(0, color=".7", lw=.5)
        ax.set(title=axis+" +/- contrast; all8 fixed regions", xlabel="Fixed region index", ylabel="Caller feature units")
        ax.legend(frameon=False, fontsize=7)
    for key, label in (("REF_drift", "Repeated identical REF"), ("BG_drift", "Both outputs0 background")):
        rows = report[key]
        axes[1, 0].plot(range(len(rows)), [np.nan if row["delta_norm"] is None else row["delta_norm"] for row in rows], "o-", label=label)
    axes[1, 0].set(title="Drift relative to first matching command", xlabel="Episode ordinal within command type", ylabel="8-vector change norm")
    axes[1, 0].legend(frameon=False, fontsize=7)
    labels, values = [], []
    for axis in PAIRS:
        for method in ("raw8", "sum8", "unit_total8"):
            labels.append(axis+"\n"+method)
            value = report["axes"][axis]["cross_time"][method]["accuracy"]
            values.append(np.nan if value is None else value)
    axes[1, 1].bar(range(len(values)), values, color="#0072B2")
    axes[1, 1].set_xticks(range(len(values)), labels, fontsize=7)
    axes[1, 1].set(ylim=(0, 1.05), ylabel="Per-frame descriptive accuracy", title="A prototypes -> B frames; no test selection")
    for i, value in enumerate(values):
        if not np.isfinite(value):
            axes[1, 1].text(i, .05, "NA", ha="center")
    fig.suptitle("FIXED CURRENT LENS / OFFLINE DIAGNOSTICS / QUALIFICATION UNKNOWN\n"
                 f"{report['available_frames']}/136 frames;8 repeats per episode, not independent states", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .93))
    save_figure(fig, prefix, 180)
    plt.close(fig)
    return {suffix[1:]: str(prefix)+suffix for suffix in (".png", ".pdf")}
