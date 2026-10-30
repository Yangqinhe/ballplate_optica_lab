"""Raw-ADU metrics. No fitted per-image scale, shift, normalization or warp."""
import numpy as np


def image_stack(value, name="CCD"):
    a = np.asarray(value)
    if (a.ndim != 3 or len(a) < 2 or min(a.shape[1:]) < 2
            or not np.isfinite(a).all()):
        raise ValueError(name + " requires >=2 finite same-shape frames [N,H,W]")
    return a.astype(np.float64)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 1e-12 else None


def ncc(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    a = a-a.mean(); b = b-b.mean()
    value = ratio(np.dot(a, b), np.linalg.norm(a)*np.linalg.norm(b))
    return None if value is None else float(np.clip(value, -1., 1.))


def validate_regions(regions, shape):
    if regions is None:
        return None
    r = np.asarray(regions)
    if (r.shape != (8, *shape) or not np.isin(r, (0, 1)).all()
            or not np.all(r.sum(0) == 1) or np.any(r.sum((1, 2)) == 0)):
        raise ValueError("eight binary, nonempty, disjoint regions covering the native ROI required")
    return r.astype(np.float64)


def integrate(stack, regions):
    return np.asarray(stack, np.float64).reshape(len(stack), -1) @ regions.reshape(8, -1).T


def describe_real(real_stack, dark_stack=None, regions=None):
    real = image_stack(real_stack)
    mean = real.mean(0)
    rms = float(np.sqrt(real.var(0, ddof=1).mean()))
    dark = None if dark_stack is None else image_stack(dark_stack, "dark")
    if dark is not None and dark.shape[1:] != real.shape[1:]:
        raise ValueError("dark frame shape differs; no implicit resize/crop")
    masks = validate_regions(regions, mean.shape)
    result = {
        "n_frames": len(real), "native_shape_hw": list(mean.shape),
        "raw_mean_ADU": float(mean.mean()), "spatial_contrast_std_ADU": float(mean.std()),
        "single_frame_repeat_pixel_RMS_ADU": rms,
        "single_frame_repeat_RMS_over_mean_contrast": ratio(rms, mean.std()),
        "saturated_fraction_at_255": float((real >= 255).mean()),
        "matched_zero_output_background_available": dark is not None,
        "sensor_physical_dark_measurement_verified": False,
        "absolute_optical_efficiency": None,
    }
    if dark is not None:
        net = mean-dark.mean(0)
        signal_flux = float(net.sum())
        result.update(net_ROI_sum_ADU=signal_flux,
                      net_negative_pixel_fraction=float((net < 0).mean()),
                      single_frame_ROI_flux_std_ADU=float(real.sum((1, 2)).std(ddof=1)),
                      ROI_flux_repeat_CV=ratio(real.sum((1, 2)).std(ddof=1), signal_flux))
    if masks is not None:
        features = integrate(real, masks)
        result.update(native8_mean_ADU=features.mean(0).tolist(),
                      native8_single_std_ADU=features.std(0, ddof=1).tolist(),
                      native8_covariance_ADU2=np.cov(features, rowvar=False).tolist())
    return result


def compare_model(real_stack, model, regions=None, head=None, dark_stack=None):
    real = image_stack(real_stack)
    target = np.asarray(model, np.float64)
    mean = real.mean(0)
    if target.shape != mean.shape or not np.isfinite(target).all():
        raise ValueError("model must be finite native [H,W]; no implicit resize")
    error = target-mean
    rmse = float(np.sqrt(np.mean(error**2)))
    repeat = float(np.sqrt(real.var(0, ddof=1).mean()))
    masks = validate_regions(regions, mean.shape)
    result = {
        "raw_RMSE_ADU": rmse, "raw_MAE_ADU": float(np.mean(np.abs(error))),
        "raw_mean_bias_ADU": float(error.mean()), "NCC": ncc(mean, target),
        "RMSE_over_real_spatial_contrast": ratio(rmse, mean.std()),
        "model_mean_error_over_single_frame_repeat_RMS": ratio(rmse, repeat),
        "no_per_image_fitted_gain_shift_or_warp": True,
    }
    if dark_stack is not None:
        dark = image_stack(dark_stack, "dark")
        if dark.shape[1:] != mean.shape:
            raise ValueError("dark shape differs")
        net = mean-dark.mean(0)
        result["raw_RMSE_over_measured_net_signal_RMS"] = ratio(rmse, np.sqrt(np.mean(net**2)))
    if masks is None:
        if head is not None:
            raise ValueError("a head requires the same frozen eight regions")
        return result
    real8 = integrate(real, masks)
    model8 = integrate(target[None], masks)[0]
    delta = model8-real8.mean(0)
    result.update(native8_model_ADU=model8.tolist(),
                  native8_delta_ADU=delta.tolist(),
                  native8_relative_L2=ratio(np.linalg.norm(delta), np.linalg.norm(real8.mean(0))),
                  native8_error_over_single_repeat_RMS=ratio(np.linalg.norm(delta), np.linalg.norm(real8.std(0, ddof=1))))
    if dark_stack is not None:
        dark8 = integrate(dark, masks).mean(0)
        result["native8_raw_error_over_measured_net_signal_L2"] = ratio(np.linalg.norm(delta), np.linalg.norm(real8.mean(0)-dark8))
    if head is not None:
        w = np.asarray(head["weight_raw_ADU_to_rad"], np.float64)
        b = np.asarray(head["bias_rad"], np.float64)
        if w.shape != (2, 8) or b.shape != (2,) or not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError("finite frozen raw18 head required")
        actual_action = real8 @ w.T+b
        predicted_action = model8 @ w.T+b
        result.update(raw_head_model_action_rad=predicted_action.tolist(),
                      raw_head_real_mean_action_rad=actual_action.mean(0).tolist(),
                      raw_head_mean_action_MAE_rad=float(np.mean(np.abs(predicted_action-actual_action.mean(0)))),
                      raw_head_single_repeat_std_rad=actual_action.std(0, ddof=1).tolist(),
                      head_diagnostic_NOT_a_closed_loop=True)
    return result


def paired_response(stack_a, stack_b, regions, model_a=None, model_b=None):
    """Descriptive difference of two cells; not a causal proof.

    Does NOT remove drift or assert simultaneous/associated camera exposure.
    RMS ratio uses single-frame repeat noise, NOT number-of-pixels or a p-value.
    """
    a = image_stack(stack_a); b = image_stack(stack_b)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError("probe pair shape differs")
    masks = validate_regions(regions, a.shape[1:])
    if masks is None:
        raise ValueError("probe assessment requires frozen regions")
    fa, fb = integrate(a, masks), integrate(b, masks)
    response = fb.mean(0)-fa.mean(0)
    repeat = np.sqrt(fa.var(0, ddof=1)+fb.var(0, ddof=1))
    out = {
        "native8_response_ADU": response.tolist(),
        "native8_difference_single_repeat_std_ADU": repeat.tolist(),
        "native8_response_over_single_repeat_RMS": ratio(np.linalg.norm(response), np.linalg.norm(repeat)),
        "image_difference_RMS_ADU": float(np.sqrt(np.mean((b.mean(0)-a.mean(0))**2))),
        "statistical_significance_or_absolute_efficiency": False,
        "time_drift_removed": False,
        "noise_scale_assumption": "single-frame quadrature reference sqrt(varA+varB); cross-cell covariance unmeasured",
        "noise_scale_is_mean_SE_or_measured_null_difference": False,
    }
    if (model_a is None) != (model_b is None):
        raise ValueError("both model endpoints are required")
    if model_a is not None:
        ma, mb = np.asarray(model_a, float), np.asarray(model_b, float)
        if ma.shape != a.shape[1:] or mb.shape != ma.shape or not np.isfinite(ma).all() or not np.isfinite(mb).all():
            raise ValueError("finite native model endpoints required")
        predicted = integrate((mb-ma)[None], masks)[0]
        out.update(model_native8_response_ADU=predicted.tolist(),
                   model_to_real_response_cosine=ratio(np.dot(response, predicted), np.linalg.norm(response)*np.linalg.norm(predicted)),
                   response_relative_L2_error=ratio(np.linalg.norm(predicted-response), np.linalg.norm(response)))
    return out


def bracketed_contrast(probe_mean, t_probe, z_before, t_before, z_after, t_after):
    """Mean-level probe minus timestamp-interpolated bracketing Z.

    All three timestamps must use the SAME clock/unit and a suitably small
    relative origin. This function does not prove optical exposure timing,
    estimate independent samples, synthesize a repeated reference, or extrapolate.
    The returned images can be negative and remain in their original ADU units.
    """
    probe, before, after = (np.asarray(x, np.float64) for x in (probe_mean, z_before, z_after))
    if (probe.ndim != 2 or min(probe.shape) < 2 or before.shape != probe.shape or after.shape != probe.shape
            or not all(np.isfinite(x).all() for x in (probe, before, after))):
        raise ValueError("three finite same-native-shape mean images required")
    times = np.asarray([t_before, t_probe, t_after], np.float64)
    if not np.isfinite(times).all() or not times[0] < times[1] < times[2]:
        raise ValueError("same-clock strict temporal brackets required; no extrapolation")
    weight_after = float((t_probe - t_before) / (t_after - t_before))
    reference = (1.0 - weight_after) * before + weight_after * after
    return {"reference": reference, "contrast": probe - reference,
            "weights_before_after": [1.0 - weight_after, weight_after],
            "mean_level_only": True, "repeat_stack_synthesized": False,
            "exposure_timestamp_association_proven": False}
