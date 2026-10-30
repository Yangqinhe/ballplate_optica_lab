"""Predeclared native-ADU repeat diagnostics; no fitting/warping/normalization.

The effective noise includes all within-stage fluctuations. Ratios are
engineering diagnostics, not an independent-pixel statistical significance test.
"""
import numpy as np
from grating_repeat_math import ncc, relative_l2, translation


def moments(stack):
    values = np.asarray(stack, dtype=np.float64)
    if values.ndim != 3 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('at least two finite native 2D frames required')
    return values.mean(0), values.var(0, ddof=1) / len(values)


def rms(array):
    return float(np.sqrt(np.mean(np.square(array))))


def ratio(numerator, denominator):
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError('nonzero effective noise denominator required')
    return float(numerator / denominator)


def learned_repeat(first, second, dark, gates):
    a, va = moments(first); b, vb = moments(second); d, vd = moments(dark)
    if a.shape != b.shape or a.shape != d.shape:
        raise ValueError('same native ROI required')
    a = a - d; b = b - d
    observed = rms(b - a)
    predicted = float(np.sqrt(np.mean(va + vb)))  # Common dark cancels.
    difference_ratio = ratio(observed, predicted)
    totals = [float(a.sum()), float(b.sum())]
    if min(totals) <= 0:
        raise ValueError('positive full-ROI dark-subtracted total signal required')
    total_difference = abs(totals[1] - totals[0]) / totals[0]
    snrs = [ratio(rms(value), float(np.sqrt(np.mean(variance + vd))))
            for value, variance in ((a, va), (b, vb))]
    return dict(difference_RMS_ADU=observed,
        predicted_difference_of_means_noise_RMS_ADU=predicted,
        difference_RMS_over_effective_frame_noise_prediction=difference_ratio,
        full_ROI_dark_subtracted_total_ADU=totals,
        full_ROI_total_signal_relative_difference=float(total_difference),
        signal_RMS_over_single_group_mean_noise_RMS=snrs,
        raw_NCC=ncc(a, b), raw_relative_L2_unaligned_unscaled=relative_l2(a, b),
        passed=bool(difference_ratio <= gates['maximum_difference_RMS_over_effective_frame_noise_prediction']
            and total_difference <= gates['maximum_full_ROI_dark_subtracted_total_signal_relative_difference']
            and min(snrs) >= gates['minimum_signal_RMS_over_single_group_mean_noise_RMS']))


def modulation(zero, learned, gates):
    a, va = moments(zero); b, vb = moments(learned)
    if a.shape != b.shape:
        raise ValueError('same native ROI required')
    observed = rms(b - a)
    predicted = float(np.sqrt(np.mean(va + vb)))
    effect = ratio(observed, predicted)
    return dict(zero_vs_learned_RMS_ADU=observed,
        predicted_difference_of_means_noise_RMS_ADU=predicted,
        RMS_over_group_mean_noise=effect,
        passed=bool(effect >= gates['minimum_zero_vs_learned_RMS_over_group_mean_noise']))


def zero_repeat(first, second, dark, gates):
    a, _ = moments(first); b, _ = moments(second); d, _ = moments(dark)
    a = a - d; b = b - d
    shift = translation(a, b); score = ncc(a, b); error = relative_l2(a, b)
    norm = float(np.linalg.norm(shift['xy_px']))
    return dict(NCC=score, relative_L2_unaligned_unscaled=error, **shift,
        shift_norm_px=norm, passed=bool(score is not None
            and score >= gates['minimum_NCC'] and error <= gates['maximum_relative_L2']
            and norm <= gates['maximum_shift_px']))
