"""Raw full-FOV link evidence; deliberately independent of ROI/ASM/LUT fitting."""
import cv2
import numpy as np


def signal_repeat_noise(stack, dark, fallback):
    """Temporal noise on illuminated pixels, NOT diluted by a large dark FOV."""
    stack=np.asarray(stack,float)
    if stack.ndim != 3 or stack.shape[0] < 3 or stack.shape[1:] != np.asarray(dark).shape:
        raise ValueError('at least three matching raw frames required')
    mask=np.median(stack,axis=0)-np.asarray(dark,float)>=5
    if mask.sum()<64:return float(fallback)
    samples=stack[:,mask]
    return max(float(fallback),float(np.sqrt(np.mean((samples-samples.mean(0))**2))))


def signal_evidence(bright, dark, noise):
    bright, dark = np.asarray(bright, float), np.asarray(dark, float)
    if bright.shape != dark.shape or bright.ndim != 2:
        raise ValueError('matching native 2-D frames required')
    delta = bright-dark
    threshold = max(5., 5.*float(noise))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((delta >= threshold).astype(np.uint8), 8)
    if n <= 1: return dict(observed=False, threshold_ADU=threshold, area=0), np.zeros(delta.shape, bool)
    index = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = map(int, stats[index])
    mask = labels == index
    return dict(observed=bool(area >= 64), threshold_ADU=threshold, area=area,
                bbox=[x,y,w,h], median_contrast_ADU=float(np.median(delta[mask])),
                near_sensor_edge=bool(x<5 or y<5 or x+w>delta.shape[1]-5 or y+h>delta.shape[0]-5)), mask


def dmd_evidence(dark, white, pattern, returned, black_end, noise):
    signal, mask = signal_evidence(white, dark, noise)
    if not signal['observed']:
        return dict(passed=False, response_observed=False, repeat_stability_passed=False, signal=signal,
                    reason='no coherent optical signal above dark noise; check illumination/path, not ROI edges')
    rms = lambda a,b: float(np.sqrt(np.mean((np.asarray(a,float)[mask]-np.asarray(b,float)[mask])**2)))
    limit = max(1., 5.*float(noise))
    repeat, black, change = rms(white,returned), rms(dark,black_end), rms(white,pattern)
    observed=bool(black<=limit and change>limit)
    stable=bool(repeat<=limit and black<=limit)
    return dict(passed=bool(observed and stable), response_observed=observed,
                repeat_stability_passed=stable, signal=signal, repeat_noise_ADU=float(noise),
                repeat_noise_scope='illuminated pixels; three raw frames, not full-FOV average',
                white_return_rmse_ADU=repeat, black_return_rmse_ADU=black,
                pattern_vs_white_rmse_ADU=change, repeat_limit_ADU=limit,
                precision_calibration=False, release_approved=False)


def slm_evidence(dark, rounds, noise):
    """Reject unchanging images and drifting changes; report basic link only.

    Two independent repeats use an identical DMD image. Each +/- grating must
    change the field above zero-return drift, and the same mask's two responses
    must agree. No fitting to simulation or selecting a new image ROI.
    """
    if len(rounds) != 2: raise ValueError('two physical repeats required')
    signal, mask = signal_evidence(rounds[0]['zero'], dark, noise)
    if not signal['observed']:
        return dict(passed=False, signal=signal, reason='insufficient zero-phase optical signal')
    # Include newly illuminated pixels when gratings move light outside the
    # original image. Never restrict capture to the narrow white-square ROI.
    support = mask.copy()
    for frames in rounds:
        for frame in frames.values(): support |= np.asarray(frame,float)-dark >= max(5.,5.*noise)
    rms = lambda values: float(np.sqrt(np.mean(np.asarray(values,float)[support]**2)))
    drifts = [rms(np.asarray(r['zero_return'],float)-r['zero']) for r in rounds]
    threshold = max(1., 5.*float(noise), 3.*max(drifts))
    cases = {}
    for name in ('lens','plus_x','minus_x','plus_y','minus_y'):
        responses = [np.asarray(r[name],float)-r['zero'] for r in rounds]
        amplitudes = [rms(v) for v in responses]
        mismatch = rms(responses[0]-responses[1])
        cases[name] = dict(change_rmse_ADU=amplitudes, repeat_response_rmse_ADU=mismatch,
            observable_and_repeatable=bool(min(amplitudes)>threshold
                and mismatch<=max(1.,5.*noise,.5*min(amplitudes))))
    passed = all(cases[name]['observable_and_repeatable']
                 for name in ('plus_x','minus_x','plus_y','minus_y'))
    return dict(passed=bool(passed), signal=signal, cases=cases,
        signal_support_pixels=int(support.sum()), change_threshold_ADU=threshold,
        zero_return_rmse_ADU=drifts, precision_calibration=False,
        geometric_grating_validation_required_later=True, phase_LUT_qualified=False,
        hardware_trigger_accepted=False, control_20ms_accepted=False, release_approved=False)
