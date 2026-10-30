"""Raw native-ROI statistics; no image alignment/rescaling in repeat gates."""
import cv2
import numpy as np


def ncc(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2:raise ValueError('matching native 2D arrays required')
    a=a-a.mean();b=b-b.mean();den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return None if den<=1e-12 else float(np.sum(a*b)/den)


def relative_l2(reference,measured):
    a=np.asarray(reference,dtype=np.float64);b=np.asarray(measured,dtype=np.float64)
    den=float(np.linalg.norm(a))
    if a.shape!=b.shape or den<=1e-12:raise ValueError('matching arrays and nonzero reference required')
    return float(np.linalg.norm(b-a)/den)


def translation(reference,measured):
    a=np.asarray(reference,dtype=np.float64);b=np.asarray(measured,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2:raise ValueError('matching native 2D arrays required')
    h,w=a.shape
    window=cv2.createHanningWindow((w,h),cv2.CV_64F)
    # Copies: phaseCorrelate may apply its window in-place. Never mutate data.
    xy,response=cv2.phaseCorrelate(a.copy(),b.copy(),window)
    return dict(xy_px=[float(xy[0]),float(xy[1])],phase_correlation_response=float(response))


def compare_repeat(reference,measured,gates):
    shift=translation(reference,measured);score=ncc(reference,measured);error=relative_l2(reference,measured)
    shift_norm=float(np.linalg.norm(shift['xy_px']))
    return dict(NCC=score,relative_L2_unaligned_unscaled=error,**shift,shift_norm_px=shift_norm,
        passed=bool(score is not None and score>=gates['minimum_repeat_NCC']
            and error<=gates['maximum_repeat_relative_L2']
            and shift_norm<=gates['maximum_zero_return_shift_px']
            and shift['phase_correlation_response']>=gates['minimum_phase_correlation_response']))
