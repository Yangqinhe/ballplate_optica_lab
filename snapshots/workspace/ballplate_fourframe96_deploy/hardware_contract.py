"""Pure offline preparation for SDON HardwareManager; no hardware constructor."""
import runtime96 as rt
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from model96 import dmd_canvas,calibration_tables


def reference_audit():
    alignment=json.loads((rt.REFERENCE/"alignment_config_latest.json").read_text())
    phase=json.loads((rt.PHASE/"slm_phase_calibration_summary.json").read_text())
    keys=("camera_roi","display_mode","camera","slm_offset","ccd_orientation","dmd_canvas_size","slm_canvas_size")
    equality={k:alignment.get(k)==phase.get(k) for k in keys}
    lut,response,commands=calibration_tables()
    return dict(archive=str(rt.HANDOFF),reference_date="2026-09-10",historical_only=True,
        internal_conditions_equal=equality,reference_chain_internally_consistent=all(equality.values()),
        SLM_model="FSLM-2K70-P02",DMD_model="DLP6500/DLPC900",camera_model=alignment["camera"]["DeviceModelName"],
        DMD_video=alignment["display_mode"]["dmd"],SLM_video=alignment["display_mode"]["slm"],
        camera=alignment["camera"],camera_roi=alignment["camera_roi"],
        canonical_canvas=600,SLM_offset=alignment["slm_offset"],CCD_orientation=alignment["ccd_orientation"],
        software_rotation_deg=-89.8,crop_TBLR=[5,0,3,3],
        affine_simulation_to_prepared_CCD=alignment["geometry_transform"],
        phase_depth_rad=float(response.max()),phase_command_curve_max_rad=float(commands.max()),
        LUT_unique_grays=int(np.unique(lut).size),LUT_endpoint_raw_grays=[int(lut[0]),int(lut[255])],
        DMD_grayscale_LUT_same_conditions_available=False,
        binary_DMD_only_diagnostic_bypass=True,current_hardware_verified=False,
        lens_reference_m=[.15,.15],registered_grid_pitch_m=8e-6,
        registered_grid_is_not_raw_CCD_pixel_pitch=True,
        phase_time_instability_still_unresolved_in_archive=True,
        live_deployment_allowed=False,sources=rt.sources())


def prepare_DMD(raw):
    tensor=torch.as_tensor(raw)
    if tensor.ndim==4:tensor=tensor[None]
    return dmd_canvas(tensor)[0,0].cpu().numpy().copy()


def slm_preview(command):
    command=np.asarray(command)
    if command.dtype!=np.uint8 or command.shape!=(600,600):raise ValueError("canonical 600 uint8 command expected")
    lut,_,_=calibration_tables()
    # Manager.update_slm_image applies LUT and then flip_horizontal once.
    # This result is a PREVIEW, never feed it back to apply_lut=True.
    return np.ascontiguousarray(np.fliplr(lut[command]))


def preprocess_sensor_roi(raw_frame,orientation="flip_vertical",rotation_deg=-89.8,crop=(5,0,3,3)):
    """Same order as archived HardwareManager.capture_ccd_image, without SDK."""
    image=np.asarray(raw_frame)
    if image.ndim!=2 or not np.isfinite(image).all():raise ValueError("finite raw sensor ROI required")
    rows,cols=image.shape;rotation=cv2.getRotationMatrix2D((cols/2.,rows/2.),rotation_deg,1.)
    image=cv2.warpAffine(image,rotation,(cols,rows))
    top,bottom,left,right=crop;image=image[top:rows-bottom if bottom else rows,left:cols-right if right else cols]
    if image.size==0:raise ValueError("empty CCD crop")
    if orientation=="flip_vertical":image=cv2.flip(image,0)
    elif orientation=="flip_horizontal":image=cv2.flip(image,1)
    elif orientation=="rotate_180":image=cv2.flip(image,-1)
    elif orientation!="identity":raise ValueError("unsupported explicit orientation")
    side=min(image.shape);y0=(image.shape[0]-side)//2;x0=(image.shape[1]-side)//2
    return cv2.resize(image[y0:y0+side,x0:x0+side],(600,600),interpolation=cv2.INTER_AREA)


def registered_energy(raw_frame,background,simulation_to_prepared_CCD,orientation="flip_vertical",
                      rotation_deg=-89.8,crop=(5,0,3,3),reference=100000.):
    """RAW input only; never pass manager's already-preprocessed output here."""
    if np.asarray(raw_frame).shape!=np.asarray(background).shape:raise ValueError("same-condition raw ROI and dark required")
    image=preprocess_sensor_roi(raw_frame,orientation,rotation_deg,crop).astype(np.float32)
    dark=preprocess_sensor_roi(background,orientation,rotation_deg,crop).astype(np.float32)
    return prepared_energy(np.maximum(image-dark,0),simulation_to_prepared_CCD,reference)


def prepared_energy(prepared_CCD,simulation_to_prepared_CCD,reference=100000.):
    """Input is SAME-CONDITION DARK-SUBTRACTED manager-prepared 600 image."""
    image=np.asarray(prepared_CCD,dtype=np.float32);matrix=np.asarray(simulation_to_prepared_CCD,dtype=np.float32)
    if image.shape!=(600,600) or matrix.shape!=(2,3) or not np.isfinite(image).all() or not np.isfinite(matrix).all():
        raise ValueError("prepared 600x600 CCD and its same-domain affine required")
    registered=cv2.warpAffine(image,matrix,(600,600),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return registered.reshape(2,300,4,150).sum((1,3)).reshape(8)/reference,registered


class FreshFrameGuard:
    """Fail closed on stale/mismatched acquisition records; never claim optical settling."""
    def __init__(self):self.last_id=None
    def check(self,record,expected_DMD_hash,applied_time_ns,current_calibration_verified=False):
        if not current_calibration_verified:raise RuntimeError("current hardware calibration not verified")
        if record.get("DMD_hash")!=expected_DMD_hash:raise RuntimeError("CCD/DMD association mismatch")
        if record.get("capture_time_ns",0)<=applied_time_ns:raise RuntimeError("stale CCD acquisition")
        if self.last_id is not None and record.get("frame_id",-1)<=self.last_id:raise RuntimeError("reused CCD frame")
        if not record.get("same_clock_verified",False):raise RuntimeError("camera/display clock relationship unknown")
        if not record.get("optical_stability_passed",False):raise RuntimeError("display event is not optical settling")
        self.last_id=record["frame_id"]
