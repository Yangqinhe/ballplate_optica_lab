"""Live RAW96 optical inference and exact CURRENT CCD/actor display pairing."""
import runtime96 as rt
import collections
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import mujoco
import numpy as np
import torch
from PIL import Image
from model96 import load,resize_binary,pack,dmd_canvas
from dashboard.acquisition import readonly
from dashboard.interfaces import CCDFrame,HistorySlot
from env.ballplate_env import BallPlateEnv
from fourframe_dashboard_session import FourFrameSession,FourFrameSnapshot

RUN=rt.ROOT/'outputs/four96_block4_split_fit_seed97711_v1'
CHECKS=rt.ROOT/'outputs/four96_block4_offline_checks_v1'
DEFAULT_CHECKPOINT=RUN/'best.pt'
REGIONS=[(r*300,(r+1)*300,c*150,(c+1)*150) for r in range(2) for c in range(4)]
VARIANTS={'learned':'96四幅 · 学习相位 / 原线性头','flat_same_head':'96四幅 · 零指令相位 / 同一线性头'}


def statistics96(counts,reference=100000.):
    counts=np.asarray(counts)
    if counts.dtype!=np.uint8 or counts.shape!=(600,600):raise ValueError('registered simulated Mono8 CCD600 required')
    sums=np.asarray([counts[y0:y1,x0:x1].sum(dtype=np.uint64) for y0,y1,x0,x1 in REGIONS])
    return dict(region_counts=sums,features=sums.astype(np.float32)/np.float32(reference))


def matching_readout96(displayed,actual):
    # CUDA scalar division can use a reciprocal multiply, whereas NumPy uses
    # rounded division. Allow only two FP32 ulps; integer sums are checked
    # separately, exactly, before the policy is evaluated.
    return np.allclose(displayed,actual,rtol=2*np.finfo(np.float32).eps,atol=0.)


class Session96(FourFrameSession):
    def __init__(self,checkpoint=DEFAULT_CHECKPOINT,flat_checkpoint=None,seed=92010,extent=.05,
                 duration=60.,device='cuda',backend='simulation',xy=None):
        if backend!='simulation':raise ValueError('only simulation is enabled; no device outputs')
        if flat_checkpoint:raise ValueError('no matched zero-phase refitted96 head exists')
        if not 0<extent<=.23 or not 0<duration<=3600:raise ValueError('invalid simulation extent/horizon')
        rt.setup(device);self.primary_path=Path(checkpoint).resolve();self.path=self.primary_path
        self.model,self.primary_payload=load(self.path,device);self.model.eval().requires_grad_(False)
        self.loaded_sources={str(self.path):rt.digest(self.path)}
        self.sampling_error=None;p=CHECKS/'sampling_guard/refinement.json'
        if p.exists():
            audit=json.loads(p.read_text())
            if audit['checkpoint_sha256']==self.loaded_sources[str(self.path)]:self.sampling_error=audit['max_action_error_rad']
        learned=self.model.optical.phase_raw.detach().cpu().clone()
        self.variant_states={'learned':learned,'flat_same_head':torch.full_like(learned,-30.)}
        self.extent,self.seed=extent,seed;self.rng=np.random.default_rng(seed)
        self.env=BallPlateEnv(seed=seed,initial_range=extent,duration=duration,xml=self.primary_payload['metadata']['model_xml'])
        self.overview=mujoco.Renderer(self.env.model,height=320,width=400)
        self.overview_camera=mujoco.MjvCamera();self.overview_camera.lookat[:]=[0,0,.30]
        self.overview_camera.distance=.98;self.overview_camera.azimuth=110;self.overview_camera.elevation=-57
        self.acquisition=SimpleNamespace(close=lambda:None)
        self.records=collections.deque(maxlen=1)
        self.raw_history=collections.deque(maxlen=4);self.raw_ids=collections.deque(maxlen=4)
        self.episode,self.sequence,self.global_steps=0,-1,0;self.compute_seconds=0.
        self.force_steps=0;self.force_local=np.zeros(3)
        self.ball_id=self.env.model.body('ball').id;self.plate_id=self.env.model.body('plate').id
        self.optical_forward_calls=0;self.last_control_row=None;self.signal_valid=True
        self._install_variant('learned');self.reset(np.zeros(2) if xy is None else xy)

    def model_digest(self):
        digest=hashlib.sha256()
        items=[('phase',self.model.optical.phase_raw),*self.model.actor_head.state_dict().items()]
        for key,value in items:digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _install_variant(self,name):
        if name not in self.variant_states:raise ValueError('unsupported96 phase ablation')
        with torch.no_grad():self.model.optical.phase_raw.copy_(self.variant_states[name].to(next(self.model.parameters()).device))
        self.variant=name;self.path=self.primary_path
        self.payload=dict(self.primary_payload,loaded_checkpoint_sha256=self.loaded_sources[str(self.path)])
        self.expected_runtime_digest=self.model_digest()
        self.slm_active=readonly(self.model.optical.phase_uint8().cpu().numpy())
        self.phase_gray_range=(int(self.slm_active.min()),int(self.slm_active.max()))
        self.phase_levels=len(np.unique(self.slm_active))

    @torch.no_grad()
    def capture(self,reset=False):
        started=time.perf_counter();self.sequence+=1;image=resize_binary(self.binary)
        if reset:
            self.raw_history.clear();self.raw_ids.clear()
            self.raw_history.extend(image.copy() for _ in range(4));self.raw_ids.extend([self.sequence]*4)
        else:self.raw_history.append(image.copy());self.raw_ids.append(self.sequence)
        raw=torch.from_numpy(np.stack(self.raw_history))[None]
        self.logical=readonly(pack(raw)[0,0].numpy());self.dmd_canvas=readonly(dmd_canvas(raw)[0,0].numpy())
        device=next(self.model.parameters()).device
        counts=self.model.ccd_counts(raw.to(device));self.optical_forward_calls+=1
        q=self.model.features(counts);self.features=q
        displayed=counts[0].to(torch.uint8).cpu().numpy();stats=statistics96(displayed,self.model.cfg.feature_reference)
        error=float(np.abs(stats['features']-q[0].cpu().numpy()).max())
        integer_sums=counts.reshape(-1,2,300,4,150).sum((2,4)).flatten(1)[0].cpu().numpy()
        if not np.array_equal(integer_sums,stats['region_counts']):raise RuntimeError('displayed CCD600 region counts differ from policy sums')
        if not matching_readout96(stats['features'],q[0].cpu().numpy()):raise RuntimeError('displayed CCD600 and actual actor CCD8 differ')
        self.region_counts=readonly(stats['region_counts'])
        reference=self.model.actor_head(q)[0].cpu().numpy().clip(-.8,.8)
        w=self.model.actor_head.weight.cpu().numpy();b=self.model.actor_head.bias.cpu().numpy()
        decoded=(w@stats['features']+b).clip(-.8,.8)
        if float(np.abs(decoded-reference).max())>2e-5:raise RuntimeError('electrical NumPy/Torch head mismatch')
        self.signal_valid=bool(np.any(self.region_counts));self.next_action=reference.copy()
        if not self.signal_valid:
            self.next_action=np.zeros(2,dtype=np.float32)
            if not self.finished:self.finished,self.end_reason=True,'missing_optical_signal'
        self.overview.update_scene(self.env.data,camera=self.overview_camera);self.scene_rgb=readonly(self.overview.render())
        frame=CCDFrame(self.sequence,self.episode,self.simulation_time,time.time(),0,
            readonly(displayed),readonly(q[0].cpu().numpy()),error,time.perf_counter()-started)
        self.records.clear();self.records.append(HistorySlot(frame,False));self.current_ccd=frame

    def snapshot(self):
        cfg=self.model.cfg
        return FourFrameSnapshot(episode=self.episode,frame_id=self.current_ccd.frame_id,
            simulated_time_s=self.simulation_time,rgb=self.scene_rgb,dmd_logical=self.logical,
            dmd_active=self.dmd_canvas,slm_active=self.slm_active,history=tuple(self.records),
            next_mean_action=readonly(self.next_action),last_applied_action=readonly(self.last_applied),
            last_action_input_id=self.last_action_input_id,position=readonly(self.env.position()),speed=self.speed,
            reward_sum=self.reward_sum,stable=self.stable,finished=self.finished,end_reason=self.end_reason,
            force_steps_remaining=self.force_steps,capture_seconds=self.current_ccd.acquisition_seconds,
            mean_step_seconds=self.compute_seconds/max(self.global_steps,1),
            checkpoint_steps=self.payload['metadata'].get('selected_update',0),
            checkpoint_sha256=self.payload['loaded_checkpoint_sha256'],checkpoint_name=self.path.name,
            phase_frozen=True,propagation='ASM注册网格代理',grid_size=cfg.simulation_size,device=str(next(self.model.parameters()).device),
            cache_error=self.current_ccd.cache_max_error,variant=self.variant,raw_input_ids=tuple(self.raw_ids),
            region_counts=self.region_counts,optical_forward_calls=self.optical_forward_calls,signal_valid=self.signal_valid,
            phase_gray_range=self.phase_gray_range,phase_levels=self.phase_levels)

    def metadata(self):
        cfg=self.model.cfg
        return dict(role='four_RAW96_C150_simulation_dashboard_NOT_device_output',checkpoint=str(self.path),
            checkpoint_sha256=self.payload['loaded_checkpoint_sha256'],model_config=asdict(cfg),geometry=self.model.geometry(),
            available_variants=list(self.variant_states),variant=self.variant,hardware_enabled=False,calibrated=False,
            release_approved=False,phase_frozen=True,actor_parameters=18,actor_macs=16,
            input_ages_s=[.06,.04,.02,0.],input_history_frames=4,ccd_frames_retained=1,ccd_history_used_by_actor=0,
            optical_feature_memoization=False,initial_extent_m=self.extent,simulation_horizon_s=self.env.max_steps*self.env.control_dt,
            control_dt_s=self.env.control_dt,seed=self.seed,numerical_refinement_passed=False,
            known_main_model_sampling_action_difference_rad=self.sampling_error,
            ccd_readout=dict(corrected_regions=[list(r) for r in REGIONS],reference_counts=cfg.feature_reference,
                coordinate_order='registered600 half-open [y0,y1,x0,x1]; no additional flip',display_overlay_only=True,
                integer_sums_exactly_checked=True,normalized_readout_rtol=float(2*np.finfo(np.float32).eps)),
            CCD_is_simulated_registered_domain_NOT_raw_sensor_ROI=True,SLM_is_canonical_uint8_command=True,
            SLM_historical_LUT_preview_is_display_only=True,training_role=self.payload['metadata']['role'],
            training_selected_update=self.payload['metadata'].get('selected_update'),
            force_frame='plate-local XY',velocity_kick_frame='world XY')


def save96(snapshot,output,metadata):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    Image.fromarray(snapshot.rgb).save(output/'simulation_overview.png')
    Image.fromarray(snapshot.dmd_logical).save(output/'DMD_RAW96x4_LOGICAL192.png')
    Image.fromarray(snapshot.dmd_active).save(output/'DMD_SIMULATION600_BINARY.png')
    Image.fromarray(snapshot.slm_active).save(output/'SLM_CANONICAL_COMMAND_SIMULATION600.png')
    frame=snapshot.history[0].frame;Image.fromarray(frame.raw_counts).save(output/'CURRENT_CCD_REGISTERED600_SIMULATED.png')
    rt.json_save(output/'bundle.json',dict(metadata,frame_id=snapshot.frame_id,episode=snapshot.episode,
        simulated_time_s=snapshot.simulated_time_s,raw_input_ids=list(snapshot.raw_input_ids),
        current_q8=frame.features.tolist(),next_mean_action=snapshot.next_mean_action.tolist(),
        last_applied_action=snapshot.last_applied_action.tolist(),current_region_counts=snapshot.region_counts.tolist(),
        displayed_native_readout_error=snapshot.cache_error,optical_forward_calls=snapshot.optical_forward_calls,
        hardware_enabled=False,release_approved=False))
