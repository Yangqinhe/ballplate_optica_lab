"""Live four-RAW-frame optical inference; one CURRENT CCD, no CCD history/cache."""
import fourframe_runtime as rt
import collections
from dataclasses import dataclass
import json
from pathlib import Path
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from PIL import Image
from dashboard.session import DashboardSession as ExistingSession, Snapshot, full_panel
from dashboard.acquisition import readonly, native_counts
from dashboard.interfaces import CCDFrame, HistorySlot
from env.ballplate_env import BallPlateEnv
from fourframe_checkpoint import load
from fourframe_model import pack_four
from fourframe_decoder import CurrentCCDLinearHead,integrate_current_mono8


VARIANTS = {"learned": "学习相位 + 原全连接", "flat_same_head": "平面相位 + 同一全连接",
            "flat_refitted": "平面相位 + 重拟合全连接"}
CAMPAIGN = rt.ROOT/"outputs/campaign_fourframe_optical18_20261015_v1"
DEFAULT_CHECKPOINT = CAMPAIGN/"native_optical_PPO/last.pt"
DEFAULT_FLAT = CAMPAIGN/"native_flat_baseline/best.pt"


@dataclass(frozen=True)
class FourFrameSnapshot(Snapshot):
    variant: str
    raw_input_ids: tuple
    region_counts: np.ndarray
    optical_forward_calls: int
    signal_valid: bool
    phase_gray_range: tuple
    phase_levels: int


class FourFrameSession(ExistingSession):
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, flat_checkpoint=DEFAULT_FLAT,
                 seed=92010, extent=.05, duration=60., device="cuda", backend="simulation", xy=None):
        if backend!="simulation": raise ValueError("hardware is disabled; no SDK will be opened")
        if not 0<extent<=.23 or not 0<duration<=3600: raise ValueError("invalid simulation range/horizon")
        rt.setup(device)
        self.primary_path=Path(checkpoint).resolve()
        self.model,payload=load(self.primary_path,device)
        if self.model.electric_history or self.model.feature_dim!=8 or self.model.actor_parameters()!=18:
            raise ValueError("GUI requires four raw inputs, CURRENT CCD8, linear18")
        self.model.eval().requires_grad_(False)
        self.loaded_sources={str(self.primary_path):rt.digest(self.primary_path)}
        self.sampling_error=None
        refinement=CAMPAIGN/"selected_sampling_refinement/refinement.json"
        if refinement.exists():
            measured=json.loads(refinement.read_text())
            if measured["checkpoint_sha256"]==self.loaded_sources[str(self.primary_path)]:
                self.sampling_error=measured["max_action_error_rad"]
        self.variant_states={"learned":{k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}}
        self.variant_payloads={"learned":payload}
        self.variant_paths={"learned":self.primary_path}
        same={k:v.clone() for k,v in self.variant_states["learned"].items()}
        same["encoder.slm.raw"].zero_()
        self.variant_states["flat_same_head"]=same
        self.variant_payloads["flat_same_head"]=payload
        self.variant_paths["flat_same_head"]=self.primary_path
        if flat_checkpoint:
            path=Path(flat_checkpoint).resolve()
            flat=torch.load(path,map_location="cpu",weights_only=False)
            rt.reused.verify_sources(flat["physical_manifest"])
            for filename,expected in flat["source_sha256"].items():
                if rt.digest(filename)!=expected:raise RuntimeError("flat baseline source hash mismatch")
            if flat["optical_config"]!=payload["optical_config"] or flat["options"]!=payload["options"]:
                raise ValueError("flat/learned optical settings must match")
            if flat["metadata"]["model_xml"]!=payload["metadata"]["model_xml"]:
                raise ValueError("flat/learned MuJoCo environments must match")
            if bool(flat["model"]["encoder.slm.raw"].ne(0).any()):raise ValueError("baseline is not a flat mask")
            self.loaded_sources[str(path)]=rt.digest(path)
            self.variant_states["flat_refitted"]=flat["model"]
            self.variant_payloads["flat_refitted"]=flat;self.variant_paths["flat_refitted"]=path
        self.extent,self.seed=extent,seed;self.rng=np.random.default_rng(seed)
        self.env=BallPlateEnv(seed=seed,initial_range=extent,duration=duration,xml=payload["metadata"]["model_xml"])
        self.overview=mujoco.Renderer(self.env.model,height=320,width=400)
        self.overview_camera=mujoco.MjvCamera();self.overview_camera.lookat[:]=[0,0,.30]
        self.overview_camera.distance=.98;self.overview_camera.azimuth=110;self.overview_camera.elevation=-57
        self.acquisition=SimpleNamespace(close=lambda:None)
        self.records=collections.deque(maxlen=1)
        self.raw_history=collections.deque(maxlen=4);self.raw_ids=collections.deque(maxlen=4)
        self.episode,self.sequence,self.global_steps=0,-1,0
        self.compute_seconds=0.;self.force_steps=0;self.force_local=np.zeros(3)
        self.ball_id=self.env.model.body("ball").id;self.plate_id=self.env.model.body("plate").id
        self.optical_forward_calls=0;self.last_control_row=None;self.signal_valid=True
        self._install_variant("learned")
        self.reset(np.zeros(2) if xy is None else xy)

    def _install_variant(self,name):
        if name not in self.variant_states:raise ValueError("unavailable phase variant")
        self.model.load_state_dict(self.variant_states[name]);self.variant=name
        self.path=self.variant_paths[name];self.payload=self.variant_payloads[name].copy()
        self.payload["loaded_checkpoint_sha256"]=self.loaded_sources[str(self.path)]
        self.expected_runtime_digest=self.model_digest()
        self.slm_active=readonly(self.model.encoder.slm.native_gray().detach().cpu().numpy().astype(np.uint8))
        self.phase_gray_range=(int(self.slm_active.min()),int(self.slm_active.max()))
        self.phase_levels=len(np.unique(self.slm_active))
        self.decoder=CurrentCCDLinearHead(self.model.actor_head.weight.detach().cpu().numpy(),
            self.model.actor_head.bias.detach().cpu().numpy(),float(self.model.encoder.regions.reference))

    def set_variant(self,name):
        xy=self.last_reset_xy.copy()
        self.release();self._install_variant(name)
        return self.reset(xy)

    def reset(self,xy=None):
        result=super().reset(xy)
        self.last_reset_xy=self.env.data.qpos[self.env.ball_adr:self.env.ball_adr+2].copy()
        return result

    @torch.no_grad()
    def capture(self,reset=False):
        started=time.perf_counter();self.sequence+=1
        if reset:
            self.raw_history.clear();self.raw_ids.clear()
            self.raw_history.extend(self.binary.copy() for _ in range(4))
            self.raw_ids.extend([self.sequence]*4)
        else:
            self.raw_history.append(self.binary.copy());self.raw_ids.append(self.sequence)
        logical=pack_four(torch.from_numpy(np.stack(self.raw_history))[None])
        self.logical=readonly(logical[0,0].numpy())
        noise_seed=int(self.rng.integers(0,2**31-1))
        field=self.model.encoder.field(logical.to(next(self.model.parameters()).device))
        self.optical_forward_calls+=1
        corrected=native_counts(self.model.encoder.detector,field,[noise_seed])
        q=self.model.encoder.regions.full_reference(corrected)
        selected=self.model.encoder.regions(field) if not self.model.encoder.cfg.noise_enabled else q
        error=float((q-selected).abs().max())
        if error!=0:raise RuntimeError("displayed native CCD and actor readout differ")
        self.features=q
        raw=corrected[0].flip((-2,-1)).to(torch.uint8).cpu().numpy()
        self.region_counts=readonly(integrate_current_mono8(raw))
        reference=self.model.actor_mean(q)[0].cpu().numpy().clip(-.8,.8)
        self.signal_valid=bool(np.any(self.region_counts))
        if self.signal_valid:
            decoded=self.decoder.step_counts(self.region_counts)
            if float(np.abs(decoded-reference).max())>2e-5:raise RuntimeError("stateless electronic decoder mismatch")
            self.next_action=reference.copy()
        else:
            self.next_action=np.zeros(2,dtype=np.float32)
            if not self.finished:self.finished,self.end_reason=True,"missing_optical_signal"
        self.overview.update_scene(self.env.data,camera=self.overview_camera)
        self.scene_rgb=readonly(self.overview.render())
        frame=CCDFrame(self.sequence,self.episode,self.simulation_time,time.time(),noise_seed,
            readonly(raw),readonly(q[0].cpu().numpy()),error,time.perf_counter()-started)
        self.records.clear();self.records.append(HistorySlot(frame,False));self.current_ccd=frame

    def kick(self,direction,speed=.35):
        vector=np.asarray(direction,dtype=float)
        if vector.shape!=(2,) or not np.isfinite(vector).all() or np.linalg.norm(vector)==0:
            raise ValueError("finite nonzero world XY kick direction required")
        if not np.isfinite(speed) or not 0<speed<=.5:raise ValueError("simulation speed kick must be (0,.5] m/s")
        if self.finished:raise ValueError("reset the ended episode first")
        delta=vector/np.linalg.norm(vector)*speed
        dof=int(self.env.model.jnt_dofadr[self.env.model.joint("ball_free").id])
        self.env.data.qvel[dof:dof+2]+=delta;mujoco.mj_forward(self.env.model,self.env.data)
        # A kick changes velocity, not camera time: never invent a new historical frame.
        self.speed=float(np.linalg.norm(self.env.data.qvel[dof:dof+2]))
        return delta

    def step(self):
        ids=list(self.raw_ids);before=self.global_steps
        result=super().step()
        if self.global_steps>before:
            self.last_control_row.update(raw_input_frame_ids_old_to_new=ids,
                raw_input_ages_s=[.06,.04,.02,0.],ccd_history_used_by_actor=0,
                ccd_frames_retained=1,actor_parameters=18,variant=self.variant,
                optical_field_evaluations=self.optical_forward_calls)
        return result

    def snapshot(self):
        cfg=self.model.encoder.cfg
        active=self.logical.repeat(cfg.superpixel,0).repeat(cfg.superpixel,1)
        return FourFrameSnapshot(episode=self.episode,frame_id=self.current_ccd.frame_id,
            simulated_time_s=self.simulation_time,rgb=self.scene_rgb,dmd_logical=self.logical,
            dmd_active=readonly(active),slm_active=self.slm_active,history=tuple(self.records),
            next_mean_action=readonly(self.next_action),last_applied_action=readonly(self.last_applied),
            last_action_input_id=self.last_action_input_id,position=readonly(self.env.position()),
            speed=self.speed,reward_sum=self.reward_sum,stable=self.stable,finished=self.finished,
            end_reason=self.end_reason,force_steps_remaining=self.force_steps,
            capture_seconds=self.current_ccd.acquisition_seconds,
            mean_step_seconds=self.compute_seconds/max(self.global_steps,1),
            checkpoint_steps=self.payload["metadata"].get("steps",0),
            checkpoint_sha256=self.payload["loaded_checkpoint_sha256"],checkpoint_name=self.path.name,
            phase_frozen=True,propagation=cfg.propagation,grid_size=cfg.grid_size,
            device=str(next(self.model.parameters()).device),cache_error=self.current_ccd.cache_max_error,
            variant=self.variant,raw_input_ids=tuple(self.raw_ids),region_counts=self.region_counts,
            optical_forward_calls=self.optical_forward_calls,signal_valid=self.signal_valid,
            phase_gray_range=self.phase_gray_range,phase_levels=self.phase_levels)

    def metadata(self):
        return dict(role="four_raw_frames_current_CCD_simulation_dashboard",checkpoint=str(self.path),
            checkpoint_sha256=self.payload["loaded_checkpoint_sha256"],variant=self.variant,
            available_variants=list(self.variant_states),optical_config=self.model.encoder.cfg.to_dict(),
            initial_extent_m=self.extent,simulation_horizon_s=self.env.max_steps*self.env.control_dt,
            checkpoint_steps=self.payload["metadata"].get("steps",0),training_stage=0,seed=self.seed,
            hardware_enabled=False,calibrated=False,phase_frozen=True,control_dt_s=self.env.control_dt,
            policy_input="four RAW input times tiled 2x2 on one DMD; one current CCD8 vector; linear18",
            ccd_history_used_by_actor=0,ccd_frames_retained=1,optical_feature_memoization=False,
            actor_parameters=18,actor_macs=16,numerical_refinement_passed=False,
            known_main_model_sampling_action_difference_rad=self.sampling_error,release_approved=False,
            ccd_readout=dict(corrected_regions=[list(r) for r in self.model.encoder.regions.regions],
                reference_counts=float(self.model.encoder.regions.reference),
                coordinate_order="half-open [y0,y1,x0,x1], 180-degree corrected",display_overlay_only=True),
            force_frame="plate-local XY",velocity_kick_frame="world XY",input_history_frames=4,
            input_history_initialization="repeat first RAW observation at startup; repeated IDs identify fill slots",
            input_ages_s=[.06,.04,.02,0.],phase_gray_range=list(self.phase_gray_range),
            actuation_display="current CCD next clipped mean; previous filtered simulation action")

    def close(self):
        self.release();self.overview.close();self.env.close()
        if self.model_digest()!=self.expected_runtime_digest:raise RuntimeError("active frozen variant mutated")
        for filename,expected in self.loaded_sources.items():
            if rt.digest(filename)!=expected:raise RuntimeError("source checkpoint changed")


def save_fourframe_bundle(snapshot,output,metadata):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    Image.fromarray(snapshot.rgb).save(output/"simulation_overview.png")
    Image.fromarray(snapshot.dmd_logical).save(output/"DMD_four_raw_times_logical256.png")
    Image.fromarray(full_panel(snapshot.dmd_active,0)).save(output/"DMD_SIMULATION_ONLY_1920x1080.png")
    Image.fromarray(snapshot.slm_active).save(output/"SLM_SIMULATION_ONLY_UNCALIBRATED.png")
    frame=snapshot.history[0].frame
    Image.fromarray(frame.raw_counts).save(output/"CURRENT_CCD_SIMULATED_RAW.png")
    rt.json_save(output/"bundle.json",dict(metadata,deployable=False,frame_id=snapshot.frame_id,
        episode=snapshot.episode,simulation_time_s=snapshot.simulated_time_s,
        raw_input_ids=list(snapshot.raw_input_ids),current_q8=frame.features.tolist(),
        current_region_counts=snapshot.region_counts.tolist(),next_mean_action=snapshot.next_mean_action.tolist(),
        last_applied_action=snapshot.last_applied_action.tolist(),last_action_input_id=snapshot.last_action_input_id,
        displayed_native_readout_error=snapshot.cache_error,optical_forward_calls=snapshot.optical_forward_calls))
    return output
