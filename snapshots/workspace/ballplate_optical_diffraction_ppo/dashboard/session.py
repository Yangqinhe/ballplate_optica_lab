"""Frozen deterministic optical policy with synchronized display/history data."""
import collections
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from env.ballplate_env import BallPlateEnv
from rl.electrical_networks import FeatureHistory
from training import load_model
from dashboard.acquisition import readonly, SimulationAcquisition
from dashboard.interfaces import HardwareAcquisition, HistorySlot


@dataclass(frozen=True)
class Snapshot:
    episode: int
    frame_id: int
    simulated_time_s: float
    rgb: np.ndarray
    dmd_logical: np.ndarray
    dmd_active: np.ndarray
    slm_active: np.ndarray
    history: tuple
    next_mean_action: np.ndarray
    last_applied_action: np.ndarray
    last_action_input_id: int | None
    position: np.ndarray
    speed: float
    reward_sum: float
    stable: bool
    finished: bool
    end_reason: str | None
    force_steps_remaining: int
    capture_seconds: float
    mean_step_seconds: float
    checkpoint_steps: int
    checkpoint_sha256: str
    checkpoint_name: str
    phase_frozen: bool
    propagation: str
    grid_size: int
    device: str
    cache_error: float


def full_panel(active, background):
    h, w = active.shape
    if h > 1080 or w > 1920:
        raise ValueError("active image exceeds reference panel")
    panel = np.full((1080,1920), background, dtype=np.uint8)
    y, x = (1080-h)//2, (1920-w)//2
    panel[y:y+h, x:x+w] = active
    return panel


class DashboardSession:
    def __init__(self, checkpoint, seed=90001, extent=.05, duration=15.,
                 device="cuda", backend="simulation", xy=None):
        if backend != "simulation":
            HardwareAcquisition()
        if not 0 < extent <= .23 or not 0 < duration <= 3600:
            raise ValueError("extent in (0,.23], duration in (0,3600] required")
        torch.set_num_threads(2)
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; use --device cpu")
            torch.cuda.set_per_process_memory_fraction(.45)
        self.path = Path(checkpoint).resolve()
        self.model, self.payload = load_model(self.path, device)
        self.model.eval().requires_grad_(False)
        self.initial_model_digest = self.model_digest()
        self.extent, self.seed = extent, seed
        self.rng = np.random.default_rng(seed)
        self.env = BallPlateEnv(seed=seed, initial_range=extent, duration=duration,
                                xml=self.payload["model_xml"])
        self.acquisition = SimulationAcquisition(self.model)
        self.cache = FeatureHistory()
        self.records = collections.deque(maxlen=8)
        self.episode, self.sequence, self.global_steps = 0, -1, 0
        self.compute_seconds = 0.
        self.force_steps = 0
        self.force_local = np.zeros(3)
        self.ball_id = self.env.model.body("ball").id
        self.plate_id = self.env.model.body("plate").id
        self.slm_active = readonly(self.model.encoder.slm.native_gray().cpu().numpy().astype(np.uint8))
        self.last_control_row = None
        self.reset(np.zeros(2) if xy is None else xy)

    def model_digest(self):
        digest = hashlib.sha256()
        for name,tensor in sorted(self.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @property
    def simulation_time(self):
        return self.env.steps*self.env.control_dt

    @torch.no_grad()
    def capture(self, reset=False):
        self.sequence += 1
        seed = int(self.rng.integers(0,2**31-1))
        q, frame = self.acquisition.acquire(self.binary, self.sequence, self.episode,
                                            self.simulation_time, seed)
        if reset:
            self.features = self.cache.reset(q, (self.payload["loaded_checkpoint_sha256"],self.episode))
            self.records.clear()
            self.records.extend([HistorySlot(frame,True)]*7+[HistorySlot(frame,False)])
        else:
            self.features = self.cache.push(q, (self.payload["loaded_checkpoint_sha256"],self.episode))
            self.records.append(HistorySlot(frame,False))
        from_records = np.stack([slot.frame.features for slot in self.records])
        if not np.array_equal(from_records,self.features[0].cpu().numpy()):
            raise RuntimeError("display history IDs and policy history are not synchronized")
        self.next_action = self.model.actor_mean(self.features)[0].cpu().numpy().copy()
        self.current_ccd = frame

    @torch.no_grad()
    def reset(self, xy=None):
        self.episode += 1
        self.binary = self.env.reset(xy)
        self.finished,self.end_reason = False,None
        self.reward_sum,self.speed = 0.,0.
        self.stable = False
        self.last_applied = np.zeros(2)
        self.last_action_input_id = None
        self.force_steps,self.force_local = 0,np.zeros(3)
        self.env.data.xfrc_applied[self.ball_id] = 0
        self.capture(reset=True)
        return self.snapshot()

    def push(self, direction, force=.1, duration=.1):
        direction = np.asarray(direction,dtype=float)
        if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
            raise ValueError("finite nonzero plate-local XY direction required")
        if not np.isfinite([force,duration]).all() or not .01 <= force <= .5 or not .02 <= duration <= .5:
            raise ValueError("force .01-.5N, duration .02-.5s required")
        if self.finished:
            raise RuntimeError("reset after an ended episode")
        self.force_local = np.r_[direction/np.linalg.norm(direction)*force,0.]
        self.force_steps = int(np.ceil(duration/self.env.control_dt-1e-10))

    def release(self):
        self.force_steps,self.force_local = 0,np.zeros(3)
        self.env.data.xfrc_applied[self.ball_id] = 0

    @torch.no_grad()
    def step(self):
        if self.finished:
            return self.snapshot()
        started = time.perf_counter()
        observed_id = self.current_ccd.frame_id
        history_ids = [slot.frame.frame_id for slot in self.records]
        observed_time = self.simulation_time
        action = self.next_action.copy()
        before = self.env.position()
        if self.force_steps:
            world_force = self.env.data.xmat[self.plate_id].reshape(3,3)@self.force_local
        else:
            world_force = np.zeros(3)
        self.env.data.xfrc_applied[self.ball_id,:3] = world_force
        self.env.data.xfrc_applied[self.ball_id,3:] = 0
        try:
            self.binary,reward,terminated,truncated,info = self.env.step(action)
        finally:
            self.env.data.xfrc_applied[self.ball_id] = 0
        self.last_applied = self.env.action.copy()
        self.last_action_input_id = observed_id
        self.speed = info["speed"]
        self.stable = info["stable"]
        self.reward_sum += reward
        self.force_steps = max(0,self.force_steps-1)
        self.finished = terminated or truncated
        self.end_reason = "fell" if terminated else "episode_complete" if truncated else None
        self.capture()
        self.global_steps += 1
        elapsed = time.perf_counter()-started
        self.compute_seconds += elapsed
        self.last_control_row = {"episode":self.episode,"input_frame_id":observed_id,
            "observation_time_s":observed_time,"history_ids_old_to_new":history_ids,
            "applied_interval_s":[observed_time,self.simulation_time],"mean_action":action.tolist(),
            "executed_action":self.last_applied.tolist(),"result_frame_id":self.current_ccd.frame_id,
            "position_diagnostic":self.env.position().tolist(),"speed_diagnostic":self.speed,
            "reward":reward,"stable":self.stable,"force_world_n":world_force.tolist(),
            "cache_max_error":self.current_ccd.cache_max_error,"wall_seconds":elapsed}
        return self.snapshot()

    def snapshot(self):
        active = np.repeat(np.repeat(self.binary[0],self.model.encoder.cfg.superpixel,0),
                            self.model.encoder.cfg.superpixel,1)
        return Snapshot(self.episode,self.current_ccd.frame_id,self.simulation_time,
            readonly(self.env.camera.last_rgb),readonly(self.binary[0]),readonly(active),self.slm_active,
            tuple(self.records),readonly(self.next_action),readonly(self.last_applied),
            self.last_action_input_id,readonly(self.env.position()),self.speed,self.reward_sum,
            self.stable,self.finished,self.end_reason,self.force_steps,self.current_ccd.acquisition_seconds,
            self.compute_seconds/max(self.global_steps,1),self.payload["state"]["steps"],
            self.payload["loaded_checkpoint_sha256"],self.path.name,
            self.payload["config"]["phase_frozen"],self.model.encoder.cfg.propagation,
            self.model.encoder.cfg.grid_size,str(next(self.model.parameters()).device),
            self.current_ccd.cache_max_error)

    def metadata(self):
        return {"role":"full_simulation_dashboard_not_hardware_or_release",
            "checkpoint":str(self.path),"checkpoint_sha256":self.payload["loaded_checkpoint_sha256"],
            "checkpoint_steps":self.payload["state"]["steps"],"optical_config":self.model.encoder.cfg.to_dict(),
            "policy_input":"original optical encoder features, 8 frames; no privileged state",
            "ccd_orientation":"native sensor image shown; policy fixed 180-degree correction",
            "phase_frozen":self.payload["config"]["phase_frozen"],"initial_extent_m":self.extent,
            "training_stage":self.payload["state"]["stage"],
            "seed":self.seed,"hardware_enabled":False,"control_dt_s":self.env.control_dt,
            "history_initialization":"same first acquisition repeated, explicitly labeled fill",
            "actuation_display":"next mean for current CCD; last actual filtered action from preceding input"}

    def close(self):
        self.release()
        self.acquisition.close()
        self.env.close()
        if self.model_digest() != self.initial_model_digest:
            raise RuntimeError("frozen model mutated during dashboard inference")
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.payload["loaded_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint changed during dashboard session")


def save_bundle(snapshot, output, metadata):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    Image.fromarray(snapshot.rgb).save(output/"simulation_rgb.png")
    Image.fromarray(snapshot.dmd_logical).save(output/"dmd_logical_128.png")
    Image.fromarray(full_panel(snapshot.dmd_active,0)).save(output/"dmd_SIMULATION_PANEL.png")
    Image.fromarray(full_panel(snapshot.slm_active,metadata["optical_config"]["slm_background_gray"])).save(
        output/"slm_SIMULATION_ONLY_NOT_CALIBRATED.png")
    stored = set()
    history = []
    for age,slot in enumerate(reversed(snapshot.history)):
        frame = slot.frame
        filename = f"ccd_{frame.frame_id:08d}_SIMULATED_RAW.png"
        if frame.frame_id not in stored:
            Image.fromarray(frame.raw_counts).save(output/filename)
            stored.add(frame.frame_id)
        history.append({"age_index":age,"frame_id":frame.frame_id,"episode":frame.episode,
            "simulated_time_s":frame.simulated_time_s,"captured_unix_s":frame.captured_unix_s,
            "noise_seed":frame.noise_seed,"repeated_fill":slot.repeated_fill,
            "features":frame.features.tolist(),"image":filename})
    payload = {**metadata,"deployable":False,"frame_id":snapshot.frame_id,"episode":snapshot.episode,
        "simulation_time_s":snapshot.simulated_time_s,"history_current_to_old":history,
        "next_mean_action":snapshot.next_mean_action.tolist(),
        "last_applied_action":snapshot.last_applied_action.tolist(),
        "last_action_input_id":snapshot.last_action_input_id,"cache_max_error":snapshot.cache_error}
    (output/"bundle.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return output
