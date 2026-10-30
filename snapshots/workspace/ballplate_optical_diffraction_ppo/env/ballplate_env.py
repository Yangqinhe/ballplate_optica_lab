import os
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco
import numpy as np

from config import ROOT, CONFIG
from env.ball_camera import BallCamera, binary_ball
from env.reward import compute_reward


class BallPlateEnv:
    def __init__(self, seed=42, initial_range=0.05, duration=15.0, xml=None, camera_fovy=None):
        self.model_xml = (ROOT / "models/ball_plate.xml").read_text() if xml is None else xml
        if camera_fovy is not None:
            if xml is not None:
                raise ValueError("checkpoint XML is authoritative; do not override its camera")
            if not 50 <= camera_fovy <= 90:
                raise ValueError("camera fovy must be in [50,90] degrees")
            root = ET.fromstring(self.model_xml)
            camera = root.find(".//camera[@name='ball_camera']")
            if camera is None:
                raise ValueError("ball_camera not found")
            camera.set("fovy", str(float(camera_fovy)))
            self.model_xml = ET.tostring(root, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.initial_range = initial_range
        self.control_dt = self.model.opt.timestep * 10
        self.max_steps = round(duration / self.control_dt)
        self.alpha = -np.expm1(-self.control_dt / 0.02)
        self.ball_adr = int(self.model.jnt_qposadr[self.model.joint("ball_free").id])
        self.sensor_adr = int(self.model.sensor_adr[self.model.sensor("ball_position").id])
        self.actuator_ids = [self.model.actuator(n).id for n in ("roll_target", "pitch_target")]
        self.servo_dofs = [int(self.model.jnt_dofadr[self.model.joint(n).id])
                           for n in ("servo_roll", "servo_pitch")]
        self.camera = BallCamera(self.model, 128, 128)

    def position(self):
        return self.data.sensordata[self.sensor_adr:self.sensor_adr + 3].copy()

    def image(self):
        return binary_ball(self.camera.capture(self.data))[None]

    def reset(self, xy=None):
        mujoco.mj_resetData(self.model, self.data)
        if xy is None:
            xy = self.rng.uniform(-self.initial_range, self.initial_range, 2)
        self.data.qpos[self.ball_adr:self.ball_adr + 2] = xy
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.action = np.zeros(2)
        self.action_delta = np.zeros(2)
        self.stable_steps = 0
        self.longest_stable = 0
        self.first_settle = None
        return self.image()

    def step(self, raw_action):
        raw_action = np.asarray(raw_action, dtype=np.float64)
        if raw_action.shape != (2,) or not np.isfinite(raw_action).all():
            raise ValueError("action must be two finite angles")
        previous_action = self.action.copy()
        previous_position = self.position()
        self.action += self.alpha * (np.clip(raw_action, -0.8, 0.8) - self.action)
        self.data.ctrl[self.actuator_ids] = self.action
        mujoco.mj_step(self.model, self.data, nstep=10)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        p = self.position()
        terminated = bool(abs(p[0]) > 0.27 or abs(p[1]) > 0.27 or p[2] < -0.05)
        truncated = self.steps >= self.max_steps and not terminated
        velocity = (p - previous_position) / self.control_dt
        reward, terms, info = compute_reward(
            p, velocity, self.action, previous_action, terminated,
            CONFIG["reward"], previous_action_delta=self.action_delta,
        )
        self.action_delta = self.action - previous_action
        self.stable_steps = self.stable_steps + 1 if info["stable"] else 0
        self.longest_stable = max(self.longest_stable, self.stable_steps)
        if self.first_settle is None and self.stable_steps >= 25:
            self.first_settle = self.steps * self.control_dt
        info.update(terminated=terminated, truncated=truncated, reward_terms=terms,
                    first_settle=self.first_settle, longest_stable=self.longest_stable * self.control_dt,
                    servo_speed=float(np.sqrt(np.mean(self.data.qvel[self.servo_dofs] ** 2))))
        return self.image(), reward, terminated, truncated, info

    def close(self):
        self.camera.close()

