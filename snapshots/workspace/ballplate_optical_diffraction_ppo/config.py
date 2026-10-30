"""Preserved MuJoCo reward; separate uncalibrated optical experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = {
    "seed": 42, "device": "cpu", "torch_threads": 2,
    "max_steps": 2_000_000, "eval_every": 25_000, "eval_episodes": 50,
    "development_seed": 71001, "screening_seed": 72001,
    "epochs": 4, "chunk_size": 64, "learning_rate": 1e-4,
    "gamma": 0.99, "lam": 0.95, "clip": 0.2,
    "value_coef": 0.5, "entropy_coef": 0.0, "max_grad_norm": 0.5,
    "gradient_clip_mode": "joint",
    "initial_std_cap": 0.15,
    "rollout_min_steps": 0, "num_mini_batches": 1,
    "critic_encoder_grad_scale": 0.0,
    "training_recipe": "reference", "initialization": "reference",
    "critic_learning_rate": None, "target_kl": None,
    "representation_learning_rate": None,
    "initial_ratio_atol": 2e-4,
    "camera_fovy": None,
    "reward": {
        "version": 5, "broad_radius": 0.18, "center_radius": 0.03,
        "stable_speed": 0.05, "stable_action_delta": 0.04,
        "speed_scale": 0.25, "action_delta_scale": 0.10,
        "broad_weight": 0.6, "center_weight": 1.0, "stable_weight": 1.4,
        "distance_weight": 0.4, "speed_weight": 0.6, "outward_weight": 0.4,
        "action_rate_weight": 0.15, "center_action_multiplier": 3.9,
        "outer_action_multiplier": 0.1, "action_acceleration_weight": 0.05,
        "action_size_weight": 0.02, "unsettled_weight": 0.05,
        "fall_penalty": 250.0, "settle_seconds": 0.5,
    },
    "initial_ranges": [0.05, 0.12, 0.18, 0.23],
}
