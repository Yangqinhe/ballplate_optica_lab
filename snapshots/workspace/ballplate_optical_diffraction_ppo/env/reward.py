"""居中、停稳与舵机平滑奖励；所有分项均返回加权后的值。"""
import numpy as np


def compute_reward(position, velocity, action, previous_action, terminated,
                   config, edge_limit=0.27, action_limit=0.8, previous_action_delta=None):
    """输入板面坐标位置/速度和已限幅目标角度，不改变策略的原始动作。"""
    distance = float(np.linalg.norm(position[:2]))
    speed = float(np.linalg.norm(velocity[:2]))
    delta = action - previous_action
    delta_rms = float(np.sqrt(np.mean(delta ** 2)))
    previous_delta = np.zeros_like(delta) if previous_action_delta is None else previous_action_delta
    acceleration_rms = float(np.sqrt(np.mean((delta - previous_delta) ** 2)))
    center = np.exp(-0.5 * (distance / config["center_radius"]) ** 2)
    stable_quality = (center
                      * np.exp(-0.5 * (speed / config["stable_speed"]) ** 2)
                      * np.exp(-0.5 * (delta_rms / config["stable_action_delta"]) ** 2))
    speed_cost = (speed / config["speed_scale"]) ** 2
    action_cost = (delta_rms / config["action_delta_scale"]) ** 2
    # 球速惩罚保留有界形式；v2 动作代价用于旧检查点评估兼容。
    terms = {
        "broad_center": config["broad_weight"] * np.exp(
            -0.5 * (distance / config["broad_radius"]) ** 2),
        "precise_center": config["center_weight"] * center,
        "stable_center": config["stable_weight"] * stable_quality,
        "distance_cost": -config["distance_weight"] * min((distance / edge_limit) ** 2, 1.0),
        "speed_cost": -config["speed_weight"] * speed_cost / (1.0 + speed_cost),
        "action_rate_cost": -config["action_rate_weight"] * action_cost / (1.0 + action_cost),
        "action_size_cost": -config["action_size_weight"] * float(np.mean((action / action_limit) ** 2)),
        "unsettled_cost": -config["unsettled_weight"] * (1.0 - stable_quality),
        "fall": -config["fall_penalty"] if terminated else 0.0,
    }
    if config["version"] >= 3:
        # pseudo-Huber：小变化近似二次，大变化近似线性；不像 z/(1+z) 那样饱和。
        # 在中心附近更强，外圈仍允许较快纠偏。二阶差分进一步惩罚持续反向。
        outer_weight = config["outer_action_multiplier"] if config["version"] >= 4 else 1.0
        weight = outer_weight + config["center_action_multiplier"] * center
        terms["action_rate_cost"] = -config["action_rate_weight"] * weight * (
            np.sqrt(1.0 + action_cost) - 1.0
        )
        terms["action_acceleration_cost"] = -config["action_acceleration_weight"] * weight * (
            np.sqrt(1.0 + (acceleration_rms / config["action_delta_scale"]) ** 2) - 1.0
        )
    if config["version"] >= 4:
        # 外圈不因持续的大目标角度受罚，救球主要由任务奖励引导。
        terms["action_size_cost"] *= center
    if config["version"] >= 5:
        # 制动信号：高速穿过中心并不等于停稳；远离中心时进一步扣分。
        terms["speed_cost"] = -config["speed_weight"] * speed_cost
        outward = max(float(np.dot(position[:2], velocity[:2])), 0.0)
        terms["outward_cost"] = -config["outward_weight"] * outward / (
            config["broad_radius"] * config["speed_scale"]
        )
    if terminated:
        # 掉球这一帧不能因经过中心或低速而继续获得正奖励。
        for name in ("broad_center", "precise_center", "stable_center"):
            terms[name] = 0.0
    stable = bool(not terminated and distance < config["center_radius"]
                  and speed < config["stable_speed"]
                  and delta_rms < config["stable_action_delta"])
    return float(sum(terms.values())), terms, {
        "distance": distance, "speed": speed, "action_delta_rms": delta_rms,
        "action_acceleration_rms": acceleration_rms,
        "stable": stable,
    }

