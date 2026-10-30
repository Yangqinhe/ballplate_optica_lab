"""Explicit offline intensity preprocessing and eight sums, not a GUI hook.

These are deterministic electronic operators. They do not establish photon
SNR improvement or turn a model mean into a noisy camera sample. Original
camera/model arrays remain unchanged; no CCD history is accumulated.
"""
from copy import deepcopy
from numbers import Real

import numpy as np

SCHEMA = "optical-lab-intensity-preprocessing-1"
CONFIG_REQUESTED = dict(schema=SCHEMA, mode="requested_clip", gain=9.5,
                        black_ADU=10.5, gamma=1., output="float64")
_KEYS = frozenset(("schema", "mode", "gain", "black_ADU", "gamma", "output"))


def _number(value, lower, upper, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or not lower <= value <= upper):
        raise ValueError("finite bounded nonboolean numeric " + name + " required")
    return float(value)


def validate_config(config):
    """Exact serialized research contract; no implicit gain/threshold fallback."""
    if (not isinstance(config, dict) or set(config) != _KEYS
            or config["schema"] != SCHEMA or config["mode"] != "requested_clip"
            or config["output"] not in ("float64", "display_uint8")):
        raise ValueError("exact explicit preprocessing schema/keys/mode/output required")
    return dict(schema=SCHEMA, mode="requested_clip",
        gain=_number(config["gain"], 1., 128., "gain"),
        black_ADU=_number(config["black_ADU"], 0., 255., "black_ADU"),
        gamma=_number(config["gamma"], .3, 3., "gamma"), output=config["output"])


def _image(image):
    a = np.asarray(image)
    if (a.ndim not in (2, 3) or min(a.shape) <= 0
            or not (a.dtype == np.uint8 or np.issubdtype(a.dtype, np.floating))
            or not np.isfinite(a).all() or np.any(a < 0)):
        raise ValueError("finite nonnegative uint8 or floating-ADU HxW / NxHxW image required")
    return a.astype(np.float64, copy=True)


def preprocess_intensity(image, config=CONFIG_REQUESTED):
    """255 * clip((I-black)*gain/255, 0, 1)**gamma, with no input mutation.

    ``float64`` preserves fractional transformed values and performs NO
    rounding. ``display_uint8`` exactly rounds that result to an independent
    display-compatible byte buffer. Both have the explicitly requested upper
    software saturation, unlike an unbounded floating ReLU experiment.
    """
    c = validate_config(config)
    a = _image(image)
    level = np.clip((a-c["black_ADU"])*c["gain"]/255., 0., 1.)
    processed = np.power(level, c["gamma"])*255.
    return np.rint(processed).astype(np.uint8) if c["output"] == "display_uint8" else processed


def integrate_pixels(image, weights):
    """Independent per-frame float64 sums; no division, averaging or history.

    Nonnegative disjoint weights may be fractional, but no pixel may belong
    to more than one region. Full ROI coverage is an external contract, not
    silently assumed by this arithmetic helper.
    """
    a = _image(image)
    w = np.asarray(weights, dtype=np.float64)
    if (w.shape != (8, *a.shape[-2:]) or not np.isfinite(w).all()
            or np.any(w < 0) or np.any(w > 1) or np.any(np.count_nonzero(w, axis=0) > 1)):
        raise ValueError("eight finite [0,1] disjoint matching-plane weights required")
    return np.einsum("khw,...hw->...k", w, a, optimize=True)


def linear_gain_head(head, gain):
    """Algebraic counterfactual ONLY: W/g preserves Wf+b if new f is g*f.

    It does NOT support black subtraction, thresholding, clipping, gamma,
    different pixel regions/normalization, or any sensor-domain substitution.
    The returned copy is deliberately not an eligible old deployment HEAD.
    """
    g = _number(gain, 1., 128., "gain")
    if not isinstance(head, dict):
        raise ValueError("explicit raw-integral head dictionary required")
    try:
        w = np.asarray(head["weight_raw_ADU_to_rad"], dtype=np.float64)
        b = np.asarray(head["bias_rad"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("raw18 weight/bias required") from exc
    if w.shape != (2, 8) or b.shape != (2,) or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("finite Linear8to2 /18 coefficients required")
    answer = deepcopy(head)
    answer.update(schema="optical-lab-pure-linear-gain-head-counterfactual-1",
        weight_raw_ADU_to_rad=(w/g).tolist(), bias_rad=b.tolist(),
        release_approved=False, production_qualified=False, control_release_approved=False,
        real_hardware_features=False, short_MuJoCo_pilot_eligible=False,
        algebraic_transform=dict(source_schema=head.get("schema"), gain=g,
            rule="new_features=gain*original_features; new_weight=original_weight/gain; bias unchanged",
            supports_requested_clip=False, requires_threshold_free_linear_intensity=True,
            verified_on_actual_camera=False, head_refitted=False))
    return answer
