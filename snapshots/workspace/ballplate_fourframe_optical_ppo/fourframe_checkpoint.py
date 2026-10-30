"""Independent checkpoint family; never alter pinned earlier model sources."""
from pathlib import Path
import torch
import fourframe_runtime as rt
from fourframe_model import FourFramePolicy
from optical.config import OpticalConfig


def save(path, model, metadata, optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(architecture=model.architecture_id, options=model.options,
                    optical_config=model.encoder.cfg.to_dict(), model=model.state_dict(),
                    optimizer=None if optimizer is None else optimizer.state_dict(),
                    source_sha256=rt.sources(), physical_manifest=rt.reused.physical_manifest(),
                    metadata=metadata, hardware_enabled=False, calibrated=False), temporary)
    temporary.replace(path)


def load(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rt.reused.verify_sources(payload["physical_manifest"])
    for name, expected in payload["source_sha256"].items():
        if rt.digest(name) != expected:
            raise RuntimeError(f"four-frame model source changed: {name}")
    cfg = OpticalConfig.from_dict(payload["optical_config"])
    model = FourFramePolicy(cfg, **payload["options"]).to(device)
    if payload["architecture"] != model.architecture_id:
        raise ValueError("wrong checkpoint family")
    model.load_state_dict(payload["model"])
    return model, payload
