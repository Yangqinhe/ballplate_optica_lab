"""New experiment directories only; raw and derived data remain distinct."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise ValueError("explicit safe cell ID required")
    return value


class Experiment:
    """No overwrite/resume, and no auto-delete even on failure."""
    def __init__(self, output):
        self.root = Path(output).resolve()
        # Deliberately no parents=True: a typo must not create a remote tree.
        self.root.mkdir(exist_ok=False)
        for name in ("raw", "derived", "figures", "provenance"):
            (self.root / name).mkdir()
        self.started = utc_now()
        write_json_new(self.root / "STARTED.json", {
            "started_utc": self.started, "hardware_connected": False,
            "source": "historical_real_capture_reanalysis", "complete": False,
        })

    def archive_arrays(self, cell_id, real_stack, dark_stack, models, regions=None):
        import numpy as np
        cid = safe_id(cell_id)
        raw = {"real_stack": real_stack}
        if dark_stack is not None:
            raw["dark_stack"] = dark_stack
        with (self.root / "raw" / (cid + ".npz")).open("xb") as stream:
            np.savez_compressed(stream, **raw)
        derived = {"model_" + safe_id(name): value for name, value in models.items()}
        if regions is not None:
            derived["regions"] = regions
        with (self.root / "derived" / (cid + ".npz")).open("xb") as stream:
            np.savez_compressed(stream, **derived)

    def finish(self, summary):
        write_json_new(self.root / "RESULT.json", summary)
        files = {str(p.relative_to(self.root)): sha256(p)
                 for p in sorted(self.root.rglob("*")) if p.is_file()}
        write_json_new(self.root / "MANIFEST.json", {
            "schema": "optical-lab-experiment-1", "created_utc": utc_now(),
            "file_sha256": files, "manifest_excludes_itself": True,
            "production_qualified": False, "hardware_release_approved": False,
        })


def verify_experiment(folder):
    """Verify the completed offline archive, never a hardware release."""
    root = Path(folder).resolve()
    if (root/"FAILED.json").exists():
        raise ValueError("failed experiment is not a completed evidence viewer")
    if (root/"MANIFEST.json").is_symlink() or (root/"RESULT.json").is_symlink():
        raise ValueError("archive identity files cannot be symlinks")
    result = read_json(root/"RESULT.json")
    manifest = read_json(root/"MANIFEST.json")
    if (result.get("schema") != "optical-lab-historical-assessment-1" or result.get("complete") is not True
            or manifest.get("schema") != "optical-lab-experiment-1"
            or result.get("hardware_release_approved") is not False
            or result.get("production_qualified") is not False):
        raise ValueError("completed, explicitly unqualified research archive required")
    expected = manifest["file_sha256"]
    if not {"RESULT.json", "dashboard.html", "总结.md", "cells.json", "probes.json"}.issubset(expected):
        raise ValueError("mandatory evidence files missing")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root/"MANIFEST.json"}
    if actual != set(expected):
        raise ValueError("experiment file list changed")
    for name, digest in expected.items():
        relative = Path(name)
        p = root/relative
        if relative.is_absolute() or ".." in relative.parts or p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError("archive path escapes experiment")
        if sha256(p) != digest:
            raise ValueError("experiment SHA mismatch: "+name)
    return {"verified":True, "checked_files":len(expected), "hardware_release_approved":False}
