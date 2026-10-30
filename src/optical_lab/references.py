"""Read-only, stdlib-only identities for two explicitly selected old references.

Nothing here imports old project code, deserializes checkpoints, launches a
process, initializes a device, or grants a deployment qualification. Paths in
the manifests are relative to the sibling-project workspace, not the cwd.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

SCHEMA = "optical-lab-readonly-reference-1"
PROFILES = ("simulation", "hardware")
LAB_ROOT = Path(__file__).resolve().parents[2]
REFERENCES_ROOT = LAB_ROOT / "references"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("canonical relative POSIX path required")
    path = PurePosixPath(value)
    if path.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")):
        raise ValueError("absolute or escaping reference path rejected")
    return path


def _entry(value: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(value, dict) or not _SHA.fullmatch(str(value.get("sha256", ""))):
        raise ValueError("each reference requires an explicit lowercase SHA256")
    return str(_relative(value["path"])), value["sha256"]


def load_manifest(profile: str, references_root: Path | str | None = None) -> dict[str, Any]:
    """Load a small identity manifest only; do not import any referenced file."""
    if profile not in PROFILES:
        raise ValueError("profile must be simulation or hardware")
    root = Path(references_root) if references_root is not None else REFERENCES_ROOT
    manifest = json.loads((root / profile / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("profile") != profile:
        raise ValueError("reference schema/profile mismatch")
    legacy = _relative(manifest["legacy_project"])
    if len(legacy.parts) != 1 or manifest.get("launch_enabled") is not False:
        raise ValueError("single sibling legacy project and disabled launch required")
    if not isinstance(manifest.get("artifacts"), list) or not manifest["artifacts"]:
        raise ValueError("an explicit nonempty artifact list is required")
    identifiers: set[str] = set()
    for entry in manifest["artifacts"]:
        _entry(entry)
        identity = entry.get("id")
        if not isinstance(identity, str) or not identity or identity in identifiers:
            raise ValueError("unique nonempty artifact IDs required")
        identifiers.add(identity)
    for group in manifest.get("source_groups", []):
        if not isinstance(group.get("files"), list) or len(group["files"]) != group.get("expected_count"):
            raise ValueError("source group count mismatch")
        paths = [_entry(x)[0] for x in group["files"]]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate source group paths")
        if group.get("from_artifact") not in identifiers:
            raise ValueError("source group must bind a listed historical manifest")
    return manifest


def verify(profile: str, legacy_root: Path | str | None = None,
           references_root: Path | str | None = None) -> dict[str, Any]:
    """Hash only listed files and frozen source identities, without executing them.

    ``legacy_root`` can relocate the whole original sibling-project layout.
    Sibling dependencies are resolved under its parent. Symlinks resolving out
    of that workspace fail closed. No glob, newest-file or fallback selection.
    """
    manifest = load_manifest(profile, references_root)
    legacy = Path(legacy_root).resolve() if legacy_root is not None else (
        LAB_ROOT.parent / manifest["legacy_project"]).resolve()
    workspace = legacy.parent.resolve()
    expected: dict[str, str] = {}
    artifacts = {x["id"]: x for x in manifest["artifacts"]}
    for entry in [*manifest["artifacts"], *[x for g in manifest.get("source_groups", []) for x in g["files"]]]:
        path, sha = _entry(entry)
        if path in expected and expected[path] != sha:
            raise ValueError("conflicting frozen identities for " + path)
        expected[path] = sha
    checked: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for name, sha in expected.items():
        parts = PurePosixPath(name).parts
        target = (legacy.joinpath(*parts[1:]) if parts[0] == manifest["legacy_project"]
                  else workspace.joinpath(*parts)).resolve()
        row: dict[str, Any] = {"path": name, "expected_sha256": sha, "exists": False,
                               "actual_sha256": None, "matches": False}
        try:
            target.relative_to(workspace)
            if not target.is_file():
                row["error"] = "missing_regular_file"
            else:
                row["exists"] = True
                digest = hashlib.sha256()
                # Only historical JSON ledgers need bytes in memory; never load
                # a checkpoint, image array, SDK library, or original dataset.
                capture = name in {artifacts[g["from_artifact"]]["path"]
                                   for g in manifest.get("source_groups", [])}
                chunks: list[bytes] | None = [] if capture else None
                with target.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                        if chunks is not None:
                            chunks.append(block)
                row["actual_sha256"] = digest.hexdigest()
                row["matches"] = row["actual_sha256"] == sha
                if chunks is not None and row["matches"]:
                    payloads[name] = b"".join(chunks)
        except (OSError, ValueError) as exc:
            row["error"] = type(exc).__name__ + ": " + str(exc)
        checked.append(row)
    groups: list[dict[str, Any]] = []
    for group in manifest.get("source_groups", []):
        row = {"id": group["id"], "expected_count": group["expected_count"], "matches": False}
        try:
            ledger = artifacts[group["from_artifact"]]["path"]
            source = json.loads(payloads[ledger])[group["json_key"]]
            prefix = PurePosixPath(group["original_workspace_prefix"])
            converted = {str(PurePosixPath(p).relative_to(prefix)): sha for p, sha in source.items()}
            frozen = {x["path"]: x["sha256"] for x in group["files"]}
            row["matches"] = len(source) == group["expected_count"] and converted == frozen
        except (KeyError, TypeError, ValueError) as exc:
            row["error"] = type(exc).__name__ + ": " + str(exc)
        groups.append(row)
    return {"schema": "optical-lab-reference-verification-1", "profile": profile,
            "legacy_root": str(legacy), "checked_count": len(checked), "files": checked,
            "source_groups": groups,
            "verified": all(x["matches"] for x in [*checked, *groups]),
            "scope": "file_identity_only_NOT_control_or_hardware_acceptance",
            "old_code_imported": False, "checkpoint_deserialized": False,
            "process_launched": False, "hardware_accessed": False,
            "production_qualified": False, "release_approved": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--profile", choices=(*PROFILES, "all"), default="all")
    parser.add_argument("--legacy-root", type=Path)
    args = parser.parse_args(argv)
    reports = [verify(p, args.legacy_root) for p in (PROFILES if args.profile == "all" else (args.profile,))]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0 if all(x["verified"] for x in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
