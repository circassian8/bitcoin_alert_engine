from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from btc_alert_engine import __version__

DEFAULT_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git", ".idea", ".venv", "node_modules", "reports"}
DEFAULT_EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".DS_Store"}


def _sha256_bytes(parts: Iterable[bytes]) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part)
    return hasher.hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def metadata_tree_hash(root: str | Path, *, exclude_dirs: set[str] | None = None, exclude_suffixes: set[str] | None = None) -> str:
    root_path = Path(root)
    exclude_dirs = exclude_dirs or DEFAULT_EXCLUDE_DIRS
    exclude_suffixes = exclude_suffixes or DEFAULT_EXCLUDE_SUFFIXES
    if not root_path.exists():
        return _sha256_bytes([b"missing"])
    records: list[bytes] = []
    for path in sorted(root_path.rglob("*")):
        if any(part in exclude_dirs for part in path.parts):
            continue
        if path.is_dir():
            continue
        if path.suffix in exclude_suffixes:
            continue
        stat = path.stat()
        rel = path.relative_to(root_path)
        records.append(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
    return _sha256_bytes(records)


def source_tree_hash(project_root: str | Path) -> str:
    return metadata_tree_hash(Path(project_root) / "src")


def data_state_hash(data_dir: str | Path) -> str:
    # use metadata hash to avoid re-hashing large raw files while still detecting drift
    return metadata_tree_hash(Path(data_dir), exclude_dirs={"__pycache__", ".pytest_cache"}, exclude_suffixes=DEFAULT_EXCLUDE_SUFFIXES)


def current_git_commit(project_root: str | Path) -> str | None:
    root = Path(project_root)
    git_dir = root / ".git"
    if not git_dir.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def report_provenance(
    *,
    project_root: str | Path,
    data_dir: str | Path,
    registry_path: str | Path,
    contracts_path: str | Path | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    data_dir = Path(data_dir).resolve()
    registry_path = Path(registry_path).resolve()
    contracts_path = Path(contracts_path).resolve() if contracts_path is not None and Path(contracts_path).exists() else None
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "project_root": str(project_root),
        "git_commit": current_git_commit(project_root),
        "source_tree_sha256": source_tree_hash(project_root),
        "data_dir": str(data_dir),
        "data_state_sha256": data_state_hash(data_dir),
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
    }
    if contracts_path is not None:
        payload["feature_contracts_path"] = str(contracts_path)
        payload["feature_contracts_sha256"] = sha256_file(contracts_path)
    return payload


def compare_provenance(current: dict[str, Any], recorded: dict[str, Any] | None) -> dict[str, Any]:
    if not recorded:
        return {
            "has_provenance": False,
            "registry_match": False,
            "source_tree_match": False,
            "data_state_match": False,
            "feature_contracts_match": False,
        }
    return {
        "has_provenance": True,
        "registry_match": recorded.get("registry_sha256") == current.get("registry_sha256"),
        "source_tree_match": recorded.get("source_tree_sha256") == current.get("source_tree_sha256"),
        "data_state_match": recorded.get("data_state_sha256") == current.get("data_state_sha256"),
        "feature_contracts_match": (
            current.get("feature_contracts_sha256") is None
            or recorded.get("feature_contracts_sha256") == current.get("feature_contracts_sha256")
        ),
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
