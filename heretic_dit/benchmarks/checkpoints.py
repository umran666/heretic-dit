"""Checkpoint provenance: lockfile recording, hashing, and verification.

``scripts/prepare_erased_checkpoints.py`` writes a ``checkpoints.lock.yaml``
that records, for every erased checkpoint used in the paper: source (HF repo
or git URL), pinned revision / commit hash, seed (for checkpoints we train
ourselves), and the SHA-256 of every file. Consumers must verify against the
lockfile before use -- a hash mismatch is a hard error, never a silent
substitution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

__all__ = ["CheckpointEntry", "hash_file", "write_lockfile", "load_lockfile", "verify_checkpoint"]

LOCKFILE_VERSION = 1


@dataclass
class CheckpointEntry:
    """Provenance record for one erased checkpoint artifact.

    Attributes:
        method: erasure method that produced the checkpoint (esd / uce / mace).
        base_model: base model identifier (e.g. ``sd15``).
        concept: erased concept name (or ``"<multi>"``).
        source: where the artifact came from (URL, ``hf:<repo>``, or git URL).
        origin: ``released`` (downloaded) or ``trained`` (produced locally by
            running the official training scripts).
        files: mapping of relative filename -> SHA-256 hex digest.
        revision: HF revision (for ``hf:<repo>`` sources), else None.
        commit: git commit of the training repo (for ``trained`` artifacts).
        seed: training seed, or None for released artifacts.
        command: exact training command used, for ``trained`` artifacts.
        notes: free-form provenance notes.
    """

    method: str
    base_model: str
    concept: str
    source: str
    origin: str
    files: Dict[str, str] = field(default_factory=dict)
    revision: Optional[str] = None
    commit: Optional[str] = None
    seed: Optional[int] = None
    command: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.origin not in ("released", "trained"):
            raise ValueError(f"origin must be 'released' or 'trained'; got {self.origin!r}.")
        if self.origin == "released" and self.seed is not None:
            raise ValueError("released artifacts must not carry a training seed.")

    @property
    def run_key(self) -> str:
        return f"{self.method}/{self.base_model}/{self.concept}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CheckpointEntry":
        return cls(**payload)


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_lockfile(path: str | Path, entries: List[CheckpointEntry]) -> Path:
    """Write or update ``checkpoints.lock.yaml`` (sorted, deterministic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, CheckpointEntry] = {}
    if path.exists():
        for entry in load_lockfile(path):
            existing[entry.run_key] = entry
    for entry in entries:
        existing[entry.run_key] = entry
    ordered = [existing[key] for key in sorted(existing)]
    payload = {
        "version": LOCKFILE_VERSION,
        "entries": [e.to_dict() for e in ordered],
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def load_lockfile(path: str | Path) -> List[CheckpointEntry]:
    """Load all entries from a lockfile."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lockfile not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if int(payload.get("version", 0)) != LOCKFILE_VERSION:
        raise ValueError(f"Unsupported lockfile version in {path}.")
    return [CheckpointEntry.from_dict(e) for e in payload.get("entries", [])]


def verify_checkpoint(entry: CheckpointEntry, checkpoint_dir: str | Path) -> None:
    """Verify that every file of ``entry`` exists with the locked hash.

    Raises:
        FileNotFoundError: if a file is missing.
        ValueError: if a file's hash differs from the lockfile (a silent
            substitution attempt).
    """
    base = Path(checkpoint_dir)
    for relative, locked_hash in entry.files.items():
        file_path = base / relative
        if not file_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file missing for {entry.run_key}: {file_path}"
            )
        actual = hash_file(file_path)
        if actual != locked_hash:
            raise ValueError(
                f"Hash mismatch for {entry.run_key}:{relative} -- lockfile expects "
                f"{locked_hash}, found {actual}. Refusing to substitute a different "
                "checkpoint; re-fetch deliberately with prepare_erased_checkpoints.py."
            )


def entry_json(entry: CheckpointEntry) -> str:
    return json.dumps(entry.to_dict(), sort_keys=True)
