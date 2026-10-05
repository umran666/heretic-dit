"""Tests for checkpoint lockfile generation, verification, and the prep script."""

from __future__ import annotations

from pathlib import Path

import pytest

from heretic_dit.benchmarks.checkpoints import (
    CheckpointEntry,
    hash_file,
    load_lockfile,
    verify_checkpoint,
    write_lockfile,
)
from scripts.prepare_erased_checkpoints import ARTIFACTS, prepare_artifact


def _entry(tmp_path: Path, content: bytes = b"checkpoint-bytes") -> CheckpointEntry:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    (checkpoint_dir / "model.safetensors").write_bytes(content)
    return CheckpointEntry(
        method="esd",
        base_model="sd15",
        concept="van gogh",
        source="hf:rohitgandikota/erasing-models",
        origin="released",
        files={"model.safetensors": hash_file(checkpoint_dir / "model.safetensors")},
    )


def test_hash_file_is_stable_sha256(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert hash_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_lockfile_roundtrip_is_sorted_and_deterministic(tmp_path):
    b = CheckpointEntry(method="uce", base_model="sd15", concept="church", source="x", origin="released")
    a = CheckpointEntry(method="esd", base_model="sd15", concept="church", source="y", origin="released")
    path = write_lockfile(tmp_path / "checkpoints.lock.yaml", [b, a])
    entries = load_lockfile(path)
    assert [e.run_key for e in entries] == sorted([a.run_key, b.run_key])
    # Writing again with an overlapping entry updates rather than duplicates.
    updated = CheckpointEntry(method="esd", base_model="sd15", concept="church", source="y2", origin="released")
    write_lockfile(path, [updated])
    entries = load_lockfile(path)
    assert len(entries) == 2
    assert {e.source for e in entries if e.method == "esd"} == {"y2"}


def test_released_entry_must_not_carry_seed():
    with pytest.raises(ValueError, match="seed"):
        CheckpointEntry(method="esd", base_model="sd15", concept="x", source="s", origin="released", seed=1)
    with pytest.raises(ValueError, match="origin"):
        CheckpointEntry(method="esd", base_model="sd15", concept="x", source="s", origin="bogus")


def test_verify_detects_missing_and_substituted_files(tmp_path):
    entry = _entry(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    verify_checkpoint(entry, checkpoint_dir)  # ok
    # Substitution: same path, different bytes.
    (checkpoint_dir / "model.safetensors").write_bytes(b"different-checkpoint")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_checkpoint(entry, checkpoint_dir)
    # Missing file.
    (checkpoint_dir / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(entry, checkpoint_dir)


def test_prepare_artifact_local_file_never_substitutes(tmp_path, monkeypatch):
    """The fetch path, fed a local file, locks the hash and refuses mismatch."""
    calls = {}

    def fake_download_url(url, dest):
        calls["url"] = url
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"official-esd-weights")
        return url

    monkeypatch.setattr(
        "scripts.prepare_erased_checkpoints._download_url_file", fake_download_url
    )
    entry = prepare_artifact(
        "esd/sd15/van-gogh",
        checkpoint_dir=tmp_path / "checkpoints",
        lockfile=tmp_path / "checkpoints.lock.yaml",
    )
    assert entry.origin == "released"
    filename = Path(ARTIFACTS["esd/sd15/van-gogh"]["url"].split("/")[-1]).name
    assert list(entry.files) == [f"esd_sd15_van-gogh/{filename}"]
    # Lockfile now contains the entry and verifies cleanly.
    entries = load_lockfile(tmp_path / "checkpoints.lock.yaml")
    assert any(e.run_key == "esd/sd15/van gogh" for e in entries)
    verify_checkpoint(entries[0], tmp_path / "checkpoints")
    # Corrupting the file must break verification (no silent substitution).
    (tmp_path / "checkpoints" / "esd_sd15_van-gogh" / filename).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_checkpoint(entries[0], tmp_path / "checkpoints")


def test_needs_training_artifact_requires_explicit_train_flag(tmp_path):
    with pytest.raises(RuntimeError, match="--train"):
        prepare_artifact(
            "uce/sd15/van-gogh",
            checkpoint_dir=tmp_path / "checkpoints",
            lockfile=tmp_path / "checkpoints.lock.yaml",
        )


def test_manifest_only_contains_benign_concepts():
    banned = {"nsfw", "nudity", "porn", "explicit"}
    for key, artifact in ARTIFACTS.items():
        assert artifact["method"] in {"esd", "uce", "mace"}
        for word in banned:
            assert word not in artifact["concept"].lower()
            assert word not in key.lower()
