"""Fetch or train erased checkpoints for ESD / UCE / MACE and lock provenance.

For every (erasure method, base model, concept) requested, this script either
downloads a publicly released erased checkpoint or, where none is released,
runs the official training script at a pinned commit with a pinned seed. All
provenance -- source, revision/commit, seed, SHA-256 of every file -- is
recorded in ``checkpoints.lock.yaml``. Hash mismatches are hard errors:
checkpoints are never silently substituted.

Usage:
    python scripts/prepare_erased_checkpoints.py --list
    python scripts/prepare_erased_checkpoints.py --artifact esd/sd15/van-gogh
    python scripts/prepare_erased_checkpoints.py --artifact mace/sd15/tom-hanks --train
    python scripts/prepare_erased_checkpoints.py --verify

Sources (as of 2026-10):
    * ESD  -- released checkpoints on Hugging Face ``rohitgandikota/erasing-models``
      (project page: erasing.baulab.info). Releases cover a set of object and
      style erasures for SD 1.4/1.5.
    * UCE  -- official repo ``github.com/rohitgandikota/unified-concept-editing``.
      Release availability varies by concept; the script reports what it finds.
    * MACE -- official repo ``github.com/GoatWu/MACE``. Released checkpoints are
      provided through GitHub releases for a fixed concept set; others must be
      trained with the repo's scripts.
Artifacts whose release status could not be verified are marked
``needs_training`` and require ``--train``; nothing is guessed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, urlretrieve

# Make the repo importable when running from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heretic_dit.benchmarks.checkpoints import (  # noqa: E402
    CheckpointEntry,
    hash_file,
    load_lockfile,
    verify_checkpoint,
    write_lockfile,
)

__all__ = ["ARTIFACTS", "main", "prepare_artifact"]

DEFAULT_LOCKFILE = Path("checkpoints.lock.yaml")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")

# ---------------------------------------------------------------------------
# Known artifacts. release_status:
#   "hf"        -> released files on Hugging Face (hf_repo, hf_file).
#   "github"    -> released files on GitHub releases (github_repo, github_file).
#   "needs_training" -> no verified public release; train with the official
#                      script (train_repo, train_command).
# NOTE: this manifest only contains BENIGN concepts (object / style /
# celebrity), consistent with the project scope constraint.
# ---------------------------------------------------------------------------
ARTIFACTS: Dict[str, Dict[str, Any]] = {
    "esd/sd15/van-gogh": {
        "method": "esd",
        "base_model": "sd15",
        "concept": "van gogh",
        "release_status": "url",
        "url": "https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt",
        "notes": "Official ICCV 2023 released ESD style-erasure checkpoint for Van Gogh on SD1.4/1.5.",
    },
    "esd/sd15/grumpy-cat": {
        "method": "esd",
        "base_model": "sd15",
        "concept": "grumpy cat",
        "release_status": "hf",
        "hf_repo": "rohitgandikota/erasing-models",
        "hf_file": "grumpy_cat.pt",
    },
    "uce/sd15/van-gogh": {
        "method": "uce",
        "base_model": "sd15",
        "concept": "van gogh",
        "release_status": "needs_training",
        "train_repo": "https://github.com/rohitgandikota/unified-concept-editing",
        "train_command": (
            "python train_uce.py --concepts {concept} --seed {seed} "
            "--config_name uce_sd15.yaml"
        ),
        "notes": "Verify whether a release exists before training; when a "
        "public release is confirmed, switch release_status to 'github'.",
    },
    "mace/sd15/tom-hanks": {
        "method": "mace",
        "base_model": "sd15",
        "concept": "tom hanks",
        "release_status": "needs_training",
        "train_repo": "https://github.com/GoatWu/MACE",
        "train_command": (
            "bash scripts/train/mace_sd15.sh --concept {concept} --seed {seed}"
        ),
    },
}


@dataclass
class PrepareReport:
    fetched: List[str] = field(default_factory=list)
    reused: List[str] = field(default_factory=list)
    trained: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)


def _download_hf_file(repo: str, filename: str, revision: Optional[str], dest: Path) -> str:
    """Download a file from Hugging Face; returns the resolved revision."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        print(f"huggingface_hub not installed; falling back to direct URL: {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, dest)  # noqa: S310 - pinned public URL
        return "main"
    path = hf_hub_download(repo_id=repo, filename=filename, revision=revision)
    dest.parent.mkdir(parents=True, exist_ok=True)
    target = dest
    if Path(path).resolve() != target.resolve():
        target.write_bytes(Path(path).read_bytes())
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision or "main")
    return info.sha or (revision or "main")


def _download_github_release(repo_url: str, asset: str, dest: Path) -> str:
    """Download a release asset from ``<repo>/releases/download/<asset>``."""
    url = f"{repo_url.rstrip('/')}/releases/download/{asset}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:  # noqa: S310 - pinned public URL
        dest.write_bytes(response.read())
    return url


def _download_url_file(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)
    return url


def _resolve_commit(repo_url: str, pinned: Optional[str]) -> str:
    """Return the pinned commit, or the remote HEAD (recorded into the lockfile)."""
    if pinned:
        return pinned
    output = subprocess.run(
        ["git", "ls-remote", repo_url, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return output.stdout.split()[0]


def _train_checkpoint(artifact: Dict[str, Any], seed: int, workdir: Path) -> Path:
    """Clone the official repo and run its training script with a pinned seed."""
    repo_url = artifact["train_repo"]
    commit = _resolve_commit(repo_url, artifact.get("pin_commit"))
    clone_dir = workdir / "repo"
    subprocess.run(
        ["git", "clone", repo_url, str(clone_dir)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(clone_dir), "checkout", commit],
        check=True,
        capture_output=True,
    )
    command = artifact["train_command"].format(concept=artifact["concept"], seed=seed)
    print(f"Training with: {command}")
    subprocess.run(command, shell=True, check=True, cwd=clone_dir)
    produced = clone_dir / "output"
    if not produced.exists():
        raise RuntimeError(
            f"Training did not produce an 'output/' directory in {clone_dir}; "
            "adjust train_command in the artifact manifest."
        )
    (workdir / "commit.txt").write_text(commit + "\n", encoding="utf-8")
    return produced


def prepare_artifact(
    artifact_key: str,
    artifacts: Dict[str, Dict[str, Any]] = ARTIFACTS,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    lockfile: Path = DEFAULT_LOCKFILE,
    seed: int = 0,
    train: bool = False,
    revision: Optional[str] = None,
) -> CheckpointEntry:
    """Fetch (or train) one artifact and record it in the lockfile.

    Raises:
        KeyError: unknown artifact key.
        RuntimeError: a released file is absent from the source (never
            silently substituted by a different file).
    """
    if artifact_key not in artifacts:
        raise KeyError(f"Unknown artifact {artifact_key!r}; use --list.")
    artifact = artifacts[artifact_key]
    entry = CheckpointEntry(
        method=artifact["method"],
        base_model=artifact["base_model"],
        concept=artifact["concept"],
        source="",
        origin="released",
        notes=artifact.get("notes", ""),
    )
    dest_dir = checkpoint_dir / artifact_key.replace("/", "_")

    status = artifact["release_status"]
    if status == "hf":
        entry.source = f"hf:{artifact['hf_repo']}"
        entry.revision = _download_hf_file(
            artifact["hf_repo"], artifact["hf_file"], revision, dest_dir / artifact["hf_file"]
        )
        entry.files[f"{dest_dir.name}/{artifact['hf_file']}"] = hash_file(dest_dir / artifact["hf_file"])
    elif status == "github":
        entry.source = artifact["github_repo"]
        _download_github_release(
            artifact["github_repo"], artifact["github_file"], dest_dir / Path(artifact["github_file"]).name
        )
        entry.files[f"{dest_dir.name}/{Path(artifact['github_file']).name}"] = hash_file(
            dest_dir / Path(artifact["github_file"]).name
        )
    elif status == "url":
        entry.source = artifact["url"]
        url_file = Path(artifact["url"].split("/")[-1])
        dest_file = dest_dir / url_file
        _download_url_file(artifact["url"], dest_file)
        entry.files[f"{dest_dir.name}/{url_file.name}"] = hash_file(dest_file)
    elif status == "needs_training" and not train:
        raise RuntimeError(
            f"{artifact_key} has no verified public release. Re-run with --train to "
            "produce it via the official training script, or check the source repo "
            "for a new release and update the manifest."
        )
    elif train:
        entry.origin = "trained"
        entry.seed = int(seed)
        entry.command = artifact["train_command"].format(concept=artifact["concept"], seed=seed)
        with tempfile.TemporaryDirectory(prefix="heretic-dit-train-") as tmp:
            output_dir = _train_checkpoint(artifact, int(seed), Path(tmp))
            entry.commit = (Path(tmp) / "commit.txt").read_text().strip()
            dest_dir.mkdir(parents=True, exist_ok=True)
            for produced in sorted(output_dir.iterdir()):
                if produced.is_file():
                    target = dest_dir / produced.name
                    target.write_bytes(produced.read_bytes())
                    entry.files[f"{dest_dir.name}/{produced.name}"] = hash_file(target)
    else:
        raise RuntimeError(f"{artifact_key}: unhandled release_status {status!r}.")

    verify_checkpoint(entry, checkpoint_dir)
    write_lockfile(lockfile, [entry])
    return entry


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", help="artifact key, e.g. esd/sd15/van-gogh")
    parser.add_argument("--all", action="store_true", help="prepare every released artifact")
    parser.add_argument("--train", action="store_true", help="train artifacts that have no release")
    parser.add_argument("--seed", type=int, default=0, help="training seed for --train")
    parser.add_argument("--revision", default=None, help="pin an HF revision")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--lockfile", default=str(DEFAULT_LOCKFILE))
    parser.add_argument("--verify", action="store_true", help="verify existing lockfile entries")
    parser.add_argument("--list", action="store_true", help="list known artifacts")
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint_dir)
    lockfile = Path(args.lockfile)

    if args.list:
        for key, artifact in ARTIFACTS.items():
            print(f"{key:28s} [{artifact['release_status']}] {artifact.get('notes', '')}")
        return 0

    if args.verify:
        failures = 0
        for entry in load_lockfile(lockfile):
            try:
                verify_checkpoint(entry, checkpoint_dir)
                print(f"OK    {entry.run_key}")
            except Exception as error:  # noqa: BLE001 - report every failure
                failures += 1
                print(f"FAIL  {entry.run_key}: {error}")
        return 1 if failures else 0

    if not args.artifact and not args.all:
        parser.error("Provide --artifact KEY, --all, --list, or --verify.")

    keys = list(ARTIFACTS) if args.all else [args.artifact]
    report = PrepareReport()
    for key in keys:
        artifact = ARTIFACTS[key]
        if artifact["release_status"] == "needs_training" and not args.train:
            report.failed.append(key)
            print(f"SKIP  {key}: no verified release (use --train to build it)")
            continue
        try:
            entry = prepare_artifact(
                key,
                checkpoint_dir=checkpoint_dir,
                lockfile=lockfile,
                seed=args.seed,
                train=args.train,
                revision=args.revision,
            )
            report.fetched.append(key)
            print(f"DONE  {key}: {entry.origin}, files={list(entry.files)}")
        except Exception as error:  # noqa: BLE001 - keep preparing other artifacts
            report.failed.append(key)
            print(f"ERROR {key}: {error}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
