"""Config-driven, resumable experiment sweep.

Expands the grid (erasure method x concept x base model x method/baseline x
seed), delegates each run to a pluggable runner, and writes one JSON per run
into the results directory. Re-invocations skip runs whose result JSON already
exists with a matching config hash, so sweeps are resumable.

The per-run execution is pluggable because model loading and weight surgery
live with the repo-integration owner. Provide a runner via the experiment
config:

    runner: "my_pkg.runners:execute_run"

where ``execute_run(run_spec: dict, experiment_config: dict) ->
heretic_dit.interfaces.RecoveryResult``. ``run_spec`` carries the fully
expanded cell (model, erasure method, checkpoint paths from
``checkpoints.lock.yaml``, concept prompts, method, budget, seed).

Usage:
    python scripts/run_experiment.py --config configs/experiment_default.yaml --dry-run
    python scripts/run_experiment.py --config configs/experiment_default.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heretic_dit.benchmarks.concepts import load_or_create_split  # noqa: E402
from heretic_dit.interfaces import RecoveryResult  # noqa: E402

__all__ = ["load_experiment_config", "expand_runs", "run_experiment", "main"]


@dataclass
class RunSpec:
    """One fully expanded grid cell."""

    run_id: str
    base_model: str
    erasure_method: str
    concept: str
    method: str
    seed: int
    budget: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate an experiment YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("base_models", "erasure_methods", "concepts", "methods", "seeds"):
        if key not in config:
            raise ValueError(f"Experiment config is missing required key {key!r}.")
    if not config["base_models"] or not config["erasure_methods"] or not config["seeds"]:
        raise ValueError("base_models, erasure_methods, and seeds must be non-empty.")
    methods = config["methods"]
    if isinstance(methods, list) and methods and isinstance(methods[0], str):
        methods = [{"name": m} for m in methods]
        config["methods"] = methods
    for entry in methods:
        if "name" not in entry:
            raise ValueError("Each methods entry needs a 'name'.")
    return config


def _resolve_concepts(config: Dict[str, Any], concepts_config: Path, split_dir: Path) -> List[str]:
    requested = config["concepts"]
    split = load_or_create_split(config_path=concepts_config, split_dir=split_dir)
    if requested == "all":
        return list(split.registry.names)
    if isinstance(requested, str):
        requested = [requested]
    resolved: List[str] = []
    for name in requested:
        if name not in split.registry:
            raise KeyError(f"Concept {name!r} is not in the registry {list(split.registry.names)}.")
        resolved.append(name)
    return resolved


def _expand(value: Any) -> List[Any]:
    """Expand a scalar or a list of scalars (sweep values) into a list."""
    if isinstance(value, list):
        return value
    return [value]


def expand_runs(
    config: Dict[str, Any],
    concepts: List[str],
    config_hash: str,
) -> List[RunSpec]:
    """Expand the experiment grid into resumable run specs."""
    specs: List[RunSpec] = []
    for base_model in config["base_models"]:
        for erasure_method in config["erasure_methods"]:
            for concept in concepts:
                for method_entry in config["methods"]:
                    method_name = method_entry["name"]
                    budget_template = method_entry.get("budget", {}) or {}
                    sweep_keys = sorted(k for k, v in budget_template.items() if isinstance(v, list))
                    combos: List[Dict[str, Any]] = [{}]
                    for key in sweep_keys:
                        combos = [dict(c, **{key: value}) for c in combos for value in budget_template[key]]
                    for combo in combos:
                        budget = {k: v for k, v in budget_template.items() if k not in sweep_keys}
                        budget.update(combo)
                        for seed in config["seeds"]:
                            identity = {
                                "config_hash": config_hash,
                                "base_model": base_model,
                                "erasure_method": erasure_method,
                                "concept": concept,
                                "method": method_name,
                                "budget": budget,
                                "seed": int(seed),
                            }
                            blob = json.dumps(identity, sort_keys=True, separators=(",", ":"))
                            run_id = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
                            specs.append(
                                RunSpec(
                                    run_id=run_id,
                                    base_model=base_model,
                                    erasure_method=erasure_method,
                                    concept=concept,
                                    method=method_name,
                                    seed=int(seed),
                                    budget=budget,
                                )
                            )
    return specs


def _import_runner(runner_path: str) -> Callable[[RunSpec, Dict[str, Any]], RecoveryResult]:
    module_name, _, function_name = runner_path.partition(":")
    if not function_name:
        raise ValueError(f"runner must look like 'module:function'; got {runner_path!r}.")
    module = importlib.import_module(module_name)
    runner = getattr(module, function_name)
    return runner


def _result_matches_config(path: Path, config_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return False
    return payload.get("config_hash") == config_hash


def run_experiment(
    config_path: str | Path,
    results_dir: Optional[str | Path] = None,
    dry_run: bool = False,
    concepts_config: str | Path = "configs/concepts_default.yaml",
    split_dir: str | Path = "splits",
    runner_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Run (or plan) the sweep; returns a summary dict."""
    config_path = Path(config_path)
    config = load_experiment_config(config_path)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    results_dir = Path(results_dir or config.get("results_dir", "results"))
    concepts = _resolve_concepts(config, Path(concepts_config), Path(split_dir))
    specs = expand_runs(config, concepts, config_hash)

    if dry_run:
        for spec in specs:
            print(f"PLAN {spec.run_id} {spec.base_model}/{spec.erasure_method}/{spec.concept} "
                  f"{spec.method} seed={spec.seed} budget={spec.budget}")
        return {"planned": len(specs), "results_dir": str(results_dir)}

    runner_path = runner_override or config.get("runner")
    if not runner_path:
        raise ValueError(
            "No runner configured. Add 'runner: \"module:function\"' to the experiment "
            "config (a callable mapping (run_spec, config) -> RecoveryResult), which the "
            "repo-integration owner wires to the real pipelines, or use --dry-run."
        )
    runner = _import_runner(runner_path)

    completed = skipped = failed = 0
    failures: List[str] = []
    for spec in specs:
        result_path = results_dir / spec.run_id / "result.json"
        if _result_matches_config(result_path, config_hash):
            skipped += 1
            continue
        try:
            result = runner(spec.to_dict(), dict(config))
            if not isinstance(result, RecoveryResult):
                raise TypeError(
                    f"runner returned {type(result)!r}; expected heretic_dit RecoveryResult."
                )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"config_hash": config_hash, "run_spec": spec.to_dict(), "result": {
                "concept": result.concept,
                "method": result.method,
                "recovery_score": result.recovery_score,
                "drift_score": result.drift_score,
                "metrics": result.metrics,
                "cost": result.cost,
            }}
            with result_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            completed += 1
        except Exception as error:  # noqa: BLE001 - continue the sweep
            failed += 1
            failures.append(f"{spec.run_id}: {error}")
            print(f"FAIL {spec.run_id}: {error}", file=sys.stderr)
    return {
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
        "results_dir": str(results_dir),
        "total": len(specs),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--concepts-config", default="configs/concepts_default.yaml")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--runner", default=None, help="override: 'module:function'")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = run_experiment(
        args.config,
        results_dir=args.results_dir,
        dry_run=args.dry_run,
        concepts_config=args.concepts_config,
        split_dir=args.split_dir,
        runner_override=args.runner,
    )
    print(json.dumps(summary, indent=2))
    return 1 if summary.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
