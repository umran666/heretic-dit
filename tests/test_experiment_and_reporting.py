"""Tests for the resumable experiment sweep and reporting aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from heretic_dit.reporting.aggregate import group_mean, load_results, main_table_rows, pareto_front, per_erasure_breakdown
from heretic_dit.reporting.tables import breakdown_table_latex, main_table_latex, main_table_markdown
from scripts.run_experiment import expand_runs, load_experiment_config, run_experiment


@pytest.fixture
def experiment_config(tmp_path):
    config = {
        "base_models": ["sd15"],
        "erasure_methods": ["esd", "uce"],
        "concepts": ["church", "van gogh"],
        "methods": [
            {"name": "heretic-dit", "budget": {"alpha": [0.5, 1.0]}},
            {"name": "no-edit"},
            {"name": "textual-inversion", "budget": {"steps": [50, 100]}},
        ],
        "seeds": [0, 1],
        "results_dir": str(tmp_path / "results"),
        "runner": "tests.fake_runner:execute_run",
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_expand_runs_grid_size_and_deduplication():
    config = {
        "base_models": ["m1"],
        "erasure_methods": ["esd"],
        "concepts": ["c1", "c2"],
        "methods": [
            {"name": "a", "budget": {"alpha": [0.5, 1.0]}},
            {"name": "b"},
        ],
        "seeds": [0],
    }
    specs = expand_runs(config, ["c1", "c2"], config_hash="h")
    # 2 concepts x (2 alpha values + 1 method with no sweep) = 6 runs
    assert len(specs) == 6
    run_ids = {s.run_id for s in specs}
    assert len(run_ids) == 6  # every spec is distinct
    assert all(s.base_model == "m1" and s.erasure_method == "esd" for s in specs)


def test_experiment_config_validation(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"base_models": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        load_experiment_config(path)


def test_sweep_executes_every_cell_and_is_resumable(tmp_path, experiment_config):
    summary = run_experiment(
        experiment_config,
        results_dir=tmp_path / "results",
        concepts_config="configs/concepts_default.yaml",
        split_dir=tmp_path / "splits",
    )
    assert summary["failed"] == 0
    assert summary["completed"] == summary["total"]  # 2 erasure x 2 concepts x (2+1+2) x 2 seeds = 40
    results_dir = tmp_path / "results"
    payloads = list(results_dir.rglob("result.json"))
    assert len(payloads) == summary["total"]

    # Re-running skips everything (resumability).
    summary_again = run_experiment(
        experiment_config,
        results_dir=tmp_path / "results",
        concepts_config="configs/concepts_default.yaml",
        split_dir=tmp_path / "splits",
    )
    assert summary_again["completed"] == 0
    assert summary_again["skipped"] == summary_again["total"]


def test_dry_run_writes_no_results(tmp_path, experiment_config):
    summary = run_experiment(
        experiment_config,
        results_dir=tmp_path / "results",
        dry_run=True,
        concepts_config="configs/concepts_default.yaml",
        split_dir=tmp_path / "splits",
    )
    assert summary["planned"] == 40
    assert not (tmp_path / "results").exists() or not list((tmp_path / "results").rglob("result.json"))


def test_sweep_without_runner_errors(tmp_path, experiment_config):
    config = yaml.safe_load(experiment_config.read_text(encoding="utf-8"))
    config.pop("runner")
    path = tmp_path / "norunner.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="runner"):
        run_experiment(path, results_dir=tmp_path / "results",
                       concepts_config="configs/concepts_default.yaml",
                       split_dir=tmp_path / "splits")


# ---------------------------------------------------------------------------
# Aggregation and tables
# ---------------------------------------------------------------------------


def _write_result(results_dir: Path, rid, erasure, method, recovery, drift, fid=None, cost=None):
    payload = {
        "config_hash": "abc",
        "run_spec": {
            "run_id": rid,
            "base_model": "sd15",
            "erasure_method": erasure,
            "concept": "church",
            "method": method,
            "seed": 0,
            "budget": {},
        },
        "result": {
            "concept": "church",
            "method": method,
            "recovery_score": recovery,
            "drift_score": drift,
            "metrics": ({"fid": fid} if fid is not None else {}),
            "cost": cost or {"wall_clock_sec": 1.0, "peak_vram_mb": 0.0, "trainable_params": 0, "sample_count": 20},
        },
    }
    (results_dir / rid).mkdir(parents=True, exist_ok=True)
    (results_dir / rid / "result.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    _write_result(d, "r1", "esd", "heretic-dit", 0.8, 0.1, fid=5.0)
    _write_result(d, "r2", "esd", "heretic-dit", 0.9, 0.05, fid=8.0)
    _write_result(d, "r3", "esd", "no-edit", 0.1, 0.0, fid=1.0)
    _write_result(d, "r4", "uce", "heretic-dit", 0.6, 0.2, fid=6.0)
    _write_result(d, "r5", "uce", "no-edit", 0.2, 0.0, fid=1.0)
    _write_result(d, "r6", "esd", "textual-inversion", 0.5, 0.02, fid=3.0,
                  cost={"wall_clock_sec": 100.0, "peak_vram_mb": 9000.0, "trainable_params": 768, "sample_count": 40})
    return d


def test_load_results_parses_all_runs(results_dir):
    records = load_results(results_dir)
    assert len(records) == 6
    ti = next(r for r in records if r.method == "textual-inversion")
    assert ti.cost["trainable_params"] == 768
    assert ti.metrics["fid"] == 3.0


def test_main_table_rows_group_and_average(results_dir):
    rows = main_table_rows(load_results(results_dir))
    esd_heretic = next(r for r in rows if r["erasure_method"] == "esd" and r["method"] == "heretic-dit")
    assert esd_heretic["recovery"] == pytest.approx(0.85)
    assert esd_heretic["fid"] == pytest.approx(6.5)
    ti_row = next(r for r in rows if r["method"] == "textual-inversion")
    assert ti_row["trainable_params"] == 768


def test_per_erasure_breakdown_never_averages_away_methods(results_dir):
    breakdown = per_erasure_breakdown(load_results(results_dir))
    assert set(breakdown) == {"esd", "uce"}
    assert breakdown["esd"]["heretic-dit"]["recovery"] == pytest.approx(0.85)
    assert breakdown["uce"]["heretic-dit"]["recovery"] == pytest.approx(0.6)
    # different erasure methods, different numbers -- kept apart
    assert breakdown["esd"]["heretic-dit"]["recovery"] != breakdown["uce"]["heretic-dit"]["recovery"]


def test_pareto_front_dominance():
    points = [(0.1, 1.0), (0.5, 1.0), (0.5, 0.5), (0.9, 0.5), (0.9, 0.9)]
    # (0.9, 0.5) dominates everything; (0.9, 0.9) is dominated by it.
    assert pareto_front(points) == [3]
    assert pareto_front([]) == []
    with pytest.raises(ValueError):
        pareto_front([(-1.0, 0.0)])


def test_group_mean_helper(results_dir):
    means = group_mean(load_results(results_dir), "method")
    assert means["heretic-dit"] == pytest.approx((0.8 + 0.9 + 0.6) / 3)
    assert means["no-edit"] == pytest.approx(0.15)


def test_latex_tables_and_figures(tmp_path, results_dir):
    records = load_results(results_dir)
    tex = main_table_latex(records)
    assert "Heretic-DiT (ours)" in tex
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert "esd" in tex and "uce" in tex  # per-erasure sections present
    md = main_table_markdown(records)
    assert "No edit (lower bound)" in md  # methods are prettified in tables
    breakdown_tex = breakdown_table_latex(records)
    assert "textual" in breakdown_tex or "Textual" in breakdown_tex
    # escaped underscores in method names
    assert r"\_" in breakdown_tex or "_" not in "".join(breakdown_tex.split())
    assert breakdown_table_latex(records, path=tmp_path / "breakdown.tex")
    assert (tmp_path / "breakdown.tex").exists()
