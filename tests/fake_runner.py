"""A tiny fake experiment runner used by the sweep tests.

Importable as ``tests.fake_runner:execute_run``. It keeps a module-level call
log so tests can assert that resumed sweeps make no new calls (module caching
keeps the log alive within one pytest process).
"""

from __future__ import annotations

from typing import Any, Dict

from heretic_dit.interfaces import RecoveryResult

CALLS: list[str] = []


def execute_run(run_spec: Dict[str, Any], config: Dict[str, Any]) -> RecoveryResult:
    CALLS.append(run_spec["run_id"])
    return RecoveryResult(
        concept=run_spec["concept"],
        method=run_spec["method"],
        recovery_score=0.5,
        drift_score=0.1,
        metrics={"fid": 1.0},
        cost={
            "wall_clock_sec": 1.0,
            "peak_vram_mb": 0.0,
            "trainable_params": 0,
            "sample_count": 5,
        },
    )
