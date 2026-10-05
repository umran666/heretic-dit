"""Recovery baselines: what the Heretic-DiT edit must beat.

Every baseline implements ``heretic_dit.interfaces.BaselineRunner``:

    run(self, erased_model, concept, budget) -> RecoveryResult

and reports the same metrics as the main method plus a standardized ``cost``
block (``wall_clock_sec``, ``peak_vram_mb``, ``trainable_params``,
``sample_count``), because the paper's claim is "comparable recovery at far
lower cost."

Baselines are deliberately backend-agnostic: the actual fine-tuning happens
inside a pluggable ``RecoveryBackend`` (see ``training_backend.py``), so tests
run with tiny stand-ins and real diffusers/peft implementations are optional.
"""

from heretic_dit.baselines.finetune_recovery import (
    DummyFinetuneBackend,
    FinetuneRecovery,
)
from heretic_dit.baselines.null_baselines import (
    NoEditBaseline,
    RandomProjectionBaseline,
    UnerasedReferenceBaseline,
    make_null_baseline,
)
from heretic_dit.baselines.textual_inversion import (
    DummyTextualInversionBackend,
    TextualInversionRecovery,
)
from heretic_dit.baselines.training_backend import TrainingOutcome

__all__ = [
    "FinetuneRecovery",
    "DummyFinetuneBackend",
    "NoEditBaseline",
    "RandomProjectionBaseline",
    "UnerasedReferenceBaseline",
    "make_null_baseline",
    "TextualInversionRecovery",
    "DummyTextualInversionBackend",
    "TrainingOutcome",
]
