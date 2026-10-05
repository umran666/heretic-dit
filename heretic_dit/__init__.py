"""
Heretic-DiT: Reproducible Subspace Abliteration and Concept Erasure
for Diffusion Transformers (DiTs) and Non-Autoregressive Models.
"""

from heretic_dit.interfaces import (
    EditSpec,
    RecoveryResult,
    ProxyEvaluation,
    SubspaceProvider,
    NoisePredictor,
    ConceptScorer,
    ConceptClassifier,
    GenerativeValidator,
    BaselineRunner,
)

__version__ = "0.1.0"

__all__ = [
    "EditSpec",
    "RecoveryResult",
    "ProxyEvaluation",
    "SubspaceProvider",
    "NoisePredictor",
    "ConceptScorer",
    "ConceptClassifier",
    "GenerativeValidator",
    "BaselineRunner",
]
