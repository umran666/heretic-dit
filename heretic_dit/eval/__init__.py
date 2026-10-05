"""Ground-truth recovery metrics, quality metrics, and proxy validity checks."""

from heretic_dit.eval.adapters import (
    CallableAdapter,
    DiffusersAdapter,
    GenerationAdapter,
    deterministic_image_seeds,
)
from heretic_dit.eval.classifiers import (
    ClipStyleScorer,
    ClipZeroShotClassifier,
    ImagenetteClassifier,
)
from heretic_dit.eval.quality import (
    FeatureExtractor,
    InceptionV3Features,
    QualityEvaluator,
    RandomProjectionFeatures,
    clip_score,
    fid_score,
    frechet_distance,
    lpips_paired,
)
from heretic_dit.eval.proxy_validity import (
    ProxyTrial,
    ProxyValidityReport,
    evaluate_proxy_validity,
    load_trials,
    save_trials,
    spearman,
)
from heretic_dit.eval.validator import (
    DeterministicGenerativeValidator,
    ValidatorConfig,
    generate_paired_images,
    recovery_gap,
)

__all__ = [
    "CallableAdapter",
    "DiffusersAdapter",
    "GenerationAdapter",
    "deterministic_image_seeds",
    "ClipStyleScorer",
    "ClipZeroShotClassifier",
    "ImagenetteClassifier",
    "FeatureExtractor",
    "InceptionV3Features",
    "QualityEvaluator",
    "RandomProjectionFeatures",
    "clip_score",
    "fid_score",
    "frechet_distance",
    "lpips_paired",
    "ProxyTrial",
    "ProxyValidityReport",
    "evaluate_proxy_validity",
    "load_trials",
    "save_trials",
    "spearman",
    "DeterministicGenerativeValidator",
    "ValidatorConfig",
    "generate_paired_images",
    "recovery_gap",
]
