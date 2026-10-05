"""Pluggable concept classifiers behind the ``ConceptClassifier`` protocol.

All classifiers return per-image confidence scores in ``[0, 1]`` for the
target concept, matching ``heretic_dit.interfaces.ConceptClassifier``:

    classify(self, images, concept: str) -> Sequence[float]

Heavy dependencies (torchvision, open_clip) are imported lazily inside
methods so the core test suite runs without them; tests use fake classifiers
that satisfy the same protocol.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "IMAGENETTE_WNIDS",
    "ImagenetteClassifier",
    "ClipZeroShotClassifier",
    "ClipStyleScorer",
]

#: The 10 Imagenette classes with their ImageNet-1k WordNet IDs.
IMAGENETTE_WNIDS: Dict[str, str] = {
    "tench": "n01440764",
    "english springer": "n02102040",
    "cassette player": "n02950826",
    "chain saw": "n02974003",
    "church": "n03000684",
    "french horn": "n03075370",
    "garbage truck": "n03417042",
    "gas pump": "n03425413",
    "golf ball": "n03445777",
    "parachute": "n03888257",
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _as_image_batch(images: Tensor | Sequence[Any]) -> Tensor:
    """Coerce classifier input into a float ``(N, C, H, W)`` batch in [0, 1]."""
    if isinstance(images, Tensor):
        batch = images
    else:
        stacked = [torch.as_tensor(img) for img in images]
        if not stacked:
            return torch.zeros((0, 3, 1, 1))
        batch = torch.stack([img.float() if img.is_floating_point() else img.float() for img in stacked])
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    if batch.ndim != 4:
        raise ValueError(f"Classifier images must be (N, C, H, W); got {tuple(batch.shape)}.")
    return batch.to(torch.float32)


def _normalize(batch: Tensor, mean: Sequence[float], std: Sequence[float]) -> Tensor:
    mean_t = torch.tensor(mean, dtype=batch.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=batch.dtype).view(1, -1, 1, 1)
    return (batch - mean_t) / std_t


def _resize(batch: Tensor, size: int) -> Tensor:
    if batch.shape[-1] == size and batch.shape[-2] == size:
        return batch
    return F.interpolate(batch, size=(size, size), mode="bilinear", align_corners=False)


class ImagenetteClassifier:
    """ImageNet-pretrained torchvision classifier reduced to Imagenette classes.

    Confidence for an object concept is the softmax mass over the ten
    Imagenette classes assigned to the concept's WordNet ID. Requires
    ``torchvision`` (installed lazily on first use).
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        weights: str = "IMAGENET1K_V2",
        wnid_to_index: Optional[Mapping[str, int]] = None,
    ) -> None:
        self._model_name = model_name
        self._weights = weights
        self._wnid_to_index = dict(wnid_to_index) if wnid_to_index is not None else None
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from torchvision import models as tv_models
            except ImportError as error:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "ImagenetteClassifier requires torchvision. "
                    "Install it with: pip install torchvision"
                ) from error
            constructor = getattr(tv_models, self._model_name)
            loaded = constructor(weights=self._weights)
            loaded.eval()
            self._model = loaded
        return self._model

    def _index_for_wnid(self, wnid: str) -> int:
        if self._wnid_to_index is not None:
            if wnid not in self._wnid_to_index:
                raise KeyError(f"No ImageNet index mapping for wnid {wnid!r}.")
            return int(self._wnid_to_index[wnid])
        # Fall back to matching torchvision's category names (e.g.
        # "tench, Tinca tinca") by their leading class term.
        model = self._model
        categories = model.categories if hasattr(model, "categories") else []
        name = next(n for n, w in IMAGENETTE_WNIDS.items() if w == wnid)
        for index, category in enumerate(categories):
            first_term = str(category).split(",")[0].strip().lower()
            if first_term == name:
                return index
        raise KeyError(
            f"Could not map Imagenette class {name!r} (wnid {wnid}) onto "
            "torchvision's category list; pass an explicit wnid_to_index mapping."
        )

    def classify(self, images: Tensor | Sequence[Any], concept: str) -> Sequence[float]:
        wnid = IMAGENETTE_WNIDS.get(concept.lower())
        if wnid is None:
            raise KeyError(
                f"Concept {concept!r} is not one of the ten Imagenette classes; "
                "use a CLIP-based classifier for style/celebrity concepts."
            )
        model = self._ensure_model()
        index = self._index_for_wnid(wnid)
        batch = _resize(_normalize(_as_image_batch(images), _IMAGENET_MEAN, _IMAGENET_STD), 224)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)
        return [float(row[index]) for row in probs]


def _clip_softmax_confidence(
    image_features: Tensor,
    text_features: Tensor,
    target_index: int,
    temperature: float = 100.0,
) -> list[float]:
    logits = temperature * image_features @ text_features.T
    probs = F.softmax(logits, dim=-1)
    return [float(row[target_index]) for row in probs]


class ClipZeroShotClassifier:
    """CLIP zero-shot classifier over per-concept prompt embeddings.

    Confidence for a concept is the softmax probability (over all registered
    concepts' averaged prompt embeddings) that the image matches the concept.
    Requires ``open_clip_torch``.
    """

    def __init__(
        self,
        concepts: Sequence[str],
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        templates: Sequence[str] = (
            "a photo of a {}",
            "a photograph of a {}",
            "a rendering of a {}",
            "a close-up photo of a {}",
        ),
        device: Optional[str] = None,
    ) -> None:
        if not concepts:
            raise ValueError("ClipZeroShotClassifier needs at least one concept.")
        self._concepts = [c.lower() for c in concepts]
        self._model_name = model_name
        self._pretrained = pretrained
        self._templates = list(templates)
        self._device = device
        self._text_features: Optional[Tensor] = None

    def _ensure_features(self) -> tuple[Any, Tensor]:
        if self._text_features is None:
            try:
                import open_clip
            except ImportError as error:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "ClipZeroShotClassifier requires open_clip_torch. "
                    "Install it with: pip install open_clip_torch"
                ) from error
            model, _, _ = open_clip.create_model_and_transforms(
                self._model_name, pretrained=self._pretrained
            )
            tokenizer = open_clip.get_tokenizer(self._model_name)
            model.eval()
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            with torch.no_grad():
                embeddings = []
                for concept in self._concepts:
                    tokens = tokenizer([t.format(concept) for t in self._templates]).to(device)
                    text = model.encode_text(tokens)
                    embeddings.append(F.normalize(text, dim=-1).mean(dim=0))
                text_features = F.normalize(torch.stack(embeddings), dim=-1)
            self._text_features = text_features
            self._clip_model = model
            self._device_used = device
        return self._clip_model, self._text_features

    def classify(self, images: Tensor | Sequence[Any], concept: str) -> Sequence[float]:
        key = concept.lower()
        if key not in self._concepts:
            raise KeyError(
                f"Concept {concept!r} was not registered; registered: {self._concepts}."
            )
        model, text_features = self._ensure_features()
        try:
            import open_clip
        except ImportError:  # pragma: no cover - already imported above
            raise
        _, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        batch = _as_image_batch(images)
        # Apply CLIP normalization channel-wise; resize via F.interpolate to
        # avoid torchvision dependency.
        resized = _resize(batch, 224)
        normalized = _normalize(resized, _CLIP_MEAN, _CLIP_STD).to(self._device_used)
        with torch.no_grad():
            image_features = model.encode_image(normalized)
            image_features = F.normalize(image_features.float(), dim=-1)
        return _clip_softmax_confidence(
            image_features, text_features.to(image_features.dtype), self._concepts.index(key)
        )


class ClipStyleScorer:
    """CLIP-based style presence scorer.

    Scores how strongly images match style promptings of the concept against
    neutral photographic counter-prompts; confidence is the softmax mass of
    the style side. Requires ``open_clip_torch``.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        style_templates: Sequence[str] = (
            "a painting in the style of {}",
            "an artwork in the style of {}",
            "a picture that looks like it was painted by {}",
        ),
        counter_templates: Sequence[str] = (
            "a plain photograph",
            "a regular photo taken with a camera",
        ),
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._style_templates = list(style_templates)
        self._counter_templates = list(counter_templates)
        self._device = device
        self._cache: Dict[str, Tensor] = {}

    def _features_for(self, model: Any, tokenizer: Any, concept: str, templates: Sequence[str]) -> Tensor:
        key = "|".join(templates)
        if key not in self._cache:
            tokens = tokenizer([t.format(concept) for t in templates]).to(self._device_used)
            with torch.no_grad():
                text = model.encode_text(tokens)
            self._cache[key] = F.normalize(text, dim=-1).mean(dim=0)
        return self._cache[key]

    def classify(self, images: Tensor | Sequence[Any], concept: str) -> Sequence[float]:
        try:
            import open_clip
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ImportError(
                "ClipStyleScorer requires open_clip_torch. "
                "Install it with: pip install open_clip_torch"
            ) from error
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        tokenizer = open_clip.get_tokenizer(self._model_name)
        model.eval()
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        self._device_used = device

        style = self._features_for(model, tokenizer, concept.lower(), self._style_templates)
        counter = self._features_for(model, tokenizer, "a photograph", self._counter_templates)
        text_features = F.normalize(torch.stack([style, counter]), dim=-1)

        batch = _resize(_normalize(_as_image_batch(images), _CLIP_MEAN, _CLIP_STD), 224).to(device)
        with torch.no_grad():
            image_features = model.encode_image(batch)
            image_features = F.normalize(image_features.float(), dim=-1)
        return _clip_softmax_confidence(image_features, text_features, target_index=0)
