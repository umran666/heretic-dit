"""Benign concept registry and deterministic search / held-out prompt splits.

Scope constraint: only benign proxy concepts are supported -- Imagenette
object classes, artist styles, and celebrity-erasure targets of the kind used
in the ESD and MACE papers. Adversarial-prompt attack baselines and NSFW
concept material are intentionally out of scope and rejected here.

The concept list is injectable via a YAML config (see
``configs/concepts_default.yaml``); list sizes are configurable per category.
The search / held-out prompt split is seeded, deterministic, and persisted to
disk so that every run reuses the identical split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import yaml

__all__ = [
    "BENIGN_CATEGORIES",
    "ConceptSpec",
    "ConceptRegistry",
    "PromptSplit",
    "build_concept_registry",
    "load_or_create_split",
    "DEFAULT_CONCEPTS_CONFIG",
]

#: The only categories permitted in a registry. Anything else is rejected.
BENIGN_CATEGORIES: Tuple[str, ...] = ("object", "style", "celebrity")

ConceptCategory = Literal["object", "style", "celebrity"]

DEFAULT_CONCEPTS_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "concepts_default.yaml"

_SPLIT_MANIFEST_NAME = "split_manifest.json"


def _slug(name: str) -> str:
    """Return a filesystem-safe slug for a concept name."""
    return "-".join(name.lower().strip().split())


@dataclass(frozen=True)
class ConceptSpec:
    """A single benign concept targeted by an erasure audit.

    Attributes:
        name: Human-readable concept name (e.g. ``"chain saw"``).
        category: One of ``object`` / ``style`` / ``celebrity``.
        search_prompts: Prompts usable during the Optuna proxy search.
        heldout_prompts: Prompts never seen during search; used for the
            reported recovery metrics to detect proxy overfitting.
        neutral_prompts: Control prompts (concept-free) for drift and quality.
        metadata: Extra per-concept info (e.g. ImageNet wnid for objects).
    """

    name: str
    category: str
    search_prompts: Tuple[str, ...]
    heldout_prompts: Tuple[str, ...]
    neutral_prompts: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return _slug(self.name)

    @property
    def all_prompts(self) -> Tuple[str, ...]:
        return self.search_prompts + self.heldout_prompts

    def __post_init__(self) -> None:
        if self.category not in BENIGN_CATEGORIES:
            raise ValueError(
                f"Concept {self.name!r} has category {self.category!r}; only "
                f"benign categories {BENIGN_CATEGORIES} are permitted."
            )
        for attr in ("search_prompts", "heldout_prompts"):
            if not getattr(self, attr) and attr != "heldout_prompts":
                # held-out may be empty only while a pre-split spec is built
                # internally by build_concept_registry; _apply_split guarantees
                # a non-empty held-out set on any registry it returns.
                raise ValueError(f"Concept {self.name!r} has empty {attr}.")
        overlap = set(self.search_prompts) & set(self.heldout_prompts)
        if overlap:
            raise ValueError(
                f"Concept {self.name!r} has {len(overlap)} prompt(s) in both "
                "search and held-out sets; the split must be disjoint."
            )


class ConceptRegistry:
    """An immutable collection of :class:`ConceptSpec` with lookup helpers."""

    def __init__(self, concepts: Sequence[ConceptSpec], neutral_prompts: Sequence[str]):
        if not concepts:
            raise ValueError("A registry must contain at least one concept.")
        seen: set[str] = set()
        for concept in concepts:
            if concept.slug in seen:
                raise ValueError(f"Duplicate concept slug {concept.slug!r}.")
            seen.add(concept.slug)
        if not neutral_prompts:
            raise ValueError("A registry must define neutral control prompts.")
        self._concepts: Tuple[ConceptSpec, ...] = tuple(concepts)
        self._neutral: Tuple[str, ...] = tuple(neutral_prompts)

    @property
    def concepts(self) -> Tuple[ConceptSpec, ...]:
        return self._concepts

    @property
    def neutral_prompts(self) -> Tuple[str, ...]:
        return self._neutral

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(c.name for c in self._concepts)

    def by_category(self, category: str) -> Tuple[ConceptSpec, ...]:
        if category not in BENIGN_CATEGORIES:
            raise ValueError(f"Unknown category {category!r}.")
        return tuple(c for c in self._concepts if c.category == category)

    def get(self, name: str) -> ConceptSpec:
        slug = _slug(name)
        for concept in self._concepts:
            if concept.slug == slug:
                return concept
        raise KeyError(f"Unknown concept {name!r}; available: {list(self.names)}")

    def __contains__(self, name: str) -> bool:
        try:
            self.get(name)
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        return len(self._concepts)


def _check_registry_categories(specs: List[ConceptSpec]) -> None:
    for spec in specs:
        if spec.category not in BENIGN_CATEGORIES:
            raise ValueError(f"Non-benign category {spec.category!r} for {spec.name!r}.")


def build_concept_registry(
    config_path: str | Path = DEFAULT_CONCEPTS_CONFIG,
    max_per_category: Optional[Mapping[str, int]] = None,
    seed: Optional[int] = None,
) -> ConceptRegistry:
    """Build a :class:`ConceptRegistry` from a YAML config.

    Args:
        config_path: Path to a concepts YAML file (see
            ``configs/concepts_default.yaml`` for the schema).
        max_per_category: Optional override of per-category concept limits.
        seed: Optional override of the config's split seed.

    Returns:
        A :class:`PromptSplit` whose registry carries the deterministic,
        seeded search / held-out split. Persist it with
        :func:`load_or_create_split` so every run reuses the identical split.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Concepts config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for key in ("objects", "styles", "celebrities", "templates", "neutral_prompts"):
        if key not in config:
            raise ValueError(f"Concepts config {config_path} is missing required key {key!r}.")

    limits = dict(config.get("max_per_category") or {})
    if max_per_category is not None:
        limits.update(max_per_category)

    templates_by_category: Dict[str, Sequence[str]] = {}
    raw_templates = config["templates"]
    for category in BENIGN_CATEGORIES:
        block = raw_templates.get(category)
        if not block or len(block) < 10:
            raise ValueError(
                f"Category {category!r} needs at least 10 prompt templates; "
                f"found {0 if not block else len(block)}."
            )
        templates_by_category[category] = block

    neutral = [str(p).strip() for p in config["neutral_prompts"]]

    concepts: List[ConceptSpec] = []
    config_keys = {"object": "objects", "style": "styles", "celebrity": "celebrities"}
    for category in BENIGN_CATEGORIES:
        entries = config.get(config_keys[category]) or []
        limit = limits.get(category)
        if limit is not None:
            entries = entries[: int(limit)]
        templates = templates_by_category[category]
        for entry in entries:
            name = str(entry["name"]).strip()
            meta = {k: v for k, v in entry.items() if k != "name"}
            prompts = tuple(t.replace("{{concept}}", name).strip() for t in templates)
            concepts.append(
                ConceptSpec(
                    name=name,
                    category=category,
                    search_prompts=prompts,
                    heldout_prompts=(),  # filled by the split step
                    neutral_prompts=tuple(neutral),
                    metadata=meta,
                )
            )

    # The registry is built with empty held-out sets; validate benignness now
    # so non-benign categories fail before any split is produced.
    _check_registry_categories(concepts)

    effective_seed = int(config["seed"] if seed is None else seed)
    split_cfg = config.get("split") or {}
    search_fraction = float(split_cfg.get("search_fraction", 0.6))
    min_per_concept = int(split_cfg.get("min_per_concept", 10))

    registry = _apply_split(
        ConceptRegistry(concepts, neutral),
        seed=effective_seed,
        search_fraction=search_fraction,
        min_per_concept=min_per_concept,
    )
    return PromptSplit(
        seed=effective_seed,
        registry=registry,
        fingerprint=split_fingerprint(registry, effective_seed),
    )


def _apply_split(
    registry: ConceptRegistry,
    seed: int,
    search_fraction: float,
    min_per_concept: int,
) -> ConceptRegistry:
    """Deterministically re-split each concept's prompts into search / held-out."""
    if not 0.0 < search_fraction < 1.0:
        raise ValueError(f"search_fraction must be in (0, 1); got {search_fraction}.")
    rebuilt: List[ConceptSpec] = []
    for concept in registry.concepts:
        prompts = sorted(concept.search_prompts)
        if len(prompts) < min_per_concept:
            raise ValueError(
                f"Concept {concept.name!r} has {len(prompts)} prompts but the "
                f"config requires at least {min_per_concept}."
            )
        # Sort first so the split depends only on the prompt set, then derive
        # a per-concept RNG seed from the global seed + slug.
        digest = hashlib.sha256(f"{seed}:{concept.slug}".encode("utf-8")).digest()
        concept_seed = int.from_bytes(digest[:8], "little")
        order = list(range(len(prompts)))
        rng = _DeterministicShuffle(concept_seed)
        rng.shuffle(order)
        n_search = max(1, min(len(prompts) - 1, int(round(search_fraction * len(prompts)))))
        search_idx = set(order[:n_search])
        search = tuple(prompts[i] for i in range(len(prompts)) if i in search_idx)
        heldout = tuple(prompts[i] for i in range(len(prompts)) if i not in search_idx)
        rebuilt.append(
            ConceptSpec(
                name=concept.name,
                category=concept.category,
                search_prompts=search,
                heldout_prompts=heldout,
                neutral_prompts=concept.neutral_prompts,
                metadata=concept.metadata,
            )
        )
    return ConceptRegistry(rebuilt, registry.neutral_prompts)


class _DeterministicShuffle:
    """A tiny platform-independent deterministic shuffle (splitmix64-based).

    Using this instead of ``random.Random.shuffle`` keeps the split identical
    across Python versions and platforms.
    """

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self._state = seed & self._MASK

    def _next(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & self._MASK
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self._MASK
        return z ^ (z >> 31)

    def shuffle(self, items: List[int]) -> None:
        for i in reversed(range(1, len(items))):
            j = self._next() % (i + 1)
            items[i], items[j] = items[j], items[i]


@dataclass(frozen=True)
class PromptSplit:
    """The persisted, verifiable search / held-out split for a registry."""

    seed: int
    registry: ConceptRegistry
    fingerprint: str

    def save(self, split_dir: str | Path) -> Path:
        """Write the split manifest; returns the manifest path."""
        split_dir = Path(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "neutral_prompts": list(self.registry.neutral_prompts),
            "concepts": [
                {
                    "name": c.name,
                    "category": c.category,
                    "search_prompts": list(c.search_prompts),
                    "heldout_prompts": list(c.heldout_prompts),
                    "metadata": c.metadata,
                }
                for c in self.registry.concepts
            ],
        }
        path = split_dir / _SPLIT_MANIFEST_NAME
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, split_dir: str | Path) -> "PromptSplit":
        """Load a previously saved split from disk."""
        path = Path(split_dir) / _SPLIT_MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(f"No split manifest at {path}; create it first.")
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        concepts = [
            ConceptSpec(
                name=c["name"],
                category=c["category"],
                search_prompts=tuple(c["search_prompts"]),
                heldout_prompts=tuple(c["heldout_prompts"]),
                neutral_prompts=tuple(manifest["neutral_prompts"]),
                metadata=dict(c.get("metadata") or {}),
            )
            for c in manifest["concepts"]
        ]
        registry = ConceptRegistry(concepts, manifest["neutral_prompts"])
        seed = int(manifest["seed"])
        expected = split_fingerprint(registry, seed)
        if manifest.get("fingerprint") != expected:
            raise ValueError(
                f"Split manifest at {path} has fingerprint {manifest.get('fingerprint')!r} "
                f"but its contents hash to {expected!r}; the manifest was tampered with "
                "or hand-edited."
            )
        return cls(seed=seed, registry=registry, fingerprint=expected)


def split_fingerprint(registry: ConceptRegistry, seed: int) -> str:
    """Stable SHA-256 fingerprint of a registry split (prompts + seed)."""
    payload = {
        "seed": seed,
        "neutral_prompts": list(registry.neutral_prompts),
        "concepts": [
            {
                "name": c.name,
                "category": c.category,
                "search": list(c.search_prompts),
                "heldout": list(c.heldout_prompts),
            }
            for c in registry.concepts
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_or_create_split(
    config_path: str | Path = DEFAULT_CONCEPTS_CONFIG,
    split_dir: str | Path = "splits",
    max_per_category: Optional[Mapping[str, int]] = None,
    seed: Optional[int] = None,
) -> PromptSplit:
    """Load the persisted split, or create and persist it on first use.

    Every run that points at the same ``split_dir`` and config reuses the
    identical split. If a manifest already exists, the freshly-built split is
    compared against it by fingerprint; a mismatch is an error, never a
    silent regeneration.
    """
    built = build_concept_registry(config_path, max_per_category=max_per_category, seed=seed)
    path = Path(split_dir) / _SPLIT_MANIFEST_NAME
    if path.exists():
        existing = PromptSplit.load(split_dir)
        if existing.fingerprint != built.fingerprint:
            raise ValueError(
                f"Existing split in {split_dir} (fingerprint {existing.fingerprint[:12]}...) "
                f"does not match the requested config ({built.fingerprint[:12]}...). Use a "
                "different split_dir or delete the old manifest deliberately."
            )
        return existing
    built.save(split_dir)
    return built
