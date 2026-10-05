"""Tests for the benign concept registry and deterministic prompt splits."""

from __future__ import annotations

import json

import pytest
import yaml

from heretic_dit.benchmarks.concepts import (
    BENIGN_CATEGORIES,
    ConceptSpec,
    PromptSplit,
    build_concept_registry,
    load_or_create_split,
)


def test_default_config_builds_all_categories():
    split = build_concept_registry()
    registry = split.registry
    assert len(registry) == 17  # 10 objects + 4 styles + 3 celebrities
    assert set(registry.names) >= {"tench", "chain saw", "van gogh", "tom hanks"}
    for category in BENIGN_CATEGORIES:
        assert registry.by_category(category)


def test_every_concept_has_ten_plus_prompts_and_disjoint_split():
    registry = build_concept_registry().registry
    for concept in registry.concepts:
        assert len(concept.search_prompts) >= 3
        assert len(concept.heldout_prompts) >= 2
        assert len(concept.search_prompts) + len(concept.heldout_prompts) >= 10
        assert not (set(concept.search_prompts) & set(concept.heldout_prompts))
        for prompt in concept.all_prompts:
            assert concept.name in prompt.lower() or concept.slug in prompt.lower()
        # neutral prompts are attached and concept-free
        for prompt in concept.neutral_prompts:
            assert concept.name not in prompt.lower()


def test_split_is_deterministic_across_calls_and_config_hashes():
    first = build_concept_registry()
    second = build_concept_registry()
    assert first.fingerprint == second.fingerprint
    for a, b in zip(first.registry.concepts, second.registry.concepts):
        assert a.search_prompts == b.search_prompts
        assert a.heldout_prompts == b.heldout_prompts


def test_seed_changes_the_split_fingerprint():
    assert build_concept_registry(seed=1).fingerprint != build_concept_registry(seed=2).fingerprint


def test_split_reuse_is_bit_identical_and_guarded(tmp_path):
    created = load_or_create_split(split_dir=tmp_path)
    reused = load_or_create_split(split_dir=tmp_path)
    assert reused.fingerprint == created.fingerprint
    manifest = json.loads((tmp_path / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == created.seed
    # Same dir + different config must fail loudly, never regenerate silently.
    with pytest.raises(ValueError, match="fingerprint"):
        load_or_create_split(split_dir=tmp_path, seed=created.seed + 1)


def test_split_roundtrip_via_save_and_load(tmp_path):
    split = build_concept_registry()
    path = split.save(tmp_path)
    assert path.exists()
    loaded = PromptSplit.load(tmp_path)
    assert loaded.fingerprint == split.fingerprint
    assert loaded.registry.names == split.registry.names
    original = split.registry.get("van gogh")
    roundtripped = loaded.registry.get("van gogh")
    assert original.search_prompts == roundtripped.search_prompts
    assert original.heldout_prompts == roundtripped.heldout_prompts


def test_non_benign_category_is_rejected():
    with pytest.raises(ValueError, match="benign"):
        ConceptSpec(
            name="x",
            category="unsafe",
            search_prompts=("a", "b"),
            heldout_prompts=("c",),
            neutral_prompts=("n",),
        )


def test_max_per_category_override_limits_list_size():
    registry = build_concept_registry(max_per_category={"object": 2, "style": 1, "celebrity": 1}).registry
    assert len(registry) == 4
    assert len(registry.by_category("object")) == 2


def test_missing_templates_fail_loudly(tmp_path):
    config = {
        "seed": 0,
        "objects": [{"name": "church"}],
        "styles": [],
        "celebrities": [],
        "templates": {"object": ["t1", "t2"], "style": ["s" * 10], "celebrity": ["c" * 10]},
        "neutral_prompts": ["n"],
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 10"):
        build_concept_registry(path)
