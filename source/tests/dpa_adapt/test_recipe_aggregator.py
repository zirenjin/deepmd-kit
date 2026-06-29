# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for RecipeAggregator standalone module (Deliverable 3).

Mocks load_or_extract so no GPU or real DPA checkpoint is required.

Aggregation semantics: ``concat([group_vec_i, ...])`` where each
``group_vec = Σ wᵢ eᵢ`` within the group.  For the standard two-group
schema (end_groups + repeating) this reproduces ``create_pfp`` as
``concat([end_vec, rep_vec])`` with shape ``(2 * FEAT_DIM,)``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from dpa_adapt.data.recipe_aggregator import RecipeAggregator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEAT_DIM = 6


def _write_fake_system(sys_dir: Path, n_atoms: int = 2) -> None:
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "type_map.raw").write_text("H\nC\n", encoding="utf-8")
    (sys_dir / "type.raw").write_text("\n".join(["0"] * n_atoms) + "\n", encoding="utf-8")
    sd = sys_dir / "set.000"
    sd.mkdir(exist_ok=True)
    np.save(str(sd / "coord.npy"), np.zeros((1, n_atoms * 3), dtype=np.float64))
    np.save(
        str(sd / "box.npy"), (np.eye(3) * 100.0).reshape(1, 9).astype(np.float64)
    )


def _make_recipe_dir(tmp_path: Path) -> Path:
    recipe_dir = tmp_path / "poly"
    systems_dir = recipe_dir / "systems"
    for name in ["mono_a", "mono_b", "endgrp"]:
        _write_fake_system(systems_dir / name)
    return recipe_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecipeAggregatorInit:
    def test_default_params(self) -> None:
        agg = RecipeAggregator("dummy.pt")
        assert agg.pretrained == "dummy.pt"
        assert agg.pooling == "mean"
        assert agg.cache is True
        assert agg.model_branch is None

    def test_custom_params(self) -> None:
        agg = RecipeAggregator("ckpt.pt", model_branch="Omat24", pooling="sum", cache=False)
        assert agg.model_branch == "Omat24"
        assert agg.pooling == "sum"
        assert agg.cache is False

    def test_repr(self) -> None:
        agg = RecipeAggregator("ckpt.pt")
        assert "RecipeAggregator" in repr(agg)
        assert "ckpt.pt" in repr(agg)


class TestRecipeAggregatorAggregate:
    """Tests using single-group (repeating) recipes unless stated otherwise."""

    def _single_group_recipe(
        self,
        sys_names: list[str],
        weights: list[float],
        label: float = 42.0,
    ) -> dict:
        return {
            "label": label,
            "groups": {
                "repeating": [
                    {"path": f"systems/{n}", "weight": w}
                    for n, w in zip(sys_names, weights)
                ]
            },
        }

    def test_output_shape_single_group(self, tmp_path: Path) -> None:
        """Single repeating group → shape (FEAT_DIM,)."""
        recipe_dir = _make_recipe_dir(tmp_path)
        recipe = self._single_group_recipe(["mono_a", "mono_b"], [0.6, 0.4])

        def _fake(systems, pretrained, **kwargs):
            return np.ones((1, FEAT_DIM))

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fake):
            agg = RecipeAggregator("dummy.pt")
            emb = agg.aggregate(recipe, recipe_dir=recipe_dir)

        assert emb.shape == (FEAT_DIM,)

    def test_output_shape_two_groups(self, tmp_path: Path) -> None:
        """Two-group recipe (end_groups + repeating) → shape (2*FEAT_DIM,)."""
        recipe_dir = _make_recipe_dir(tmp_path)
        recipe = {
            "label": 1.0,
            "groups": {
                "end_groups": [{"path": "systems/endgrp", "weight": 0.1}],
                "repeating": [{"path": "systems/mono_a", "weight": 0.9}],
            },
        }

        def _fake(systems, pretrained, **kwargs):
            return np.ones((1, FEAT_DIM))

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fake):
            agg = RecipeAggregator("dummy.pt")
            emb = agg.aggregate(recipe, recipe_dir=recipe_dir)

        assert emb.shape == (2 * FEAT_DIM,)

    def test_weighted_sum_single_group(self, tmp_path: Path) -> None:
        """Verify group_vec = Σ wᵢ eᵢ for a single-group recipe."""
        recipe_dir = _make_recipe_dir(tmp_path)

        emb_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        emb_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        def _fixed(systems, pretrained, **kwargs):
            src = str(getattr(systems[0], "_dpa_source", "") or "")
            if "mono_a" in src:
                return emb_a.reshape(1, -1)
            return emb_b.reshape(1, -1)

        recipe = self._single_group_recipe(["mono_a", "mono_b"], [0.4, 0.6], label=10.0)

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fixed):
            agg = RecipeAggregator("dummy.pt")
            emb = agg.aggregate(recipe, recipe_dir=recipe_dir)

        expected = 0.4 * emb_a + 0.6 * emb_b
        np.testing.assert_allclose(emb, expected, atol=1e-12)

    def test_two_group_concat_values(self, tmp_path: Path) -> None:
        """Two groups: output = concat([end_vec, rep_vec])."""
        recipe_dir = _make_recipe_dir(tmp_path)

        end_emb = np.arange(FEAT_DIM, dtype=float)          # [0,1,2,3,4,5]
        rep_emb = np.arange(FEAT_DIM, dtype=float) + 10.0  # [10,11,12,13,14,15]

        def _fixed(systems, pretrained, **kwargs):
            src = str(getattr(systems[0], "_dpa_source", "") or "")
            if "endgrp" in src:
                return end_emb.reshape(1, -1)
            return rep_emb.reshape(1, -1)

        recipe = {
            "label": 99.0,
            "groups": {
                "end_groups": [{"path": "systems/endgrp", "weight": 0.05}],
                "repeating": [{"path": "systems/mono_a", "weight": 0.95}],
            },
        }

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fixed):
            agg = RecipeAggregator("dummy.pt")
            emb = agg.aggregate(recipe, recipe_dir=recipe_dir)

        expected = np.concatenate([0.05 * end_emb, 0.95 * rep_emb])
        np.testing.assert_allclose(emb, expected, atol=1e-12)

    def test_zero_weight_components_skipped(self, tmp_path: Path) -> None:
        """Components with weight=0.0 must not trigger descriptor extraction."""
        recipe_dir = _make_recipe_dir(tmp_path)

        call_log: list[str] = []

        def _track(systems, pretrained, **kwargs):
            src = str(getattr(systems[0], "_dpa_source", "") or "")
            call_log.append(src)
            return np.ones((1, FEAT_DIM))

        recipe = {
            "label": 5.0,
            "groups": {
                "end_groups": [{"path": "systems/endgrp", "weight": 0.0}],
                "repeating": [{"path": "systems/mono_a", "weight": 1.0}],
            },
        }

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_track):
            agg = RecipeAggregator("dummy.pt")
            agg.aggregate(recipe, recipe_dir=recipe_dir)

        assert all("endgrp" not in p for p in call_log), (
            "Zero-weight end-group component should not trigger extraction."
        )

    def test_no_nonzero_components_raises(self, tmp_path: Path) -> None:
        recipe_dir = _make_recipe_dir(tmp_path)
        recipe = {
            "label": 0.0,
            "groups": {
                "end_groups": [{"path": "systems/endgrp", "weight": 0.0}],
                "repeating": [],
            },
        }
        with mock.patch(
            "dpa_adapt.finetuner.load_or_extract", return_value=np.ones((1, FEAT_DIM))
        ):
            agg = RecipeAggregator("dummy.pt")
            with pytest.raises(ValueError, match="non-zero-weight"):
                agg.aggregate(recipe, recipe_dir=recipe_dir)

    def test_absolute_path_resolution(self, tmp_path: Path) -> None:
        """Absolute component paths should be resolved without recipe_dir."""
        recipe_dir = _make_recipe_dir(tmp_path)
        abs_path = str((recipe_dir / "systems" / "mono_a").resolve())

        recipe = {
            "label": 1.0,
            "groups": {
                "repeating": [{"path": abs_path, "weight": 1.0}]
            },
        }

        def _fake(systems, pretrained, **kwargs):
            return np.ones((1, FEAT_DIM))

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fake):
            agg = RecipeAggregator("dummy.pt")
            emb = agg.aggregate(recipe)  # no recipe_dir

        assert emb.shape == (FEAT_DIM,)


class TestRecipeAggregatorTransform:
    def _simple_recipe(self, label: float = 0.0) -> dict:
        return {
            "label": label,
            "groups": {
                "repeating": [{"path": "systems/mono_a", "weight": 1.0}]
            },
        }

    def test_transform_shape(self, tmp_path: Path) -> None:
        """Single-group recipes: transform output shape (n, FEAT_DIM)."""
        recipe_dir = _make_recipe_dir(tmp_path)
        recipes = [self._simple_recipe(float(i)) for i in range(5)]

        def _fake(systems, pretrained, **kwargs):
            return np.ones((1, FEAT_DIM))

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fake):
            agg = RecipeAggregator("dummy.pt")
            X = agg.transform(recipes, recipe_dir=recipe_dir)

        assert X.shape == (5, FEAT_DIM)

    def test_transform_empty_raises(self) -> None:
        agg = RecipeAggregator("dummy.pt")
        with pytest.raises(ValueError, match="empty"):
            agg.transform([])

    def test_transform_deterministic(self, tmp_path: Path) -> None:
        """Same input → same output (important for reproducibility)."""
        recipe_dir = _make_recipe_dir(tmp_path)
        recipes = [self._simple_recipe(1.0)]
        fixed = np.arange(FEAT_DIM, dtype=float)

        def _fixed_emb(systems, pretrained, **kwargs):
            return fixed.reshape(1, -1)

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fixed_emb):
            agg = RecipeAggregator("dummy.pt")
            X1 = agg.transform(recipes, recipe_dir=recipe_dir)
            X2 = agg.transform(recipes, recipe_dir=recipe_dir)

        np.testing.assert_array_equal(X1, X2)


# ---------------------------------------------------------------------------
# DPAFineTuner auto-detection of recipe directory (Deliverable 3 routing)
# ---------------------------------------------------------------------------


class TestDPAFineTunerRecipeRouting:
    """Test that DPAFineTuner.fit() detects recipe dirs and routes to _fit_recipe."""

    def _make_recipe_dataset(self, tmp_path: Path) -> Path:
        recipe_dir = _make_recipe_dir(tmp_path)

        recipes = {
            f"recipe_{i:03d}": {
                "label": float(i * 5),
                "groups": {
                    "end_groups": [
                        {"path": "systems/endgrp", "weight": 0.05},
                    ],
                    "repeating": [
                        {"path": "systems/mono_a", "weight": 0.6},
                        {"path": "systems/mono_b", "weight": 0.4},
                    ],
                },
            }
            for i in range(4)
        }
        (recipe_dir / "recipe.json").write_text(
            json.dumps(recipes, indent=2), encoding="utf-8"
        )
        return recipe_dir

    def test_fit_detects_recipe_dir(self, tmp_path: Path) -> None:
        from dpa_adapt.finetuner import DPAFineTuner

        recipe_dir = self._make_recipe_dataset(tmp_path)

        call_log: list = []

        def _fake_extract(systems, pretrained, **kwargs):
            call_log.append(len(systems))
            return np.random.default_rng(0).random((1, FEAT_DIM))

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fake_extract), \
             mock.patch("dpa_adapt.finetuner.DPAFineTuner._load_descriptor_model", return_value=None):
            ft = DPAFineTuner(
                pretrained="dummy.pt",
                strategy="frozen_sklearn",
                predictor="linear",
            )
            ft._checkpoint_type_map = ["H", "C"]
            ft.fit(str(recipe_dir), target_key="label")

        assert ft._fitted
        assert ft.predictor is not None

    def test_fit_finetune_strategy_warns_and_falls_back(self, tmp_path: Path, recwarn) -> None:
        """finetune strategy on recipe dir should warn and fall back to sklearn."""
        from dpa_adapt.finetuner import DPAFineTuner

        recipe_dir = self._make_recipe_dataset(tmp_path)

        def _fake_extract(systems, pretrained, **kwargs):
            return np.random.default_rng(1).random((1, FEAT_DIM))

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fake_extract), \
             mock.patch("dpa_adapt.finetuner.DPAFineTuner._load_descriptor_model", return_value=None):
            ft = DPAFineTuner(
                pretrained="dummy.pt",
                strategy="finetune",
                predictor="linear",
                property_name="label",
            )
            ft._checkpoint_type_map = ["H", "C"]
            ft.fit(str(recipe_dir))

        assert ft._fitted
