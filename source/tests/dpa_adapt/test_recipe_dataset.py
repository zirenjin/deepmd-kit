# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for RecipeDataset (Deliverable 2).

The DPA backbone extraction (load_or_extract) is patched with a
deterministic stub so no GPU or real checkpoint is needed.

Aggregation semantics: ``concat([group_vec_i, ...])`` where each
``group_vec = Σ wᵢ eᵢ``.  For the standard two-group schema (end_groups
+ repeating) the output shape is (2 * FEAT_DIM,).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from dpa_adapt.data.recipe_dataset import RecipeDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEAT_DIM = 4


def _write_fake_system(sys_dir: Path, n_atoms: int = 2) -> None:
    """Minimal deepmd/npy system (no RDKit / dpdata needed)."""
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "type_map.raw").write_text("H\nC\n", encoding="utf-8")
    (sys_dir / "type.raw").write_text("\n".join(["0"] * n_atoms) + "\n", encoding="utf-8")
    sd = sys_dir / "set.000"
    sd.mkdir(exist_ok=True)
    np.save(str(sd / "coord.npy"), np.zeros((1, n_atoms * 3), dtype=np.float64))
    np.save(
        str(sd / "box.npy"), (np.eye(3) * 100.0).reshape(1, 9).astype(np.float64)
    )


def _make_recipe_dir(
    tmp_path: Path,
    n_recipes: int = 3,
    n_components: int = 2,
) -> tuple[Path, list[str]]:
    """Build a minimal recipe directory with a single-group (repeating) schema.

    Uses one ``repeating`` group so embedding shape is ``(FEAT_DIM,)``.
    """
    recipe_dir = tmp_path / "dataset"
    systems_dir = recipe_dir / "systems"

    smiles = [f"smiles_{i}" for i in range(n_components)]
    sys_paths: list[str] = []
    for smi in smiles:
        h = smi  # use smi directly as "hash" for test simplicity
        sys_dir = systems_dir / f"monomer_{h}"
        _write_fake_system(sys_dir)
        sys_paths.append(f"systems/monomer_{h}")

    # Build recipe.json with groups schema (single repeating group).
    recipes: dict = {}
    for i in range(n_recipes):
        weight_a = 0.6
        weight_b = 0.4
        recipes[f"recipe_{i:03d}"] = {
            "label": float(i * 10 + 1),
            "groups": {
                "repeating": [
                    {"path": sys_paths[0], "weight": weight_a},
                    {"path": sys_paths[1], "weight": weight_b},
                ]
            },
        }

    recipe_path = recipe_dir / "recipe.json"
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(json.dumps(recipes, indent=2), encoding="utf-8")

    return recipe_dir, smiles


def _fake_load_or_extract(systems, pretrained, **kwargs):
    """Return deterministic stub descriptors for any system list."""
    rng = np.random.default_rng(42)
    n_frames = sum(
        s.data["coords"].shape[0] if hasattr(s, "data") else 1 for s in systems
    )
    return rng.random((n_frames, FEAT_DIM))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecipeDatasetInit:
    def test_loads_recipes(self, tmp_path: Path) -> None:
        recipe_dir, _ = _make_recipe_dir(tmp_path, n_recipes=4)
        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fake_load_or_extract):
            ds = RecipeDataset(recipe_dir, pretrained="dummy.pt")
        assert len(ds) == 4

    def test_missing_recipe_json_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="recipe.json"):
            RecipeDataset(empty_dir, pretrained="dummy.pt")

    def test_repr(self, tmp_path: Path) -> None:
        recipe_dir, _ = _make_recipe_dir(tmp_path)
        ds = RecipeDataset.__new__(RecipeDataset)
        ds.recipe_dir = recipe_dir
        ds.pretrained = "dummy.pt"
        ds.model_branch = None
        ds.pooling = "mean"
        ds.cache = True
        ds._recipes = {
            "recipe_000": {
                "label": 1.0,
                "groups": {"repeating": []},
            }
        }
        ds._recipe_keys = ["recipe_000"]
        ds._embeddings = None
        ds._labels = None
        r = repr(ds)
        assert "RecipeDataset" in r
        assert "recipe_000" not in r  # keys not in repr


class TestRecipeDatasetGetitem:
    def test_getitem_returns_tuple(self, tmp_path: Path) -> None:
        """Single-group recipe → embedding shape (FEAT_DIM,)."""
        recipe_dir, _ = _make_recipe_dir(tmp_path, n_recipes=2)
        with mock.patch(
            "dpa_adapt.finetuner.load_or_extract", side_effect=_fake_load_or_extract
        ):
            ds = RecipeDataset(recipe_dir, pretrained="dummy.pt")
            emb, label = ds[0]
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (FEAT_DIM,)
        assert isinstance(label, float)

    def test_label_values(self, tmp_path: Path) -> None:
        recipe_dir, _ = _make_recipe_dir(tmp_path, n_recipes=3)
        with mock.patch(
            "dpa_adapt.finetuner.load_or_extract", side_effect=_fake_load_or_extract
        ):
            ds = RecipeDataset(recipe_dir, pretrained="dummy.pt")
            labels = [ds[i][1] for i in range(len(ds))]
        # Labels from _make_recipe_dir are 1.0, 11.0, 21.0
        assert labels == pytest.approx([1.0, 11.0, 21.0])


class TestRecipeDatasetGetEmbeddings:
    def test_get_embeddings_shape(self, tmp_path: Path) -> None:
        """Single-group recipe → (n_recipes, FEAT_DIM)."""
        n = 5
        recipe_dir, _ = _make_recipe_dir(tmp_path, n_recipes=n)
        with mock.patch(
            "dpa_adapt.finetuner.load_or_extract", side_effect=_fake_load_or_extract
        ):
            ds = RecipeDataset(recipe_dir, pretrained="dummy.pt")
            X = ds.get_embeddings()
        assert X.shape == (n, FEAT_DIM)

    def test_two_group_shape(self, tmp_path: Path) -> None:
        """Two-group recipe (end_groups + repeating) → (n, 2*FEAT_DIM)."""
        recipe_dir = tmp_path / "two_group"
        systems_dir = recipe_dir / "systems"
        for name in ["endgrp", "mono_a"]:
            _write_fake_system(systems_dir / name)

        recipes = {
            "recipe_000": {
                "label": 5.0,
                "groups": {
                    "end_groups": [{"path": "systems/endgrp", "weight": 0.1}],
                    "repeating": [{"path": "systems/mono_a", "weight": 0.9}],
                },
            }
        }
        (recipe_dir / "recipe.json").parent.mkdir(parents=True, exist_ok=True)
        (recipe_dir / "recipe.json").write_text(json.dumps(recipes), encoding="utf-8")

        def _fake(systems, pretrained, **kwargs):
            return np.ones((1, FEAT_DIM))

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fake):
            ds = RecipeDataset(recipe_dir, pretrained="dummy.pt")
            X = ds.get_embeddings()

        assert X.shape == (1, 2 * FEAT_DIM)

    def test_get_embeddings_cached(self, tmp_path: Path) -> None:
        """Second call to get_embeddings() does not re-extract."""
        recipe_dir, _ = _make_recipe_dir(tmp_path)
        call_count = {"n": 0}

        def _counting_extract(systems, pretrained, **kwargs):
            call_count["n"] += 1
            return np.zeros((1, FEAT_DIM))

        with mock.patch(
            "dpa_adapt.finetuner.load_or_extract", side_effect=_counting_extract
        ):
            ds = RecipeDataset(recipe_dir, pretrained="dummy.pt")
            ds.get_embeddings()
            n_after_first = call_count["n"]
            ds.get_embeddings()
            n_after_second = call_count["n"]

        assert n_after_second == n_after_first  # no new extractions on 2nd call


class TestRecipeDatasetGetLabels:
    def test_get_labels_shape(self, tmp_path: Path) -> None:
        recipe_dir, _ = _make_recipe_dir(tmp_path, n_recipes=4)
        ds = RecipeDataset.__new__(RecipeDataset)
        ds.recipe_dir = recipe_dir
        ds.pretrained = "dummy.pt"
        ds.model_branch = None
        ds.pooling = "mean"
        ds.cache = True
        # Load from JSON directly
        with (recipe_dir / "recipe.json").open() as f:
            ds._recipes = json.load(f)
        ds._recipe_keys = sorted(ds._recipes.keys())
        ds._embeddings = None
        ds._labels = None
        y = ds.get_labels()
        assert y.shape == (4,)
        assert y.dtype == np.float64


class TestRecipeDatasetWeightedAggregation:
    def test_embedding_is_weighted_sum(self, tmp_path: Path) -> None:
        """Single repeating group: embedding = Σ wᵢ eᵢ over components."""
        recipe_dir = tmp_path / "ds"
        systems_dir = recipe_dir / "systems"

        emb_a = np.array([1.0, 0.0, 0.0, 0.0])
        emb_b = np.array([0.0, 1.0, 0.0, 0.0])

        for name in ["sys_a", "sys_b"]:
            _write_fake_system(systems_dir / name)

        recipe_path = recipe_dir / "recipe.json"
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(
            json.dumps(
                {
                    "recipe_000": {
                        "label": 99.0,
                        "groups": {
                            "repeating": [
                                {"path": "systems/sys_a", "weight": 0.3},
                                {"path": "systems/sys_b", "weight": 0.7},
                            ]
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        def _fixed_emb(systems, pretrained, **kwargs):
            src = getattr(systems[0], "_dpa_source", None) or ""
            if "sys_a" in str(src):
                return emb_a.reshape(1, -1)
            return emb_b.reshape(1, -1)

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fixed_emb):
            ds = RecipeDataset(recipe_dir, pretrained="dummy.pt")
            emb, _ = ds[0]

        expected = 0.3 * emb_a + 0.7 * emb_b
        np.testing.assert_allclose(emb, expected, atol=1e-10)

    def test_two_group_concat_values(self, tmp_path: Path) -> None:
        """Two groups: output = concat([end_vec, rep_vec])."""
        recipe_dir = tmp_path / "ds"
        systems_dir = recipe_dir / "systems"

        end_emb = np.array([1.0, 2.0, 3.0, 4.0])   # returned for endgrp
        rep_emb = np.array([5.0, 6.0, 7.0, 8.0])   # returned for mono_a

        for name in ["endgrp", "mono_a"]:
            _write_fake_system(systems_dir / name)

        recipe_path = recipe_dir / "recipe.json"
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(
            json.dumps(
                {
                    "recipe_000": {
                        "label": 42.0,
                        "groups": {
                            "end_groups": [
                                {"path": "systems/endgrp", "weight": 0.05}
                            ],
                            "repeating": [
                                {"path": "systems/mono_a", "weight": 0.95}
                            ],
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        def _fixed_emb(systems, pretrained, **kwargs):
            src = str(getattr(systems[0], "_dpa_source", "") or "")
            if "endgrp" in src:
                return end_emb.reshape(1, -1)
            return rep_emb.reshape(1, -1)

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_fixed_emb):
            ds = RecipeDataset(recipe_dir, pretrained="dummy.pt")
            emb, _ = ds[0]

        expected = np.concatenate([0.05 * end_emb, 0.95 * rep_emb])
        assert emb.shape == (2 * FEAT_DIM,)
        np.testing.assert_allclose(emb, expected, atol=1e-10)

    def test_zero_weight_end_groups_contribute_zero_vector(self, tmp_path: Path) -> None:
        """End groups with weight=0.0 skip extraction; their group vec is zeros."""
        recipe_dir = tmp_path / "ds"
        systems_dir = recipe_dir / "systems"

        _write_fake_system(systems_dir / "main")
        _write_fake_system(systems_dir / "endgrp")

        recipe_path = recipe_dir / "recipe.json"
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(
            json.dumps(
                {
                    "recipe_000": {
                        "label": 1.0,
                        "groups": {
                            "end_groups": [
                                {"path": "systems/endgrp", "weight": 0.0}
                            ],
                            "repeating": [
                                {"path": "systems/main", "weight": 1.0}
                            ],
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        call_log: list[str] = []

        def _track_calls(systems, pretrained, **kwargs):
            src = getattr(systems[0], "_dpa_source", "") or ""
            call_log.append(str(src))
            return np.ones((1, FEAT_DIM))

        with mock.patch("dpa_adapt.finetuner.load_or_extract", side_effect=_track_calls):
            ds = RecipeDataset(recipe_dir, pretrained="dummy.pt")
            emb, _ = ds[0]

        # Only the main system (weight=1.0) should have triggered an extraction call.
        assert all("endgrp" not in p for p in call_log)

        # end_groups vector is zeros; repeating vector is ones.
        expected = np.concatenate([np.zeros(FEAT_DIM), np.ones(FEAT_DIM)])
        np.testing.assert_allclose(emb, expected, atol=1e-10)
