# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the acquisition layer (acquisition.py).

No real DPA checkpoint or torch required: ``dpa_adapt.acquisition.load_or_extract``
is monkeypatched to return deterministic synthetic descriptors.  Real tiny
deepmd/npy system directories are created on disk so that ``load_data`` and
``System.get_nframes`` work correctly.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import dpa_adapt.acquisition as acq_mod
from dpa_adapt.acquisition import (
    Surrogate,
    _ei,
    _farthest_point,
    _pi,
    _ucb,
    acquire,
    composition_pool,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEAT_DIM = 8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sys_dir(
    tmp_path: Path,
    name: str = "sys",
    n_frames: int = 3,
    n_atoms: int = 2,
) -> str:
    """Create a minimal deepmd/npy system directory.

    Returns the string path to the root.  Only coord.npy/box.npy/type.raw/
    type_map.raw are written — enough for dpdata to load and for
    ``System.get_nframes()`` to return *n_frames*.
    """
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "type.raw").write_text(
        "\n".join(str(i % 2) for i in range(n_atoms)) + "\n"
    )
    (root / "type_map.raw").write_text("H\nO\n")
    sd = root / "set.000"
    sd.mkdir()
    rng = np.random.default_rng(42)
    np.save(sd / "coord.npy", rng.random((n_frames, n_atoms * 3)))
    np.save(sd / "box.npy", np.tile(np.eye(3).ravel(), (n_frames, 1)))
    return str(root)


def _fake_load_or_extract(
    systems,
    pretrained,
    model_branch=None,
    pooling="mean",
    cache=True,
    type_map=None,
):
    """Stub for load_or_extract: returns deterministic features, no torch."""
    n_frames = sum(s.get_nframes() for s in systems)
    return np.random.default_rng(0).random((n_frames, FEAT_DIM))


def _surrogate_fit_mock(tmp_path: Path, n_train: int = 5) -> tuple[Surrogate, str]:
    """Create and fit a Surrogate with monkeypatched descriptor extraction.

    Returns the fitted surrogate and the training system directory path.
    """
    train_dir = _make_sys_dir(tmp_path, "train", n_frames=n_train)
    labels = np.linspace(0.1, 1.0, n_train)

    with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
        surr = Surrogate(pretrained="fake.pt")
        surr.fit([train_dir], labels=labels)

    return surr, train_dir


# ---------------------------------------------------------------------------
# 1) Pure helpers
# ---------------------------------------------------------------------------


class TestUCB:
    def test_max_objective(self):
        mu = np.array([0.0, 1.0, 2.0])
        sigma = np.array([1.0, 1.0, 1.0])
        scores = _ucb(mu, sigma, beta=1.0, objective="max")
        # Scores should be mu + sigma = [1, 2, 3]
        np.testing.assert_allclose(scores, [1.0, 2.0, 3.0])

    def test_min_objective_flips(self):
        mu = np.array([0.0, 1.0, 2.0])
        sigma = np.array([1.0, 1.0, 1.0])
        scores_max = _ucb(mu, sigma, beta=1.0, objective="max")
        scores_min = _ucb(mu, sigma, beta=1.0, objective="min")
        # For min, lower mu should score higher
        assert scores_min[0] > scores_min[2], "min objective should favour low mu"
        # Ranking should be reversed
        assert np.argmax(scores_min) == 0
        assert np.argmax(scores_max) == 2

    def test_beta_zero_ignores_sigma(self):
        mu = np.array([3.0, 1.0, 2.0])
        sigma = np.array([10.0, 10.0, 10.0])
        scores = _ucb(mu, sigma, beta=0.0, objective="max")
        np.testing.assert_allclose(scores, mu)


class TestEI:
    def test_ei_nonnegative(self):
        mu = np.random.default_rng(0).random(20)
        sigma = np.abs(np.random.default_rng(1).random(20)) + 0.01
        ei = _ei(mu, sigma, best=0.5, xi=0.0, objective="max")
        assert np.all(ei >= 0), "EI must be non-negative"

    def test_ei_zero_where_sigma_zero(self):
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.0, 1.0, 0.0])
        ei = _ei(mu, sigma, best=0.0, xi=0.0, objective="max")
        assert ei[0] == 0.0
        assert ei[2] == 0.0
        assert ei[1] > 0.0

    def test_ei_max_objective_ranks_high_mu(self):
        mu = np.array([0.0, 5.0])
        sigma = np.array([1.0, 1.0])
        ei = _ei(mu, sigma, best=0.0, xi=0.0, objective="max")
        assert ei[1] > ei[0], "higher mu → higher EI (max objective)"

    def test_ei_min_objective_ranks_low_mu(self):
        mu = np.array([0.0, 5.0])
        sigma = np.array([1.0, 1.0])
        ei = _ei(mu, sigma, best=5.0, xi=0.0, objective="min")
        assert ei[0] > ei[1], "lower mu → higher EI (min objective)"


class TestPI:
    def test_pi_in_unit_interval(self):
        mu = np.random.default_rng(0).random(20)
        sigma = np.abs(np.random.default_rng(1).random(20)) + 0.01
        pi = _pi(mu, sigma, best=0.5, xi=0.0, objective="max")
        assert np.all(pi >= 0) and np.all(pi <= 1), "PI must be in [0, 1]"

    def test_pi_zero_where_sigma_zero(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.0, 1.0])
        pi = _pi(mu, sigma, best=0.0, xi=0.0, objective="max")
        assert pi[0] == 0.0
        assert pi[1] > 0.0

    def test_pi_min_objective(self):
        mu = np.array([0.0, 5.0])
        sigma = np.array([1.0, 1.0])
        pi = _pi(mu, sigma, best=5.0, xi=0.0, objective="min")
        assert pi[0] > pi[1], "lower mu → higher PI (min objective)"


class TestFarthestPoint:
    def test_returns_k_indices(self):
        X_pool = np.eye(5)  # 5 orthogonal unit vectors
        X_seen = np.zeros((0, 5))
        selected = _farthest_point(X_pool, X_seen, k=3)
        assert len(selected) == 3
        assert len(set(selected)) == 3, "indices must be distinct"

    def test_indices_in_range(self):
        rng = np.random.default_rng(7)
        X_pool = rng.random((10, 4))
        X_seen = rng.random((2, 4))
        selected = _farthest_point(X_pool, X_seen, k=5)
        assert all(0 <= idx < 10 for idx in selected)

    def test_spread_across_extremes(self):
        # Two clusters far apart: left cluster (x<0) and right cluster (x>0).
        left = np.column_stack([np.full(3, -10.0), np.zeros(3)])
        right = np.column_stack([np.full(3, 10.0), np.zeros(3)])
        X_pool = np.vstack([left, right])  # indices 0-2 = left, 3-5 = right
        X_seen = np.zeros((0, 2))
        selected = _farthest_point(X_pool, X_seen, k=2)
        # Should pick one from each side
        sides = {int(idx >= 3) for idx in selected}
        assert sides == {0, 1}, f"Should pick from both clusters, got {selected}"

    def test_with_weights_first_pick_is_argmax_weights(self):
        X_pool = np.eye(4)
        X_seen = np.zeros((0, 4))
        weights = np.array([0.1, 0.9, 0.5, 0.2])
        selected = _farthest_point(X_pool, X_seen, k=1, weights=weights)
        assert selected[0] == 1, "first pick should be argmax(weights) when X_seen empty"

    def test_with_seen_points_avoids_seen(self):
        # X_pool = two points; X_seen = first point.
        # The farthest from X_seen is the second pool point.
        X_pool = np.array([[0.0, 0.0], [10.0, 0.0]])
        X_seen = np.array([[0.0, 0.0]])
        selected = _farthest_point(X_pool, X_seen, k=1)
        assert selected[0] == 1, "should pick the point farthest from X_seen"


# ---------------------------------------------------------------------------
# 2) Surrogate.fit / predict
# ---------------------------------------------------------------------------


class TestSurrogateFitPredict:
    def test_fit_sets_fitted_flag(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path)
        assert surr._fitted is True

    def test_fit_stores_y_train_and_scaler(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=5)
        assert surr._y_train is not None
        assert surr._y_train.shape == (5, 1)
        assert surr._scaler is not None

    def test_predict_shape_single_target(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=10)

        # Pool: 4 systems, each with 1 frame.
        pool_dirs = [
            _make_sys_dir(tmp_path, f"pool_{i}", n_frames=1) for i in range(4)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = surr.predict(pool_dirs)

        assert result.mean.shape == (4,), f"Expected (4,), got {result.mean.shape}"
        assert result.std.shape == (4,), f"Expected (4,), got {result.std.shape}"
        assert result.features.shape == (4, FEAT_DIM)
        assert len(result.systems) == 4

    def test_predict_std_positive(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=10)
        pool_dirs = [
            _make_sys_dir(tmp_path, f"pool2_{i}", n_frames=1) for i in range(3)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = surr.predict(pool_dirs)

        assert np.all(result.std > 0), "BayesianRidge should give positive std"

    def test_predict_multi_target_shape(self, tmp_path):
        train_dir = _make_sys_dir(tmp_path, "train_mt", n_frames=8)
        # Two-column labels: (n_frames, 2)
        labels = np.column_stack(
            [np.linspace(0, 1, 8), np.linspace(1, 0, 8)]
        )

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            surr = Surrogate(pretrained="fake.pt")
            surr.fit([train_dir], labels=labels)

        pool_dirs = [
            _make_sys_dir(tmp_path, f"pool_mt_{i}", n_frames=1) for i in range(3)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = surr.predict(pool_dirs)

        assert result.mean.shape == (3, 2), f"Expected (3,2), got {result.mean.shape}"
        assert result.std.shape == (3, 2), f"Expected (3,2), got {result.std.shape}"

    def test_predict_before_fit_raises(self, tmp_path):
        surr = Surrogate(pretrained="fake.pt")
        pool_dirs = [_make_sys_dir(tmp_path, "unfitted_pool", n_frames=1)]
        with pytest.raises(RuntimeError, match="fit"):
            with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
                surr.predict(pool_dirs)

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            Surrogate(pretrained="fake.pt", kind="xgboost")

    def test_gp_kind_raises_not_implemented(self, tmp_path):
        train_dir = _make_sys_dir(tmp_path, "train_gp", n_frames=5)
        surr = Surrogate(pretrained="fake.pt", kind="gp")
        with pytest.raises(NotImplementedError, match="gp"):
            with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
                surr.fit([train_dir], labels=np.ones(5))


# ---------------------------------------------------------------------------
# 3) acquire function
# ---------------------------------------------------------------------------


class TestAcquireUncertainty:
    def test_picks_highest_sigma(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=10)
        pool_dirs = [
            _make_sys_dir(tmp_path, f"ua_{i}", n_frames=1) for i in range(5)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = acquire(surr, pool_dirs, strategy="uncertainty", batch_size=1)

        # The selected candidate should have the max std.
        assert len(result.selected) == 1
        assert len(result.indices) == 1
        sel_idx = result.indices[0]
        assert result.std[sel_idx] == result.std.max(), (
            "uncertainty picks the highest-sigma candidate"
        )


class TestAcquireUCB:
    def test_ucb_returns_batch(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=10)
        pool_dirs = [
            _make_sys_dir(tmp_path, f"ub_{i}", n_frames=1) for i in range(6)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = acquire(
                surr, pool_dirs, strategy="ucb", batch_size=3, beta=2.0, objective="max"
            )

        assert len(result.selected) == 3
        assert len(result.indices) == 3
        assert len(set(result.indices)) == 3, "selected indices must be distinct"

    def test_ucb_ranking_consistent_with_scores(self, tmp_path):
        """Selected candidates should have the top-k UCB scores."""
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=10)
        pool_dirs = [
            _make_sys_dir(tmp_path, f"ucb_rank_{i}", n_frames=1) for i in range(8)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = acquire(
                surr, pool_dirs, strategy="ucb", batch_size=3, beta=2.0
            )
            # Compute what the UCB scores should be.
            from dpa_adapt.acquisition import _ucb as _ucb_fn

            all_ucb = _ucb_fn(result.mean, result.std, beta=2.0, objective="max")

        top3_expected = set(np.argsort(all_ucb)[::-1][:3].tolist())
        assert set(result.indices) == top3_expected


class TestAcquireMinObjective:
    def test_min_objective_flips_selection(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=10)
        pool_dirs = [
            _make_sys_dir(tmp_path, f"min_obj_{i}", n_frames=1) for i in range(5)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            res_max = acquire(
                surr, pool_dirs, strategy="ucb", batch_size=1, beta=0.0, objective="max"
            )
            res_min = acquire(
                surr, pool_dirs, strategy="ucb", batch_size=1, beta=0.0, objective="min"
            )

        # With beta=0, UCB reduces to ±mu; min should pick the lowest mu.
        mu = res_max.mean
        assert res_max.indices[0] == int(np.argmax(mu))
        assert res_min.indices[0] == int(np.argmin(mu))


class TestAcquireEI:
    def test_ei_returns_nonneg_scores(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=8)
        pool_dirs = [
            _make_sys_dir(tmp_path, f"ei_{i}", n_frames=1) for i in range(4)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = acquire(surr, pool_dirs, strategy="ei", batch_size=2)

        assert result.scores is not None
        assert np.all(result.scores >= 0)

    def test_ei_min_objective(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=8)
        pool_dirs = [
            _make_sys_dir(tmp_path, f"ei_min_{i}", n_frames=1) for i in range(4)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = acquire(
                surr, pool_dirs, strategy="ei", batch_size=1, objective="min"
            )

        assert result.scores is not None
        assert np.all(result.scores >= 0)


class TestAcquirePI:
    def test_pi_returns_valid_result(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=8)
        pool_dirs = [
            _make_sys_dir(tmp_path, f"pi_{i}", n_frames=1) for i in range(4)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = acquire(surr, pool_dirs, strategy="pi", batch_size=2)

        assert len(result.selected) == 2
        assert result.scores is not None


class TestAcquireDiverse:
    def test_diverse_returns_k_distinct(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=10)
        pool_dirs = [
            _make_sys_dir(tmp_path, f"div_{i}", n_frames=1) for i in range(6)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = acquire(surr, pool_dirs, strategy="diverse", batch_size=4)

        assert len(result.selected) == 4
        assert len(set(result.indices)) == 4, "diverse must return distinct indices"

    def test_diverse_systems_paths_correct(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=8)
        pool_dirs = [
            _make_sys_dir(tmp_path, f"div2_{i}", n_frames=1) for i in range(5)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = acquire(surr, pool_dirs, strategy="diverse", batch_size=3)

        # All returned system paths should be in pool_dirs.
        for path in result.selected:
            assert path in pool_dirs


class TestAcquireValidation:
    def test_invalid_objective_raises(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path)
        with pytest.raises(ValueError, match="objective"):
            acquire(surr, [], objective="maximize")

    def test_invalid_strategy_raises(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path)
        with pytest.raises(ValueError, match="strategy"):
            acquire(surr, [], strategy="random")

    def test_batch_size_zero_raises(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path)
        with pytest.raises(ValueError, match="batch_size"):
            acquire(surr, [], batch_size=0)

    def test_bo_strategy_multi_target_raises(self, tmp_path):
        train_dir = _make_sys_dir(tmp_path, "bo_mt_train", n_frames=8)
        labels = np.column_stack(
            [np.linspace(0, 1, 8), np.linspace(1, 0, 8)]
        )

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            surr = Surrogate(pretrained="fake.pt")
            surr.fit([train_dir], labels=labels)

        pool_dirs = [
            _make_sys_dir(tmp_path, f"bo_mt_pool_{i}", n_frames=1) for i in range(3)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            with pytest.raises(ValueError, match="single-target"):
                acquire(surr, pool_dirs, strategy="ucb")

    def test_return_scores_false(self, tmp_path):
        surr, _ = _surrogate_fit_mock(tmp_path, n_train=8)
        pool_dirs = [
            _make_sys_dir(tmp_path, f"no_score_{i}", n_frames=1) for i in range(4)
        ]

        with patch.object(acq_mod, "load_or_extract", _fake_load_or_extract):
            result = acquire(
                surr, pool_dirs, strategy="uncertainty", return_scores=False
            )

        assert result.scores is None


# ---------------------------------------------------------------------------
# 4) composition_pool
# ---------------------------------------------------------------------------


class TestCompositionPool:
    def test_csv_written_with_formulas_and_placeholder(self, tmp_path):
        """candidates.csv must contain the formulas with Property=0.0."""
        formulas = ["Fe0.5Ni0.5", "Co0.3Fe0.7"]
        out_dir = str(tmp_path / "pool_out")

        # Monkeypatch formula_to_npy so we don't need a real POSCAR.
        fake_dirs = [str(tmp_path / f"sys_{i:04d}") for i in range(2)]

        with patch.object(acq_mod, "formula_to_npy", return_value=fake_dirs):
            result = composition_pool(
                formulas,
                poscar="fake.vasp",
                output_dir=out_dir,
                property_name="bandgap",
            )

        csv_path = os.path.join(out_dir, "candidates.csv")
        assert os.path.isfile(csv_path), "candidates.csv must be created"

        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["formula"] == "Fe0.5Ni0.5"
        assert float(rows[0]["Property"]) == 0.0
        assert rows[1]["formula"] == "Co0.3Fe0.7"
        assert float(rows[1]["Property"]) == 0.0

    def test_formula_to_npy_called_with_correct_args(self, tmp_path):
        formulas = ["Mn1"]
        out_dir = str(tmp_path / "pool_args")
        fake_dirs = [str(tmp_path / "sys_0000")]

        with patch.object(
            acq_mod, "formula_to_npy", return_value=fake_dirs
        ) as mock_fn:
            composition_pool(
                formulas,
                poscar="template.vasp",
                output_dir=out_dir,
                base_element="Mn",
                sets=2,
                seed=99,
                property_name="gap",
            )

        mock_fn.assert_called_once()
        _, kwargs = mock_fn.call_args
        # Check keyword args (formula_to_npy is called with keyword args).
        call_args = mock_fn.call_args
        pos_args = call_args[0]
        kw_args = call_args[1]

        # csv_path, output_dir, poscar are positional
        assert pos_args[0] == os.path.join(out_dir, "candidates.csv")
        assert pos_args[1] == out_dir
        assert pos_args[2] == "template.vasp"
        assert kw_args.get("base_element") == "Mn"
        assert kw_args.get("sets") == 2
        assert kw_args.get("seed") == 99
        assert kw_args.get("property_name") == "gap"

    def test_returns_list_of_dirs(self, tmp_path):
        formulas = ["Ti1", "V1", "Cr1"]
        out_dir = str(tmp_path / "pool_ret")
        fake_dirs = [str(tmp_path / f"s{i}") for i in range(3)]

        with patch.object(acq_mod, "formula_to_npy", return_value=fake_dirs):
            result = composition_pool(formulas, poscar="fake.vasp", output_dir=out_dir)

        assert result == fake_dirs

    def test_output_dir_created(self, tmp_path):
        formulas = ["Fe1"]
        out_dir = str(tmp_path / "new_dir" / "nested")
        assert not os.path.exists(out_dir)

        with patch.object(acq_mod, "formula_to_npy", return_value=[]):
            composition_pool(formulas, poscar="fake.vasp", output_dir=out_dir)

        assert os.path.isdir(out_dir)


# ---------------------------------------------------------------------------
# 5) Public API via dpa_adapt package
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_surrogate_importable_from_dpa_adapt(self):
        from dpa_adapt import Surrogate as S

        assert S is acq_mod.Surrogate

    def test_acquire_importable_from_dpa_adapt(self):
        # The module is named ``acquisition`` so there is no submodule
        # shadowing the ``acquire`` function — the public import resolves
        # to the callable.
        from dpa_adapt import acquire as acq_fn

        assert acq_fn is acq_mod.acquire
        assert callable(acq_fn)

    def test_composition_pool_importable_from_dpa_adapt(self):
        from dpa_adapt import composition_pool as cp

        assert cp is acq_mod.composition_pool

    def test_public_acquire_is_callable_even_if_surrogate_imported_first(self):
        # Regression: locks the public API against import-order fragility.
        # Importing Surrogate triggers the lazy load of the acquisition
        # submodule; this must NOT shadow dpa_adapt.acquire (the function).
        # Run in a clean subprocess so import state is pristine.
        code = (
            "import dpa_adapt; from dpa_adapt import Surrogate; "
            "assert callable(dpa_adapt.acquire), type(dpa_adapt.acquire)"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
