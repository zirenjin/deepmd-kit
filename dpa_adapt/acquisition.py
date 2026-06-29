# SPDX-License-Identifier: LGPL-3.0-or-later
"""Acquisition layer for active learning and Bayesian optimisation on DPA descriptors.

Provides a probabilistic surrogate (mean + std via BayesianRidge) and a set of
acquisition policies (UCB, EI, PI, uncertainty sampling, diversity-based) that
operate over a candidate pool of deepmd/npy system directories.  The heavy
descriptor extraction is delegated to :func:`~dpa_adapt.finetuner.load_or_extract`
and cached transparently.

Typical usage::

    from dpa_adapt import Surrogate, acquire

    surrogate = Surrogate(pretrained="model.pt")
    surrogate.fit(train_dirs, labels=y_train)

    result = acquire(surrogate, candidate_dirs, strategy="ucb", batch_size=5)
    next_batch = result.selected
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from dpa_adapt.data.formula import formula_to_npy
from dpa_adapt.data.loader import load_data
from dpa_adapt.finetuner import _load_labels, load_or_extract
from dpa_adapt.utils.dotdict import DotDict

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Pure-numpy acquisition helpers
# ---------------------------------------------------------------------------


def _ucb(
    mu: np.ndarray,
    sigma: np.ndarray,
    beta: float,
    objective: str,
) -> np.ndarray:
    """Upper-confidence-bound acquisition score.

    Returns a score where *higher is always better* (i.e. select top-k).

    Parameters
    ----------
    mu : np.ndarray
        Predicted mean, shape (n,).
    sigma : np.ndarray
        Predicted std, shape (n,).
    beta : float
        Exploration–exploitation trade-off coefficient.
    objective : str
        ``"max"`` or ``"min"``.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if objective == "max":
        return mu + beta * sigma
    # "min": negate so higher score still means "pick first"
    return -(mu - beta * sigma)


def _ei(
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    xi: float,
    objective: str,
) -> np.ndarray:
    """Expected Improvement acquisition score.

    Parameters
    ----------
    mu : np.ndarray
        Predicted mean, shape (n,).
    sigma : np.ndarray
        Predicted std, shape (n,).
    best : float
        Current best observed value.
    xi : float
        Jitter / exploration bonus.
    objective : str
        ``"max"`` or ``"min"``.

    Returns
    -------
    np.ndarray
        EI values, shape (n,); always >= 0; zero where sigma == 0.
    """
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if objective == "max":
        imp = mu - best - xi
    else:
        imp = best - mu - xi

    mask = sigma > 0
    safe_sigma = np.where(mask, sigma, 1.0)
    z = imp / safe_sigma

    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(mask, ei, 0.0)
    return np.maximum(ei, 0.0)


def _pi(
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    xi: float,
    objective: str,
) -> np.ndarray:
    """Probability of Improvement acquisition score.

    Parameters
    ----------
    mu : np.ndarray
        Predicted mean, shape (n,).
    sigma : np.ndarray
        Predicted std, shape (n,).
    best : float
        Current best observed value.
    xi : float
        Jitter / exploration bonus.
    objective : str
        ``"max"`` or ``"min"``.

    Returns
    -------
    np.ndarray
        PI values, shape (n,); zero where sigma == 0.
    """
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if objective == "max":
        imp = mu - best - xi
    else:
        imp = best - mu - xi

    mask = sigma > 0
    safe_sigma = np.where(mask, sigma, 1.0)
    z = imp / safe_sigma

    pi = norm.cdf(z)
    return np.where(mask, pi, 0.0)


def _farthest_point(
    X_pool: np.ndarray,
    X_seen: np.ndarray,
    k: int,
    weights: np.ndarray | None = None,
) -> list[int]:
    """Greedy k-centre / farthest-point selection from ``X_pool``.

    Selects *k* row indices from *X_pool* that are maximally spread in
    Euclidean space relative to *X_seen* (the already-covered points).

    Parameters
    ----------
    X_pool : np.ndarray, shape (n_pool, d)
        Candidate feature matrix.
    X_seen : np.ndarray, shape (n_seen, d)
        Already-selected / training feature matrix.  May be empty
        (shape ``(0, d)`` or an empty array).
    k : int
        Number of points to select.
    weights : np.ndarray or None, shape (n_pool,)
        Per-pool-point weights (e.g. predicted sigma).  When provided,
        each step selects ``argmax(min_dist * weights)`` to jointly
        favour far and uncertain points.  Exception: the very first pick
        when ``X_seen`` is empty uses ``argmax(weights)`` (distances are
        undefined).

    Returns
    -------
    list[int]
        Selected indices from ``X_pool``, in selection order (length k).
    """
    X_pool = np.asarray(X_pool, dtype=float)
    n = len(X_pool)

    seen_nonempty = X_seen is not None and len(X_seen) > 0

    if seen_nonempty:
        X_seen_arr = np.asarray(X_seen, dtype=float)
        diffs = X_pool[:, np.newaxis, :] - X_seen_arr[np.newaxis, :, :]
        sq_dists = np.sum(diffs**2, axis=2)
        min_dist = np.sqrt(np.min(sq_dists, axis=1))
    else:
        min_dist = np.full(n, np.inf)

    available = np.ones(n, dtype=bool)
    selected: list[int] = []

    for step in range(k):
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if step == 0 and not seen_nonempty:
                # Cannot use distances yet; fall back to pure uncertainty.
                raw = np.where(available, w, -np.inf)
            else:
                raw = np.where(available, min_dist * w, -np.inf)
        else:
            raw = np.where(available, min_dist, -np.inf)

        idx = int(np.argmax(raw))
        selected.append(idx)
        available[idx] = False

        # Update min distances with the newly selected point.
        new_dists = np.sqrt(np.sum((X_pool - X_pool[idx]) ** 2, axis=1))
        min_dist = np.minimum(min_dist, new_dists)

    return selected


# ---------------------------------------------------------------------------
# Pool featurisation  (candidate granularity = one row per system)
# ---------------------------------------------------------------------------


def _featurize_pool(
    pool: list[str] | str,
    pretrained: str,
    model_branch: str | None,
    pooling: str,
    cache: bool,
) -> tuple[np.ndarray, list[str]]:
    """Featurise *pool* systems into per-system descriptor vectors.

    Parameters
    ----------
    pool : list[str] or str
        deepmd/npy system directories (one candidate each).
    pretrained : str
        Path to the pretrained DPA checkpoint.
    model_branch : str or None
        Branch name for multi-task checkpoints.
    pooling : str
        Pooling strategy passed to ``load_or_extract``.
    cache : bool
        Whether to use the descriptor cache.

    Returns
    -------
    X : np.ndarray, shape (n_systems, feat_dim)
        Per-system mean-pooled descriptor vectors.
    sys_ids : list[str]
        System directory paths in the same order as rows of ``X``.
    """
    if isinstance(pool, str):
        pool = [pool]

    systems = load_data(pool)
    frame_counts = [s.get_nframes() for s in systems]

    feats = load_or_extract(systems, pretrained, model_branch, pooling, cache)

    X_rows: list[np.ndarray] = []
    start = 0
    for n_frames in frame_counts:
        block = feats[start : start + n_frames]
        X_rows.append(block.mean(axis=0))
        start += n_frames

    X = np.stack(X_rows, axis=0)
    return X, list(pool)


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------


class Surrogate:
    """Probabilistic surrogate model (mean + std) on DPA descriptor features.

    Fits one :class:`sklearn.linear_model.BayesianRidge` per property
    target on standardised per-frame DPA descriptors.  Call :meth:`fit` on
    a labelled dataset, then :meth:`predict` or :func:`acquire` on a pool of
    candidate systems.

    Parameters
    ----------
    pretrained : str
        Path to the pretrained DPA checkpoint.
    model_branch : str or None
        Branch name for multi-task checkpoints.
    pooling : str
        Descriptor pooling strategy; default ``"mean"``.
    kind : str
        Surrogate type: ``"bayesian_ridge"`` (default) or ``"gp"``.
    seed : int
        Random seed (reserved for future use / GP).
    cache : bool
        Whether to use the descriptor cache when calling ``load_or_extract``.
    """

    _VALID_KINDS = {"bayesian_ridge", "gp"}

    def __init__(
        self,
        pretrained: str,
        model_branch: str | None = None,
        pooling: str = "mean",
        kind: str = "bayesian_ridge",
        seed: int = 42,
        cache: bool = True,
    ) -> None:
        if kind not in self._VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(self._VALID_KINDS)}, got {kind!r}"
            )
        self.pretrained = pretrained
        self.model_branch = model_branch
        self.pooling = pooling
        self.kind = kind
        self.seed = seed
        self.cache = cache

        self._scaler = None
        self._models: list = []
        self._y_train: np.ndarray | None = None
        self._X_train_scaled: np.ndarray | None = None
        self._n_targets: int = 0
        self._fitted: bool = False

    def fit(
        self,
        data: str | list[str],
        target_key: str = "property",
        labels: np.ndarray | None = None,
    ) -> "Surrogate":
        """Fit the surrogate on per-frame descriptors.

        Parameters
        ----------
        data : str or list[str]
            Paths to deepmd/npy system directories.
        target_key : str
            Label key used when *labels* is ``None``.
        labels : np.ndarray, optional
            Pre-computed labels array of shape ``(n_frames,)`` or
            ``(n_frames, k)``.  When provided, *target_key* is ignored.

        Returns
        -------
        self
        """
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler

        if self.kind == "gp":
            raise NotImplementedError("kind='gp' not yet supported")

        systems = load_data(data)
        X = load_or_extract(
            systems, self.pretrained, self.model_branch, self.pooling, self.cache
        )

        if labels is not None:
            y = np.asarray(labels, dtype=float)
        else:
            y = _load_labels(systems, target_key).astype(float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        k = y.shape[1]
        self._n_targets = k
        self._y_train = y

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._X_train_scaled = X_scaled

        self._models = []
        for col in range(k):
            model = BayesianRidge()
            model.fit(X_scaled, y[:, col])
            self._models.append(model)

        self._fitted = True
        return self

    def _score_pool(
        self,
        pool: list[str] | str,
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        """Featurise *pool* and return surrogate statistics.

        Returns
        -------
        sys_ids : list[str]
        Xs : np.ndarray, shape (n, feat_dim)  — scaled features
        mu : np.ndarray
            shape (n,) when k==1, else (n, k).
        sigma : np.ndarray
            shape (n,) — per-candidate std; mean over targets when k>1.
        """
        if not self._fitted:
            raise RuntimeError(
                "Surrogate.predict() / _score_pool() called before fit()."
            )

        X, sys_ids = _featurize_pool(
            pool, self.pretrained, self.model_branch, self.pooling, self.cache
        )
        Xs = self._scaler.transform(X)

        means_list: list[np.ndarray] = []
        stds_list: list[np.ndarray] = []
        for model in self._models:
            m, s = model.predict(Xs, return_std=True)
            means_list.append(m)
            stds_list.append(s)

        mean_arr = np.stack(means_list, axis=1)  # (n, k)
        std_arr = np.stack(stds_list, axis=1)    # (n, k)

        if self._n_targets == 1:
            mu = mean_arr[:, 0]
            sigma = std_arr[:, 0]
        else:
            mu = mean_arr                     # (n, k)
            sigma = std_arr.mean(axis=1)      # (n,) — mean over targets

        return sys_ids, Xs, mu, sigma

    def predict(self, pool: list[str] | str) -> DotDict:
        """Predict mean and uncertainty for each system in *pool*.

        Parameters
        ----------
        pool : list[str] or str
            Candidate system directories.

        Returns
        -------
        DotDict with keys:

        - ``systems`` : list[str]
        - ``mean`` : np.ndarray, shape (n,) or (n, k)
        - ``std`` : np.ndarray, shape (n,) or (n, k)
        - ``features`` : np.ndarray, shape (n, feat_dim)  — scaled
        """
        sys_ids, Xs, mu, sigma = self._score_pool(pool)

        # For predict, return per-target std when k>1.
        if self._n_targets > 1:
            # _score_pool already returns mean std; re-derive per-target std.
            means_list: list[np.ndarray] = []
            stds_list: list[np.ndarray] = []
            for model in self._models:
                m, s = model.predict(Xs, return_std=True)
                means_list.append(m)
                stds_list.append(s)
            mean_arr = np.stack(means_list, axis=1)
            std_arr = np.stack(stds_list, axis=1)
        else:
            mean_arr = mu.reshape(-1, 1)
            std_arr = sigma.reshape(-1, 1)

        if self._n_targets == 1:
            mean_out = mean_arr[:, 0]
            std_out = std_arr[:, 0]
        else:
            mean_out = mean_arr
            std_out = std_arr

        return DotDict(systems=sys_ids, mean=mean_out, std=std_out, features=Xs)


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------


def acquire(
    surrogate: Surrogate,
    pool: list[str] | str,
    *,
    strategy: str = "ucb",
    batch_size: int = 1,
    beta: float = 2.0,
    xi: float = 0.0,
    objective: str = "max",
    return_scores: bool = True,
) -> DotDict:
    """Select the most informative candidates from *pool* for labelling.

    Parameters
    ----------
    surrogate : Surrogate
        A fitted :class:`Surrogate` instance.
    pool : list[str] or str
        Candidate deepmd/npy system directories.
    strategy : str
        Acquisition strategy: ``"ucb"``, ``"ei"``, ``"pi"``,
        ``"uncertainty"``, or ``"diverse"``.
    batch_size : int
        Number of candidates to select.
    beta : float
        UCB exploration coefficient.
    xi : float
        EI / PI jitter.
    objective : str
        ``"max"`` (maximise) or ``"min"`` (minimise).
    return_scores : bool
        Whether to include acquisition scores in the return value.

    Returns
    -------
    DotDict with keys:

    - ``selected`` : list[str]  — system paths of chosen candidates
    - ``indices`` : list[int]   — their indices into *pool*
    - ``scores`` : np.ndarray or None — acquisition scores of selected
    - ``mean`` : np.ndarray     — surrogate mean for all pool points
    - ``std`` : np.ndarray      — surrogate std for all pool points
    - ``strategy`` : str
    - ``objective`` : str
    """
    _valid_objectives = {"max", "min"}
    _valid_strategies = {"uncertainty", "diverse", "ucb", "ei", "pi"}

    if objective not in _valid_objectives:
        raise ValueError(
            f"objective must be one of {sorted(_valid_objectives)}, got {objective!r}"
        )
    if strategy not in _valid_strategies:
        raise ValueError(
            f"strategy must be one of {sorted(_valid_strategies)}, got {strategy!r}"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    sys_ids, Xs, mu, sigma = surrogate._score_pool(pool)

    # BO strategies require a single-target surrogate.
    bo_strategies = {"ucb", "ei", "pi"}
    if strategy in bo_strategies and surrogate._n_targets != 1:
        raise ValueError(
            f"BO strategy {strategy!r} requires a single-target surrogate "
            f"(n_targets=1), but this surrogate has {surrogate._n_targets} targets. "
            "Use strategy='uncertainty' or 'diverse' for multi-target surrogates."
        )

    # Compute acquisition scores.
    if strategy == "uncertainty":
        scores = sigma
    elif strategy == "ucb":
        scores = _ucb(mu, sigma, beta, objective)
    elif strategy == "ei":
        best = (
            float(surrogate._y_train[:, 0].max())
            if objective == "max"
            else float(surrogate._y_train[:, 0].min())
        )
        scores = _ei(mu, sigma, best, xi, objective)
    elif strategy == "pi":
        best = (
            float(surrogate._y_train[:, 0].max())
            if objective == "max"
            else float(surrogate._y_train[:, 0].min())
        )
        scores = _pi(mu, sigma, best, xi, objective)
    else:  # "diverse"
        scores = sigma  # reported score = sigma at selected picks

    if strategy in {"uncertainty", "ucb", "ei", "pi"}:
        selected = list(np.argsort(scores)[::-1][:batch_size])
    else:  # "diverse"
        selected = _farthest_point(
            Xs, surrogate._X_train_scaled, batch_size, weights=sigma
        )

    selected_scores = scores[selected] if return_scores else None

    return DotDict(
        selected=[sys_ids[i] for i in selected],
        indices=selected,
        scores=selected_scores,
        mean=mu,
        std=sigma,
        strategy=strategy,
        objective=objective,
    )


# ---------------------------------------------------------------------------
# composition_pool
# ---------------------------------------------------------------------------


def composition_pool(
    formulas: list[str],
    poscar: str,
    output_dir: str,
    *,
    base_element: str | None = None,
    sets: int = 1,
    seed: int = 42,
    property_name: str = "Property",
) -> list[str]:
    """Generate a pool of candidate structures for composition-space BO.

    Materialises *formulas* as deepmd/npy systems by applying random atomic
    substitution to a fixed POSCAR prototype.  Labels are placeholders
    (``0.0``) because real values must be supplied by the oracle after
    running DFT or experiments.

    This is BO design-space level (b): compositions materialised on a fixed
    POSCAR prototype; half-automatic loop —
    ``propose → user runs oracle → attach_labels → Surrogate.fit``.

    Parameters
    ----------
    formulas : list[str]
        Composition strings (candidate design points, no labels yet).
    poscar : str
        Path to template POSCAR (VASP format).
    output_dir : str
        Directory to write the candidate CSV and deepmd/npy systems.
    base_element : str, optional
        Host element for random substitution.  Auto-inferred from POSCAR.
    sets : int
        Number of random realisations per formula.
    seed : int
        Random seed for reproducibility.
    property_name : str
        Label key written into each generated system.

    Returns
    -------
    list[str]
        Resolved paths of the generated deepmd/npy system directories.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "candidates.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fh.write("formula,Property\n")
        for formula in formulas:
            fh.write(f"{formula},0.0\n")

    return formula_to_npy(
        csv_path,
        output_dir,
        poscar,
        formula_col="formula",
        property_col="Property",
        property_name=property_name,
        base_element=base_element,
        sets=sets,
        seed=seed,
    )
