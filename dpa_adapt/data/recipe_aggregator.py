# SPDX-License-Identifier: LGPL-3.0-or-later
"""RecipeAggregator — standalone weighted-descriptor aggregation for polymer recipes.

This module provides a **standalone** aggregation step that reproduces the
``create_pfp`` semantics from the reference ``dpa_fingerprints.py`` script.

**Why descriptor-level (not deep-forward-level) aggregation?**

The ``finetune`` and ``mft`` strategies in dpa_adapt delegate training to
``dp --pt train`` via ``subprocess.run()`` — there is no in-process PyTorch
forward/loss in dpa_adapt.  Injecting recipe-level weighted aggregation
into the deep forward/loss of ``dp --pt train`` would require modifying
deepmd-kit core, which is prohibited.  Therefore, recipe aggregation is
realised at the descriptor level:

  1. Extract per-component DPA descriptor embeddings via the frozen backbone
     (using :func:`dpa_adapt.finetuner.load_or_extract`).
  2. Aggregate using the groups schema: ``concat([group_vec_i, ...])``
     where each ``group_vec = Σ wᵢ eᵢ`` within each group.
  3. Train an sklearn head on the recipe embeddings (handled by
     :class:`DPAFineTuner._fit_recipe`).

**Aggregation schema**

Recipes use the **groups schema** (see :mod:`dpa_adapt.data.polymer`).
The aggregator processes groups in insertion order and concatenates their
weighted-sum vectors to reproduce ``create_pfp(end_units, repeating_units, Mn)``::

    recipe["groups"]["end_groups"] → end_vec  = Σ end_share · embed(g)
    recipe["groups"]["repeating"]  → rep_vec  = Σ molp_i   · embed(u_i)
    output                         = concat([end_vec, rep_vec])

Components with ``weight == 0.0`` are skipped.

Usage (single recipe)::

    agg = RecipeAggregator(pretrained="path/to/ckpt.pt")
    recipe = {
        "label": 85.5,
        "groups": {
            "end_groups": [
                {"path": "systems/monomer_abc", "weight": 0.05},
                {"path": "systems/monomer_def", "weight": 0.05},
            ],
            "repeating": [
                {"path": "systems/monomer_ghi", "weight": 0.8},
                {"path": "systems/monomer_jkl", "weight": 0.2},
            ],
        },
    }
    embedding = agg.aggregate(recipe, recipe_dir="polymer_recipe_dataset/")
    # shape (2 * feat_dim,)

Usage (batch)::

    import json
    all_recipes = json.load(open("polymer_recipe_dataset/recipe.json"))
    X = agg.transform(
        list(all_recipes.values()),
        recipe_dir="polymer_recipe_dataset/",
    )
    # shape (n_recipes, 2 * feat_dim)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

_LOG = logging.getLogger("dpa_adapt.data.recipe_aggregator")


class RecipeAggregator:
    """Standalone recipe-level descriptor aggregation (groups schema).

    Computes ``concat([group_vec_i, ...])`` for one or more polymer recipes,
    where each ``group_vec = Σ wᵢ eᵢ`` within a group.  For the standard
    two-group ``(end_groups, repeating)`` schema this reproduces
    ``create_pfp`` as ``concat([end_vec, rep_vec])``.

    The DPA backbone is used as a **frozen** feature extractor — no backbone
    weights are updated.  Per-system descriptors are shared with the global
    descriptor cache used by :class:`DPAFineTuner`.

    Parameters
    ----------
    pretrained : str
        Path to the pretrained DPA checkpoint (.pt), or a built-in name
        (e.g. ``"DPA-3.1-3M"``).
    model_branch : str, optional
        Multi-task branch for descriptor extraction.
    pooling : str
        Descriptor pooling strategy: ``"mean"`` (default) or ``"sum"``.
    cache : bool
        Cache per-system descriptors to disk (default True).
    """

    def __init__(
        self,
        pretrained: str,
        *,
        model_branch: str | None = None,
        pooling: str = "mean",
        cache: bool = True,
    ) -> None:
        self.pretrained = pretrained
        self.model_branch = model_branch
        self.pooling = pooling
        self.cache = cache

    # ------------------------------------------------------------------
    # Core aggregation (groups schema)
    # ------------------------------------------------------------------

    def aggregate(
        self,
        recipe: dict,
        recipe_dir: str | Path | None = None,
    ) -> np.ndarray:
        """Aggregate per-component DPA descriptors for a single recipe.

        Processes ``recipe["groups"]`` in insertion order.  For each group:

            ``group_vec = Σ wᵢ · embed(pathᵢ)``  (zero-weight components skipped)

        The group vectors are concatenated in insertion order.

        Parameters
        ----------
        recipe : dict
            A single recipe dict with a ``"groups"`` key.
        recipe_dir : str | Path, optional
            Directory to resolve relative component paths from.  Required
            when ``path`` values in the recipe are relative.

        Returns
        -------
        np.ndarray
            Shape ``(n_groups × feat_dim,)``.

        Raises
        ------
        ValueError
            If the recipe has no non-zero-weight components.
        """
        from dpa_adapt.finetuner import load_or_extract  # noqa: PLC0415
        from dpa_adapt.data.loader import load_data  # noqa: PLC0415

        base = Path(recipe_dir) if recipe_dir is not None else Path(".")
        groups: dict[str, list[dict]] = recipe.get("groups", {})

        group_vecs: list[np.ndarray | None] = []
        for _group_name, components in groups.items():
            accum: np.ndarray | float = 0.0
            for comp in components:
                weight = float(comp["weight"])
                if weight == 0.0:
                    continue

                path = comp["path"]
                sys_path = base / path if not Path(path).is_absolute() else Path(path)

                systems = load_data(str(sys_path))
                descs = load_or_extract(
                    systems=systems,
                    pretrained=self.pretrained,
                    model_branch=self.model_branch,
                    pooling=self.pooling,
                    cache=self.cache,
                )
                # descs: (n_frames, feat_dim) — monomer systems have 1 frame.
                comp_emb = descs.mean(axis=0)  # (feat_dim,)
                accum = accum + weight * comp_emb  # type: ignore[operator]

            group_vecs.append(accum if isinstance(accum, np.ndarray) else None)

        # Determine the embedding dimension from the first non-None group.
        dim = next((v.shape[0] for v in group_vecs if v is not None), None)
        if dim is None:
            raise ValueError(
                "Recipe has no non-zero-weight components with valid system "
                "directories. Check the recipe dict and recipe_dir."
            )

        filled = [
            (np.zeros(dim, dtype=np.float64) if v is None else v.astype(np.float64))
            for v in group_vecs
        ]
        return np.concatenate(filled)

    def transform(
        self,
        recipes: list[dict],
        recipe_dir: str | Path | None = None,
    ) -> np.ndarray:
        """Aggregate descriptors for a batch of recipes.

        Parameters
        ----------
        recipes : list[dict]
            List of recipe dicts (see :meth:`aggregate`).
        recipe_dir : str | Path, optional
            Base directory for resolving relative component paths.

        Returns
        -------
        np.ndarray
            Shape ``(n_recipes, n_groups × feat_dim)``.
        """
        if not recipes:
            raise ValueError("recipes list is empty.")
        embs = [self.aggregate(r, recipe_dir=recipe_dir) for r in recipes]
        return np.array(embs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RecipeAggregator(pretrained={self.pretrained!r}, "
            f"pooling={self.pooling!r}, "
            f"cache={self.cache!r})"
        )
