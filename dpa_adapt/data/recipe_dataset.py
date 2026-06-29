# SPDX-License-Identifier: LGPL-3.0-or-later
"""RecipeDataset — polymer recipe-level dataloader for DPA descriptor extraction.

Each recipe in ``recipe.json`` encodes a polymer using the **groups schema**:

.. code-block:: json

    {
      "recipe_000": {
        "label": 32.1,
        "groups": {
          "end_groups": [
            {"path": "systems/monomer_abc", "weight": 0.05},
            {"path": "systems/monomer_def", "weight": 0.05}
          ],
          "repeating": [
            {"path": "systems/monomer_ghi", "weight": 0.8},
            {"path": "systems/monomer_jkl", "weight": 0.2}
          ]
        }
      }
    }

Aggregation reproduces the reference ``create_pfp`` semantics:

    ``concat([end_vec, rep_vec])``   →   length 2 × descriptor_dim

where each ``group_vec = Σ wᵢ · embed(pathᵢ)`` is accumulated over the
components in that group.  Groups are concatenated in the order they appear
in the ``groups`` dict (insertion order, Python ≥ 3.7).  Components with
``weight == 0.0`` are skipped.

Usage::

    dataset = RecipeDataset("polymer_recipe_dataset/", pretrained="DPA-3.1-3M")
    X = dataset.get_embeddings()   # (n_recipes, 2*feat_dim)
    y = dataset.get_labels()       # (n_recipes,)

The ``DPAFineTuner.fit()`` interface auto-detects a recipe directory and
routes to the recipe-level descriptor path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger("dpa_adapt.data.recipe_dataset")


class RecipeDataset:
    """Recipe-level dataset for polymer property prediction.

    Reads ``recipe.json`` from *recipe_dir*; each top-level key is one
    recipe (one training sample).  The recipe embedding is computed as
    ``concat([group_vec_0, group_vec_1, ...])``, where each
    ``group_vec = Σ wᵢ eᵢ`` is the weighted sum over components in that
    group.  For the standard two-group schema (``end_groups``, ``repeating``)
    this gives ``concat([end_vec, rep_vec])``.

    Parameters
    ----------
    recipe_dir : str | Path
        Directory containing ``recipe.json``.  The ``groups[*][path]``
        values inside the JSON are resolved relative to ``recipe_dir``.
    pretrained : str
        Path to the pretrained DPA checkpoint (.pt), or a built-in model
        name (e.g. ``"DPA-3.1-3M"``).
    model_branch : str, optional
        Multi-task branch for descriptor extraction.
    pooling : str
        Descriptor pooling strategy: ``"mean"`` (default), ``"sum"``,
        ``"mean+std"``, or ``"mean+std+max+min"``.
    cache : bool
        Cache per-system descriptors to disk (default True).
    """

    def __init__(
        self,
        recipe_dir: str | Path,
        pretrained: str,
        *,
        model_branch: str | None = None,
        pooling: str = "mean",
        cache: bool = True,
    ) -> None:
        self.recipe_dir = Path(recipe_dir).resolve()
        self.pretrained = pretrained
        self.model_branch = model_branch
        self.pooling = pooling
        self.cache = cache

        recipe_path = self.recipe_dir / "recipe.json"
        if not recipe_path.is_file():
            raise FileNotFoundError(
                f"recipe.json not found in {recipe_dir}. "
                "Run polymer_fingerprint_to_npy() or assemble_recipe_json() first."
            )
        with recipe_path.open(encoding="utf-8") as fh:
            self._recipes: dict[str, Any] = json.load(fh)

        self._recipe_keys: list[str] = sorted(self._recipes.keys())
        self._embeddings: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of recipes (= training samples)."""
        return len(self._recipe_keys)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, float]:
        """Return ``(recipe_embedding, label)`` for recipe at *idx*."""
        key = self._recipe_keys[idx]
        recipe = self._recipes[key]
        emb = self._aggregate_recipe(recipe)
        label = float(recipe["label"])
        return emb, label

    # ------------------------------------------------------------------
    # Aggregation (groups schema)
    # ------------------------------------------------------------------

    def _embed_system(self, sys_path: Path) -> np.ndarray:
        """Extract and pool the DPA descriptor for one system directory.

        Returns shape ``(feat_dim,)``.
        """
        from dpa_adapt.finetuner import load_or_extract  # noqa: PLC0415
        from dpa_adapt.data.loader import load_data  # noqa: PLC0415

        systems = load_data(str(sys_path))
        descs = load_or_extract(
            systems=systems,
            pretrained=self.pretrained,
            model_branch=self.model_branch,
            pooling=self.pooling,
            cache=self.cache,
        )
        # descs: (n_frames, feat_dim) — pool to (feat_dim,)
        return descs.mean(axis=0)

    def _aggregate_recipe(self, recipe: dict) -> np.ndarray:
        """Compute ``concat([group_vec_i, ...])`` for a single recipe dict.

        For each group in ``recipe["groups"]`` (in insertion order):

            ``group_vec = Σ wᵢ · embed(pathᵢ)``

        Empty groups (all weights zero) contribute a zero-vector of length
        equal to the first non-trivial group's embedding dimension.

        Returns
        -------
        np.ndarray
            Shape ``(n_groups × feat_dim,)``.

        Raises
        ------
        ValueError
            If no non-zero-weight components exist in any group.
        """
        groups: dict[str, list[dict]] = recipe.get("groups", {})

        group_vecs: list[np.ndarray | None] = []
        for _group_name, components in groups.items():
            accum: np.ndarray | float = 0.0
            for comp in components:
                w = float(comp["weight"])
                if w == 0.0:
                    continue
                sys_path = self.recipe_dir / comp["path"]
                emb = self._embed_system(sys_path)
                accum = accum + w * emb  # type: ignore[operator]

            group_vecs.append(accum if isinstance(accum, np.ndarray) else None)

        # Determine the embedding dimension from the first non-None group.
        dim = next((v.shape[0] for v in group_vecs if v is not None), None)
        if dim is None:
            raise ValueError(
                f"Recipe has no non-zero-weight components. "
                f"Check the recipe dict: {recipe}"
            )

        filled = [
            (np.zeros(dim, dtype=np.float64) if v is None else v.astype(np.float64))
            for v in group_vecs
        ]
        return np.concatenate(filled)

    # ------------------------------------------------------------------
    # Batch accessors (cached)
    # ------------------------------------------------------------------

    def get_embeddings(self) -> np.ndarray:
        """Extract all recipe embeddings and return them as a matrix.

        Cached after the first call — descriptor extraction runs only once.

        Returns
        -------
        np.ndarray
            Shape ``(n_recipes, n_groups × feat_dim)``.
        """
        if self._embeddings is None:
            _LOG.info(
                "RecipeDataset: extracting embeddings for %d recipes…", len(self)
            )
            embs = [self._aggregate_recipe(self._recipes[k]) for k in self._recipe_keys]
            self._embeddings = np.array(embs)
            _LOG.info("RecipeDataset: embeddings shape %s", self._embeddings.shape)
        return self._embeddings

    def get_labels(self) -> np.ndarray:
        """Return all property labels as a 1-D array.

        Cached after the first call.

        Returns
        -------
        np.ndarray
            Shape ``(n_recipes,)``.
        """
        if self._labels is None:
            self._labels = np.array(
                [float(self._recipes[k]["label"]) for k in self._recipe_keys],
                dtype=np.float64,
            )
        return self._labels

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RecipeDataset(n_recipes={len(self)}, "
            f"recipe_dir={str(self.recipe_dir)!r}, "
            f"pretrained={self.pretrained!r}, "
            f"pooling={self.pooling!r})"
        )
