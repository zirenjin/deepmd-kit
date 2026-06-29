# SPDX-License-Identifier: LGPL-3.0-or-later
"""Polymer recipe → deepmd/npy + recipe.json conversion.

Three independently-callable pipeline steps:

  **(a) Conformer generation only** — :func:`generate_monomer_systems`
      Reads a list of SMILES strings, generates 3-D conformers via RDKit
      (radical-capping + ETKDGv3 + MMFF), and writes one deepmd/npy system per
      *unique* SMILES.

  **(b) Recipe assembly only** — :func:`assemble_recipe_json`
      Reads the European-format CSV and builds ``recipe.json`` from
      already-existing system directories.

  **(c) Both steps in one call** — :func:`polymer_fingerprint_to_npy`
      Also reachable via ``convert(csv, out, fmt="polymer_fingerprint")``.

Conformer generation
--------------------
Polymer unit SMILES carry attachment points as *radical electrons* (e.g.
``[CH2][CH](...)``).  The standard ``Chem.AddHs`` over-fills these valences.
We port the reference ``cap_open_valences`` / ``smiles_to_mol3d`` logic from
``dpa_fingerprints.py``:

1. For each atom convert ``GetNumRadicalElectrons()`` → that many explicit H.
2. ``SanitizeMol(mol)`` to resolve valences.
3. ``Chem.AddHs(mol)`` to add any remaining implicit H.
4. ETKDGv3 with fixed ``randomSeed=0xC0FFEE``; fallback ``useRandomCoords``.
5. ``MMFFOptimizeMolecule`` (UFF if MMFF not parametrised).

Aggregation (``create_pfp`` semantics)
---------------------------------------
The fingerprint is **not** a flat ``Σ wᵢ eᵢ``.  It concatenates two group
vectors to reproduce the reference ``create_pfp`` function:

    ``concat([end_vec, rep_vec])``  → length 2 × descriptor_dim

where:

- ``rep_vec = Σ molp_i · embed(unit_i)``   (mole fractions from the CSV)
- ``end_vec = Σ end_share · embed(g)``       (both start/end groups get
  the same scalar ``end_share``)
- ``end_share = 1 / total_moles``            (see :func:`_calc_ends_share`)

``recipe.json`` uses a **groups schema** to encode this structure:

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

The :class:`RecipeAggregator` processes each group with ``Σ wᵢ eᵢ`` and
then concatenates the group vectors in the order they appear in the
``groups`` dict (``end_groups`` first, then ``repeating``).

Output layout::

    output_dir/
      systems/
        monomer_{sha256[:16]}/      # one dir per unique canonical SMILES
          type_map.raw
          type.raw
          set.000/
            coord.npy   shape (1, n_atoms * 3)  float64
            box.npy     shape (1, 9)             float64
      recipe.json

Deviation notes
---------------
- Conformer generation was **not** reused from ``smiles_to_3d_coords``
  (``dpa_adapt.data.smiles``) because that function calls ``Chem.AddHs``
  directly and does not cap radical electrons.  The reference
  ``cap_open_valences`` + ``smiles_to_mol3d`` logic is ported here instead.
- The fixed seed ``0xC0FFEE`` matches the reference script.
- The flat ``components`` recipe schema (original design-doc plan) is
  replaced by the **groups schema** to reproduce ``create_pfp`` semantics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_LOG = logging.getLogger("dpa_adapt")

# Column names for the European polymer CSV.
_UNIT_LETTERS = "ABCDE"
_MONOMER_SMILES_COLS = [f"SMILES_repeating_unit{s}" for s in _UNIT_LETTERS]
_MONOMER_PCT_COLS = [f"molpercent_repeating_unit{s}" for s in _UNIT_LETTERS]
_END_GROUP_COLS = ["SMILES_start_group", "SMILES_end_group"]

# Fixed ETKDGv3 seed from the reference script.
_ETKDG_SEED = 0xC0FFEE


# ---------------------------------------------------------------------------
# NaN helper
# ---------------------------------------------------------------------------


def _is_empty(val: object) -> bool:
    """Return True when a cell value is blank (None, NaN, or empty string)."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return str(val).strip() == ""


# ---------------------------------------------------------------------------
# SMILES utilities (deduplication)
# ---------------------------------------------------------------------------


def _canonicalize_smiles(smiles: str) -> str:
    """Return RDKit canonical SMILES; fall back to the input if RDKit fails."""
    try:
        from rdkit import Chem  # noqa: PLC0415

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return str(smiles)


def _smiles_hash(smiles: str) -> str:
    """16-char SHA-256 hex digest of canonical SMILES (for deduplication)."""
    canonical = _canonicalize_smiles(smiles)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Conformer generation — polymer-SMILES-aware
# ---------------------------------------------------------------------------


def _cap_open_valences(mol: object) -> object:
    """Cap every radical electron with an explicit H.

    Polymer unit / end-group SMILES carry attachment points as unfilled
    valences, which RDKit parses as radical electrons.  Standard
    ``Chem.AddHs`` over-fills single-attachment groups; instead, for each
    atom convert ``GetNumRadicalElectrons()`` → that many explicit H.

    Ported from ``dpa_fingerprints.cap_open_valences`` (reference script).
    """
    for atom in mol.GetAtoms():
        nrad = atom.GetNumRadicalElectrons()
        if nrad:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + nrad)
            atom.SetNumRadicalElectrons(0)
    mol.UpdatePropertyCache(strict=False)
    return mol


def polymer_smiles_to_3d(
    smiles: str,
    *,
    cap: bool = True,
    seed: int = _ETKDG_SEED,
) -> tuple[list[str], np.ndarray]:
    """Generate a 3-D conformer for a polymer SMILES.

    Ported from ``dpa_fingerprints.smiles_to_mol3d``.  Radical electrons
    (attachment points) are first converted to explicit H by
    :func:`_cap_open_valences` before ``Chem.AddHs``, preventing
    over-filling of open valences.

    Parameters
    ----------
    smiles : str
        Polymer (repeating-unit or end-group) SMILES.
    cap : bool
        If True (default), apply :func:`_cap_open_valences` before AddHs.
    seed : int
        ETKDGv3 random seed.  Default ``0xC0FFEE`` matches the reference
        script for reproducibility.

    Returns
    -------
    (symbols, coords)
        *symbols* is a list of element symbol strings (length n_atoms
        including explicit H);  *coords* is shape ``(n_atoms, 3)`` float32.

    Raises
    ------
    ValueError
        If the SMILES cannot be parsed or 3-D embedding fails.
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
        from rdkit.Chem import AllChem  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for polymer conformer generation. "
            "Install it with: pip install rdkit"
        ) from exc

    from dpa_adapt.data.smiles import ELEMENT_INDEX  # noqa: PLC0415

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse polymer SMILES: {smiles!r}")

    if cap:
        _cap_open_valences(mol)
        Chem.SanitizeMol(mol)

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"3D embedding failed for polymer SMILES: {smiles!r}")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol)
    else:
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    symbols: list[str] = []
    coords_list: list[list[float]] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ELEMENT_INDEX:
            raise ValueError(
                f"Unknown element {symbol!r} generated from SMILES {smiles!r}"
            )
        pos = conf.GetAtomPosition(atom.GetIdx())
        symbols.append(symbol)
        coords_list.append([pos.x, pos.y, pos.z])

    return symbols, np.asarray(coords_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Molar-weight and end-share helpers (ported from reference script)
# ---------------------------------------------------------------------------


def _molar_wt_from_smiles(smiles: str) -> float:
    """Return the RDKit MolWt of a SMILES string (capped with cap_open_valences).

    Ported from ``dpa_fingerprints.molar_wt_fromSmiles``.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import Descriptors  # noqa: PLC0415

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Cannot compute MolWt: invalid SMILES {smiles!r}")
    _cap_open_valences(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    return Descriptors.MolWt(mol)


def _calc_ends_share(
    end_smiles: list[str],
    rep_units: list[tuple[float, str]],
    mn: float,
) -> float:
    """Compute the mole fraction of each end group.

    Ported from ``dpa_fingerprints.calc_ends_share``:

    .. code-block:: python

        total_weight_minus = Mn - Σ MolWt(end_smiles)
        total_moles = Σ_i [ total_weight_minus * molp_i / MolWt(unit_i) ]
        end_share = 1 / total_moles

    Parameters
    ----------
    end_smiles : list[str]
        SMILES of end groups (start group + end group).
    rep_units : list[tuple[float, str]]
        List of (mole_fraction, smiles) pairs for repeating units.
    mn : float
        Number-average molar mass (Mn) of the polymer.

    Returns
    -------
    float
        Mole fraction of each end group (same for all end groups).
    """
    wt_ends = sum(_molar_wt_from_smiles(s) for s in end_smiles)
    total_weight_minus = mn - wt_ends
    total_moles = 0.0
    for molp, smiles in rep_units:
        rep_mass = total_weight_minus * float(molp)
        rep_moles = rep_mass / _molar_wt_from_smiles(smiles)
        total_moles += rep_moles
    if total_moles <= 0.0:
        raise ValueError(
            f"calc_ends_share: total_moles={total_moles} <= 0 "
            f"(Mn={mn}, end_smiles={end_smiles})"
        )
    return 1.0 / total_moles


# ---------------------------------------------------------------------------
# Low-level deepmd/npy writer
# ---------------------------------------------------------------------------


def _write_monomer_system(
    symbols: list[str],
    coords: np.ndarray,
    out_dir: Path,
) -> None:
    """Write a single-frame deepmd/npy system directly to disk.

    Writes ``type_map.raw``, ``type.raw``, ``set.000/coord.npy``,
    ``set.000/box.npy`` — the files that ``load_data()`` and dpdata expect.
    """
    from dpa_adapt.data.smiles import _build_type_map_from_elements  # noqa: PLC0415

    type_map = _build_type_map_from_elements(set(symbols))
    type_index = {el: i for i, el in enumerate(type_map)}
    atom_types = np.array([type_index[s] for s in symbols], dtype=np.int32)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "type_map.raw").write_text("\n".join(type_map) + "\n", encoding="utf-8")
    (out_dir / "type.raw").write_text(
        "\n".join(str(int(t)) for t in atom_types) + "\n", encoding="utf-8"
    )

    set_dir = out_dir / "set.000"
    set_dir.mkdir(exist_ok=True)

    # coord.npy : shape (1, n_atoms * 3)
    np.save(str(set_dir / "coord.npy"), coords.reshape(1, -1).astype(np.float64))
    # box.npy   : shape (1, 9) — large cubic box for isolated molecules
    box = np.eye(3, dtype=np.float64) * 100.0
    np.save(str(set_dir / "box.npy"), box.reshape(1, 9))


# ---------------------------------------------------------------------------
# Step (a): Conformer generation
# ---------------------------------------------------------------------------


@dataclass
class PolymerConvertResult:
    """Return value of :func:`polymer_fingerprint_to_npy`."""

    recipe_path: Path
    systems_dir: Path
    n_unique_smiles: int
    n_recipes: int
    failed_smiles: list[tuple[str, str]] = field(default_factory=list)


def generate_monomer_systems(
    smiles_list: list[str],
    output_dir: str | Path,
    *,
    seed: int = _ETKDG_SEED,
    overwrite: bool = False,
) -> dict[str, str]:
    """Step (a): Generate deepmd/npy conformer systems for unique polymer SMILES.

    Each unique SMILES is identified by its 16-char SHA-256 hash of its RDKit
    canonical form, so duplicate strings across rows generate only one system.

    Uses :func:`polymer_smiles_to_3d` (radical-capping + ETKDGv3 + MMFF),
    **not** the standard ``smiles_to_3d_coords``, because polymer SMILES carry
    attachment points as radical electrons that must be capped before ``AddHs``.

    Parameters
    ----------
    smiles_list : list[str]
        All SMILES to process (may contain duplicates and empty strings).
    output_dir : str | Path
        Root directory; each system lands in ``output_dir/monomer_{hash}/``.
    seed : int
        ETKDGv3 seed.  Default ``0xC0FFEE`` matches the reference script.
    overwrite : bool
        Re-generate a system even if its directory already exists.

    Returns
    -------
    dict
        Mapping ``canonical_smiles → absolute_system_path`` for successfully
        generated SMILES.  Failed SMILES are omitted (logged as warnings).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Deduplicate while preserving first-seen order.
    seen: dict[str, str] = {}  # canonical → original SMILES
    ordered: list[tuple[str, str]] = []
    for smi in smiles_list:
        smi = str(smi).strip()
        if not smi:
            continue
        canonical = _canonicalize_smiles(smi)
        if canonical not in seen:
            seen[canonical] = smi
            ordered.append((smi, canonical))

    result: dict[str, str] = {}
    for idx, (smi, canonical) in enumerate(ordered):
        h = _smiles_hash(smi)
        sys_dir = out / f"monomer_{h}"
        if sys_dir.exists() and (sys_dir / "set.000" / "coord.npy").exists() and not overwrite:
            _LOG.debug("Reusing cached system for SMILES %.40s → %s", smi, sys_dir.name)
            result[canonical] = str(sys_dir)
            continue
        try:
            symbols, coords = polymer_smiles_to_3d(smi, seed=seed)
            _write_monomer_system(symbols, coords, sys_dir)
            result[canonical] = str(sys_dir)
            _LOG.debug("Generated system for SMILES %.40s → %s", smi, sys_dir.name)
        except Exception as exc:
            _LOG.warning("Failed to generate system for SMILES %.60r: %s", smi, exc)

    _LOG.info(
        "generate_monomer_systems: %d unique SMILES → %d systems (seed=%#x, overwrite=%s)",
        len(ordered),
        len(result),
        seed,
        overwrite,
    )
    return result


# ---------------------------------------------------------------------------
# Step (b): Recipe assembly
# ---------------------------------------------------------------------------


def _read_european_csv(csv_path: str | Path) -> list[dict]:
    """Read a European CSV (sep=';', decimal=',') into a list of row dicts.

    Requires *pandas* for robust decimal-comma handling.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pandas is required to read European-format CSV files "
            "(sep=';', decimal=','). Install it with: pip install pandas"
        ) from exc

    df = pd.read_csv(str(csv_path), sep=";", decimal=",", encoding="utf-8")
    return df.to_dict(orient="records")


def _extract_row_groups(
    row: dict,
    systems_dir: Path,
    row_idx: int,
) -> dict | None:
    """Build the ``groups`` dict for one CSV row.

    Returns a dict with keys ``"end_groups"`` and ``"repeating"``, each
    a list of ``{"path": rel_path, "weight": float}`` dicts, or ``None``
    when the row should be skipped.

    End-group weights are computed via :func:`_calc_ends_share` using the
    ``Mn`` column.  Repeating-unit weights are the raw molpercent values
    (already mole fractions; summing to ≈1).

    Rows with missing ``Mn`` are skipped.  Repeating units with NaN
    molpercent are skipped silently.
    """
    # --- Mn (number-average molar mass) ---
    mn_raw = row.get("Mn")
    if _is_empty(mn_raw):
        _LOG.debug("Row %d: Mn is missing/NaN; skipping.", row_idx)
        return None
    try:
        mn = float(mn_raw)
    except (TypeError, ValueError):
        _LOG.warning("Row %d: cannot parse Mn=%r; skipping.", row_idx, mn_raw)
        return None

    # --- collect repeating units ---
    rep_units: list[tuple[float, str]] = []  # (molp, smiles)
    for smi_col, pct_col in zip(_MONOMER_SMILES_COLS, _MONOMER_PCT_COLS):
        smi_raw = row.get(smi_col)
        pct_raw = row.get(pct_col)
        if _is_empty(smi_raw) or _is_empty(pct_raw):
            continue
        smi = str(smi_raw).strip()
        try:
            molp = float(pct_raw)
        except (TypeError, ValueError):
            continue
        rep_units.append((molp, smi))

    if not rep_units:
        _LOG.warning("Row %d: no repeating units; skipping.", row_idx)
        return None

    # --- collect end groups ---
    end_smiles: list[str] = []
    for eg_col in _END_GROUP_COLS:
        eg_raw = row.get(eg_col)
        if not _is_empty(eg_raw):
            end_smiles.append(str(eg_raw).strip())

    # --- compute end_share if end groups present ---
    end_share: float = 0.0
    if end_smiles:
        try:
            end_share = _calc_ends_share(end_smiles, rep_units, mn)
        except Exception as exc:
            _LOG.warning(
                "Row %d: calc_ends_share failed (%s); end groups skipped.", row_idx, exc
            )
            end_smiles = []

    # --- resolve system directories and build groups ---
    def _resolve(smiles: str, weight: float) -> dict | None:
        h = _smiles_hash(smiles)
        sys_dir = systems_dir / f"monomer_{h}"
        if not sys_dir.is_dir():
            _LOG.warning(
                "Row %d: system dir not found for SMILES %.40r at %s.",
                row_idx, smiles, sys_dir,
            )
            return None
        rel = sys_dir.relative_to(systems_dir.parent)
        return {"path": str(rel), "weight": round(weight, 10)}

    end_group_comps: list[dict] = []
    for smi in end_smiles:
        c = _resolve(smi, end_share)
        if c is not None:
            end_group_comps.append(c)

    repeating_comps: list[dict] = []
    any_nonzero = False
    for molp, smi in rep_units:
        c = _resolve(smi, molp)
        if c is not None:
            repeating_comps.append(c)
            if molp > 0.0:
                any_nonzero = True

    if not any_nonzero:
        _LOG.warning("Row %d: no non-zero repeating units with system dirs; skipping.", row_idx)
        return None

    return {
        "end_groups": end_group_comps,
        "repeating": repeating_comps,
    }


def assemble_recipe_json(
    csv_path: str | Path,
    systems_dir: str | Path,
    output_dir: str | Path,
    *,
    property_name: str = "Property",
    overwrite: bool = False,
) -> Path:
    """Step (b): Build ``recipe.json`` from an already-generated systems directory.

    Reads the European-format polymer CSV (sep=';', decimal=',') and
    matches each row's SMILES to system directories under *systems_dir* by
    re-computing their hash.

    Rows with missing ``Mn`` are skipped.  End-group mole fractions are
    computed via :func:`_calc_ends_share`.  The output uses the **groups
    schema** (``end_groups`` + ``repeating``) to encode the two-block
    ``create_pfp`` fingerprint structure.

    Parameters
    ----------
    csv_path : str | Path
        Path to the European CSV (sep=';', decimal=',').
    systems_dir : str | Path
        Directory containing ``monomer_{hash}/`` subdirectories.
    output_dir : str | Path
        Root output directory; ``recipe.json`` → ``output_dir/recipe.json``.
    property_name : str
        CSV column name of the target property label.
    overwrite : bool
        Overwrite an existing ``recipe.json``.

    Returns
    -------
    Path
        Absolute path to the written ``recipe.json``.
    """
    out = Path(output_dir).resolve()
    recipe_path = out / "recipe.json"
    if recipe_path.is_file() and not overwrite:
        _LOG.info("recipe.json already exists at %s; skipping.", recipe_path)
        return recipe_path

    sys_dir = Path(systems_dir).resolve()
    rows = _read_european_csv(csv_path)

    recipes: dict[str, dict] = {}
    n_skipped = 0
    for row_idx, row in enumerate(rows):
        # --- property label ---
        raw_label = row.get(property_name)
        if _is_empty(raw_label):
            _LOG.warning(
                "Row %d: property column %r missing/empty; skipping.", row_idx, property_name
            )
            n_skipped += 1
            continue
        try:
            label = float(raw_label)
        except (TypeError, ValueError):
            _LOG.warning("Row %d: cannot parse property %r; skipping.", row_idx, raw_label)
            n_skipped += 1
            continue

        # --- groups ---
        groups = _extract_row_groups(row, sys_dir, row_idx)
        if groups is None:
            n_skipped += 1
            continue

        recipe_key = f"recipe_{len(recipes):03d}"
        recipes[recipe_key] = {"label": label, "groups": groups}

    out.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(json.dumps(recipes, indent=2), encoding="utf-8")

    _LOG.info(
        "assemble_recipe_json: %d recipes → %s (%d rows skipped).",
        len(recipes),
        recipe_path,
        n_skipped,
    )
    return recipe_path


# ---------------------------------------------------------------------------
# Step (c): Combined pipeline
# ---------------------------------------------------------------------------


def polymer_fingerprint_to_npy(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    property_name: str = "Property",
    seed: int = _ETKDG_SEED,
    overwrite: bool = False,
) -> PolymerConvertResult:
    """Convert a polymer recipe CSV to ``deepmd/npy`` systems + ``recipe.json``.

    Runs step (a) :func:`generate_monomer_systems` and then step (b)
    :func:`assemble_recipe_json` in sequence.

    Input CSV format (European, sep=';', decimal=','):
      - ``SMILES_repeating_unitA`` … ``SMILES_repeating_unitE``
      - ``molpercent_repeating_unitA`` … ``molpercent_repeating_unitE``
        (mole fractions; rows with NaN molpercent for a unit skip that unit)
      - ``SMILES_start_group``, ``SMILES_end_group``
      - ``Mn``  (number-average molar mass; rows with NaN Mn are skipped)
      - ``<property_name>``  (target label)

    Parameters
    ----------
    csv_path : str | Path
        Path to the European-format CSV.
    output_dir : str | Path
        Root output directory.  Systems → ``output_dir/systems/``;
        recipe → ``output_dir/recipe.json``.
    property_name : str
        CSV column name of the target property.
    seed : int
        ETKDGv3 random seed.  Default ``0xC0FFEE`` matches the reference.
    overwrite : bool
        Re-generate systems and recipe.json even if they already exist.

    Returns
    -------
    PolymerConvertResult
    """
    out = Path(output_dir).resolve()
    systems_dir = out / "systems"

    # --- collect all unique SMILES from the CSV ---
    rows = _read_european_csv(csv_path)

    def _cell_str(val: object) -> str:
        if _is_empty(val):
            return ""
        return str(val).strip()

    all_smiles: list[str] = []
    for row in rows:
        for col in _MONOMER_SMILES_COLS + _END_GROUP_COLS:
            smi = _cell_str(row.get(col))
            if smi:
                all_smiles.append(smi)

    # --- step (a): generate conformers ---
    smiles_map = generate_monomer_systems(
        all_smiles, systems_dir, seed=seed, overwrite=overwrite
    )
    # Identify failures (SMILES present in CSV but not in smiles_map).
    seen_canonicals = set(smiles_map.keys())
    failed_smiles: list[tuple[str, str]] = []
    for smi in set(all_smiles):
        canonical = _canonicalize_smiles(smi)
        if canonical not in seen_canonicals:
            failed_smiles.append((smi, "conformer generation failed"))

    # --- step (b): assemble recipe ---
    recipe_path = assemble_recipe_json(
        csv_path,
        systems_dir,
        out,
        property_name=property_name,
        overwrite=overwrite,
    )

    with recipe_path.open(encoding="utf-8") as fh:
        n_recipes = len(json.load(fh))

    return PolymerConvertResult(
        recipe_path=recipe_path,
        systems_dir=systems_dir,
        n_unique_smiles=len(smiles_map),
        n_recipes=n_recipes,
        failed_smiles=failed_smiles,
    )
