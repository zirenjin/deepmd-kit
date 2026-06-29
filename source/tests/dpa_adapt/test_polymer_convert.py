# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the polymer-recipe convert pipeline (Deliverable 1).

Tests for:
  - _smiles_hash / _canonicalize_smiles (deduplication)
  - generate_monomer_systems (step a, via polymer_smiles_to_3d)
  - assemble_recipe_json (step b, groups schema)
  - polymer_fingerprint_to_npy (step c)
  - convert(fmt="polymer_fingerprint") dispatch

Key design notes:
  - Conformer generation uses polymer_smiles_to_3d (cap_open_valences + ETKDGv3),
    NOT the generic smiles_to_3d_coords (which does not cap radical electrons).
  - assemble_recipe_json requires Mn in every row; rows without Mn are skipped.
  - recipe.json uses the groups schema: {"end_groups": [...], "repeating": [...]}.
  - Weights are raw molpercent values (already mole fractions in the real CSV).
  - End-group weights are calc_ends_share(ends, rep_units, Mn) — requires RDKit.

RDKit tests are guarded with pytest.importorskip so the suite degrades
gracefully when RDKit is absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from dpa_adapt.data.polymer import (
    _canonicalize_smiles,
    _smiles_hash,
    _write_monomer_system,
    assemble_recipe_json,
    generate_monomer_systems,
    polymer_fingerprint_to_npy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_system(sys_dir: Path) -> None:
    """Write a minimal deepmd/npy system (no RDKit needed) for recipe tests."""
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "type_map.raw").write_text("H\nC\n", encoding="utf-8")
    (sys_dir / "type.raw").write_text("0\n1\n", encoding="utf-8")
    sd = sys_dir / "set.000"
    sd.mkdir(exist_ok=True)
    np.save(str(sd / "coord.npy"), np.zeros((1, 6), dtype=np.float64))
    np.save(str(sd / "box.npy"), (np.eye(3) * 100.0).reshape(1, 9).astype(np.float64))


def _write_european_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal European CSV (sep=';', decimal='.').

    Values are written as-is.  For fraction weights, pass values like
    0.7 and 0.3 directly; pandas with ``decimal=','`` still parses "0.7"
    as 0.7.
    """
    if not rows:
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as fh:
        fh.write(";".join(headers) + "\n")
        for row in rows:
            fh.write(";".join(str(row.get(h, "")) for h in headers) + "\n")


# ---------------------------------------------------------------------------
# SMILES hash / canonicalisation
# ---------------------------------------------------------------------------


class TestSmilesHash:
    def test_same_smiles_same_hash(self) -> None:
        h1 = _smiles_hash("CC")
        h2 = _smiles_hash("CC")
        assert h1 == h2

    def test_different_smiles_different_hash(self) -> None:
        assert _smiles_hash("CC") != _smiles_hash("O")

    def test_hash_length(self) -> None:
        h = _smiles_hash("CCO")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_canonicalize_fallback_without_rdkit(self) -> None:
        """_canonicalize_smiles returns input unchanged when RDKit unavailable."""
        with mock.patch.dict("sys.modules", {"rdkit": None, "rdkit.Chem": None}):
            result = _canonicalize_smiles("CC")
        assert result == "CC"

    def test_hash_stable_without_rdkit(self) -> None:
        """Hash is reproducible even in fallback mode."""
        with mock.patch.dict("sys.modules", {"rdkit": None, "rdkit.Chem": None}):
            h1 = _smiles_hash("CC")
            h2 = _smiles_hash("CC")
        assert h1 == h2


class TestCanonicalizeSmiles:
    def test_rdkit_available(self) -> None:
        pytest.importorskip("rdkit")
        # Ethanol can be written in multiple ways
        c1 = _canonicalize_smiles("CCO")
        c2 = _canonicalize_smiles("OCC")
        assert c1 == c2


# ---------------------------------------------------------------------------
# _write_monomer_system (unit-level, RDKit not needed for the writer itself)
# ---------------------------------------------------------------------------


class TestWriteMonomerSystem:
    def test_writes_expected_files(self, tmp_path: Path) -> None:
        symbols = ["H", "O", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        sys_dir = tmp_path / "test_sys"
        _write_monomer_system(symbols, coords, sys_dir)
        assert (sys_dir / "type_map.raw").is_file()
        assert (sys_dir / "type.raw").is_file()
        assert (sys_dir / "set.000" / "coord.npy").is_file()
        assert (sys_dir / "set.000" / "box.npy").is_file()

    def test_coord_shape(self, tmp_path: Path) -> None:
        symbols = ["C", "H", "H", "H", "H"]
        rng = np.random.default_rng(0)
        coords = rng.random((5, 3)).astype(np.float32)
        sys_dir = tmp_path / "ch4"
        _write_monomer_system(symbols, coords, sys_dir)
        loaded = np.load(str(sys_dir / "set.000" / "coord.npy"))
        assert loaded.shape == (1, 5 * 3)

    def test_box_shape(self, tmp_path: Path) -> None:
        symbols = ["O"]
        coords = np.array([[0.0, 0.0, 0.0]])
        sys_dir = tmp_path / "O"
        _write_monomer_system(symbols, coords, sys_dir)
        box = np.load(str(sys_dir / "set.000" / "box.npy"))
        assert box.shape == (1, 9)

    def test_type_map_raw_content(self, tmp_path: Path) -> None:
        symbols = ["H", "C", "H"]
        coords = np.zeros((3, 3))
        sys_dir = tmp_path / "hch"
        _write_monomer_system(symbols, coords, sys_dir)
        tm = (sys_dir / "type_map.raw").read_text().strip().split("\n")
        assert "H" in tm
        assert "C" in tm
        assert tm.index("H") < tm.index("C")


# ---------------------------------------------------------------------------
# generate_monomer_systems — mock polymer_smiles_to_3d (new internal function)
# ---------------------------------------------------------------------------


class TestGenerateMonomersystems:
    """Tests that mock polymer_smiles_to_3d (not smiles_to_3d_coords).

    polymer_smiles_to_3d is the polymer-specific conformer builder that
    caps radical electrons before AddHs.  The mock must use the same
    signature: (smiles, *, cap=True, seed=int) -> (list[str], ndarray).
    """

    @staticmethod
    def _mock_smiles(smiles: str, *, cap: bool = True, seed: int = 0) -> tuple:
        """Return deterministic mock (symbols, coords) for any SMILES."""
        rng = np.random.default_rng(abs(hash(smiles)) % 2**32)
        return (["H", "C"], rng.random((2, 3)).astype(np.float32))

    def test_two_unique_smiles(self, tmp_path: Path) -> None:
        with mock.patch(
            "dpa_adapt.data.polymer.polymer_smiles_to_3d",
            side_effect=self._mock_smiles,
        ):
            result = generate_monomer_systems(["CC", "O"], tmp_path / "systems")
        assert len(result) == 2
        for sys_path in result.values():
            p = Path(sys_path)
            assert p.is_dir()
            assert (p / "set.000" / "coord.npy").is_file()

    def test_deduplication(self, tmp_path: Path) -> None:
        """Same canonical SMILES → only one system directory."""
        with mock.patch(
            "dpa_adapt.data.polymer.polymer_smiles_to_3d",
            side_effect=self._mock_smiles,
        ):
            result = generate_monomer_systems(["CC", "CC", "CC"], tmp_path / "systems")
        assert len(result) == 1

    def test_empty_strings_skipped(self, tmp_path: Path) -> None:
        with mock.patch(
            "dpa_adapt.data.polymer.polymer_smiles_to_3d",
            side_effect=self._mock_smiles,
        ):
            result = generate_monomer_systems(["", "O", ""], tmp_path / "systems")
        assert len(result) == 1

    def test_failed_smiles_logged_not_raised(self, tmp_path: Path) -> None:
        """A failing SMILES is logged and the result simply omits it."""

        def _fail_first(smiles: str, *, cap: bool = True, seed: int = 0):
            if smiles == "INVALID":
                raise ValueError("bad SMILES")
            return (["H", "C"], np.zeros((2, 3), dtype=np.float32))

        with mock.patch(
            "dpa_adapt.data.polymer.polymer_smiles_to_3d", side_effect=_fail_first
        ):
            result = generate_monomer_systems(["INVALID", "O"], tmp_path / "systems")
        assert len(result) == 1

    def test_overwrite_false_reuses_existing(self, tmp_path: Path) -> None:
        """If overwrite=False and dir exists with coord.npy, skip generation."""
        sys_dir = tmp_path / "systems"
        smi = "O"
        h = _smiles_hash(smi)
        existing = sys_dir / f"monomer_{h}"
        _write_fake_system(existing)

        call_count = {"n": 0}

        def _count(smiles: str, *, cap: bool = True, seed: int = 0):
            call_count["n"] += 1
            return (["H", "O", "H"], np.zeros((3, 3), dtype=np.float32))

        with mock.patch(
            "dpa_adapt.data.polymer.polymer_smiles_to_3d", side_effect=_count
        ):
            generate_monomer_systems([smi], sys_dir, overwrite=False)

        assert call_count["n"] == 0, (
            "polymer_smiles_to_3d should not be called when dir exists"
        )


# ---------------------------------------------------------------------------
# RDKit integration: polymer SMILES with radical electrons
# ---------------------------------------------------------------------------


class TestGenerateMonomersystemsRealRDKit:
    def test_water_and_methane(self, tmp_path: Path) -> None:
        pytest.importorskip("rdkit")
        result = generate_monomer_systems(
            ["O", "C"],  # water and methane — simple closed-shell SMILES
            tmp_path / "systems",
            seed=0,
        )
        assert len(result) == 2
        for sys_path in result.values():
            p = Path(sys_path)
            coord = np.load(str(p / "set.000" / "coord.npy"))
            assert coord.ndim == 2
            assert coord.shape[0] == 1  # single frame

    def test_radical_polymer_smiles(self, tmp_path: Path) -> None:
        """Polymer SMILES with open valences (radical electrons) should be capped."""
        pytest.importorskip("rdkit")
        # Polymer SMILES: [CH2][CH](C(=O)OC) is poly(methyl acrylate) repeat unit
        # [CH2] has 2 radical electrons, [CH] has 1 → cap_open_valences fills them.
        result = generate_monomer_systems(
            ["[CH2][CH](C(=O)OC)"],
            tmp_path / "systems",
            seed=0,
        )
        assert len(result) == 1
        p = Path(list(result.values())[0])
        coord = np.load(str(p / "set.000" / "coord.npy"))
        assert coord.shape[0] == 1


# ---------------------------------------------------------------------------
# assemble_recipe_json — uses the groups schema; Mn is required
# ---------------------------------------------------------------------------


class TestAssembleRecipeJson:
    def _make_systems(self, systems_dir: Path, smiles_list: list[str]) -> None:
        """Pre-create minimal system dirs at expected hash paths."""
        for smi in smiles_list:
            h = _smiles_hash(smi)
            _write_fake_system(systems_dir / f"monomer_{h}")

    def test_basic_recipe_json(self, tmp_path: Path) -> None:
        """Two repeating units (no end groups) → groups schema with repeating key."""
        smi_a = "CC"
        smi_b = "O"
        systems_dir = tmp_path / "systems"
        self._make_systems(systems_dir, [smi_a, smi_b])

        csv_path = tmp_path / "polymer.csv"
        _write_european_csv(
            csv_path,
            [
                {
                    "SMILES_repeating_unitA": smi_a,
                    "SMILES_repeating_unitB": smi_b,
                    "SMILES_repeating_unitC": "",
                    "SMILES_repeating_unitD": "",
                    "SMILES_repeating_unitE": "",
                    "molpercent_repeating_unitA": "0.7",
                    "molpercent_repeating_unitB": "0.3",
                    "molpercent_repeating_unitC": "",
                    "molpercent_repeating_unitD": "",
                    "molpercent_repeating_unitE": "",
                    "SMILES_start_group": "",
                    "SMILES_end_group": "",
                    "Mn": "10000",
                    "Tg": "85",
                },
            ],
        )

        recipe_path = assemble_recipe_json(
            csv_path, systems_dir, tmp_path, property_name="Tg"
        )

        assert recipe_path.is_file()
        data = json.loads(recipe_path.read_text())
        assert len(data) == 1
        recipe = list(data.values())[0]
        assert recipe["label"] == pytest.approx(85.0)

        # Groups schema: no end groups → only "repeating"
        assert "groups" in recipe
        groups = recipe["groups"]
        assert "repeating" in groups
        rep_comps = groups["repeating"]
        assert len(rep_comps) == 2
        weights = [c["weight"] for c in rep_comps]
        assert sum(weights) == pytest.approx(1.0)

    def test_mn_required_row_skipped_without_mn(self, tmp_path: Path) -> None:
        """Rows with missing Mn are skipped (Mn is required for all rows)."""
        smi_a = "CC"
        systems_dir = tmp_path / "systems"
        self._make_systems(systems_dir, [smi_a])

        csv_path = tmp_path / "p.csv"
        _write_european_csv(
            csv_path,
            [
                {
                    "SMILES_repeating_unitA": smi_a,
                    "SMILES_repeating_unitB": "",
                    "SMILES_repeating_unitC": "",
                    "SMILES_repeating_unitD": "",
                    "SMILES_repeating_unitE": "",
                    "molpercent_repeating_unitA": "1.0",
                    "molpercent_repeating_unitB": "",
                    "molpercent_repeating_unitC": "",
                    "molpercent_repeating_unitD": "",
                    "molpercent_repeating_unitE": "",
                    "SMILES_start_group": "",
                    "SMILES_end_group": "",
                    # Mn column absent → _is_empty → row skipped
                    "Tg": "42",
                },
            ],
        )

        recipe_path = assemble_recipe_json(
            csv_path, systems_dir, tmp_path, property_name="Tg"
        )
        data = json.loads(recipe_path.read_text())
        assert len(data) == 0  # row skipped because Mn is missing

    def test_repeating_weights_preserved(self, tmp_path: Path) -> None:
        """Repeating-unit weights equal raw molpercent values (no normalization)."""
        smi_a, smi_b, smi_c = "CC", "O", "N"
        systems_dir = tmp_path / "systems"
        self._make_systems(systems_dir, [smi_a, smi_b, smi_c])

        csv_path = tmp_path / "p.csv"
        _write_european_csv(
            csv_path,
            [
                {
                    "SMILES_repeating_unitA": smi_a,
                    "SMILES_repeating_unitB": smi_b,
                    "SMILES_repeating_unitC": smi_c,
                    "SMILES_repeating_unitD": "",
                    "SMILES_repeating_unitE": "",
                    "molpercent_repeating_unitA": "0.5",
                    "molpercent_repeating_unitB": "0.3",
                    "molpercent_repeating_unitC": "0.2",
                    "molpercent_repeating_unitD": "",
                    "molpercent_repeating_unitE": "",
                    "SMILES_start_group": "",
                    "SMILES_end_group": "",
                    "Mn": "5000",
                    "Prop": "10",
                },
            ],
        )

        recipe_path = assemble_recipe_json(
            csv_path, systems_dir, tmp_path, property_name="Prop"
        )
        data = json.loads(recipe_path.read_text())
        recipe = list(data.values())[0]
        rep_comps = recipe["groups"]["repeating"]
        weights = sorted(c["weight"] for c in rep_comps)
        # Weights 0.2, 0.3, 0.5 are preserved as-is.
        assert weights == pytest.approx(sorted([0.2, 0.3, 0.5]))

    def test_end_groups_appear_in_groups_schema(self, tmp_path: Path) -> None:
        """With RDKit: end groups appear in groups['end_groups'] with calc_ends_share."""
        pytest.importorskip("rdkit")
        # Simple SMILES that RDKit can handle
        smi_m = "CC"       # repeating unit (ethane)
        smi_sg = "C"       # start group (methane)
        smi_eg = "N"       # end group (ammonia)
        systems_dir = tmp_path / "systems"
        self._make_systems(systems_dir, [smi_m, smi_sg, smi_eg])

        csv_path = tmp_path / "p.csv"
        _write_european_csv(
            csv_path,
            [
                {
                    "SMILES_repeating_unitA": smi_m,
                    "SMILES_repeating_unitB": "",
                    "SMILES_repeating_unitC": "",
                    "SMILES_repeating_unitD": "",
                    "SMILES_repeating_unitE": "",
                    "molpercent_repeating_unitA": "1.0",
                    "molpercent_repeating_unitB": "",
                    "molpercent_repeating_unitC": "",
                    "molpercent_repeating_unitD": "",
                    "molpercent_repeating_unitE": "",
                    "SMILES_start_group": smi_sg,
                    "SMILES_end_group": smi_eg,
                    "Mn": "5000",
                    "P": "5",
                },
            ],
        )

        recipe_path = assemble_recipe_json(
            csv_path, systems_dir, tmp_path, property_name="P"
        )
        data = json.loads(recipe_path.read_text())
        recipe = list(data.values())[0]
        groups = recipe["groups"]

        # End groups should appear with non-zero weight.
        assert "end_groups" in groups
        eg_comps = groups["end_groups"]
        assert len(eg_comps) == 2
        for c in eg_comps:
            assert c["weight"] > 0.0, "End-group calc_ends_share should be positive"

        # Repeating units should appear.
        assert "repeating" in groups
        assert len(groups["repeating"]) == 1

    def test_missing_system_skips_row(self, tmp_path: Path) -> None:
        """A row whose monomer has no system dir should be skipped gracefully."""
        systems_dir = tmp_path / "systems"
        systems_dir.mkdir()

        csv_path = tmp_path / "p.csv"
        _write_european_csv(
            csv_path,
            [
                {
                    "SMILES_repeating_unitA": "CC",
                    "SMILES_repeating_unitB": "",
                    "SMILES_repeating_unitC": "",
                    "SMILES_repeating_unitD": "",
                    "SMILES_repeating_unitE": "",
                    "molpercent_repeating_unitA": "1.0",
                    "molpercent_repeating_unitB": "",
                    "molpercent_repeating_unitC": "",
                    "molpercent_repeating_unitD": "",
                    "molpercent_repeating_unitE": "",
                    "SMILES_start_group": "",
                    "SMILES_end_group": "",
                    "Mn": "10000",
                    "P": "10",
                },
            ],
        )

        recipe_path = assemble_recipe_json(
            csv_path, systems_dir, tmp_path, property_name="P"
        )
        data = json.loads(recipe_path.read_text())
        assert len(data) == 0  # row was skipped

    def test_recipe_key_format(self, tmp_path: Path) -> None:
        smi_a = "CC"
        systems_dir = tmp_path / "systems"
        self._make_systems(systems_dir, [smi_a])
        csv_path = tmp_path / "p.csv"
        rows = [
            {
                "SMILES_repeating_unitA": smi_a,
                "SMILES_repeating_unitB": "",
                "SMILES_repeating_unitC": "",
                "SMILES_repeating_unitD": "",
                "SMILES_repeating_unitE": "",
                "molpercent_repeating_unitA": "1.0",
                "molpercent_repeating_unitB": "",
                "molpercent_repeating_unitC": "",
                "molpercent_repeating_unitD": "",
                "molpercent_repeating_unitE": "",
                "SMILES_start_group": "",
                "SMILES_end_group": "",
                "Mn": "10000",
                "P": str(i),
            }
            for i in range(3)
        ]
        _write_european_csv(csv_path, rows)
        recipe_path = assemble_recipe_json(
            csv_path, systems_dir, tmp_path, property_name="P"
        )
        data = json.loads(recipe_path.read_text())
        assert list(data.keys()) == ["recipe_000", "recipe_001", "recipe_002"]


# ---------------------------------------------------------------------------
# convert() dispatch with fmt="polymer_fingerprint"
# ---------------------------------------------------------------------------


class TestConvertPolymerFingerprintDispatch:
    def test_convert_routes_to_polymer(self, tmp_path: Path) -> None:
        from dpa_adapt.data.convert import convert

        smi = "O"
        systems_dir = tmp_path / "output" / "systems"
        h = _smiles_hash(smi)
        _write_fake_system(systems_dir / f"monomer_{h}")

        csv_path = tmp_path / "p.csv"
        _write_european_csv(
            csv_path,
            [
                {
                    "SMILES_repeating_unitA": smi,
                    "SMILES_repeating_unitB": "",
                    "SMILES_repeating_unitC": "",
                    "SMILES_repeating_unitD": "",
                    "SMILES_repeating_unitE": "",
                    "molpercent_repeating_unitA": "1.0",
                    "molpercent_repeating_unitB": "",
                    "molpercent_repeating_unitC": "",
                    "molpercent_repeating_unitD": "",
                    "molpercent_repeating_unitE": "",
                    "SMILES_start_group": "",
                    "SMILES_end_group": "",
                    "Mn": "10000",
                    "P": "42",
                },
            ],
        )

        with mock.patch("dpa_adapt.data.polymer.generate_monomer_systems") as mock_gen, \
             mock.patch("dpa_adapt.data.polymer.assemble_recipe_json") as mock_asm, \
             mock.patch("dpa_adapt.data.polymer._read_european_csv") as mock_csv:
            from dpa_adapt.data.polymer import PolymerConvertResult
            mock_csv.return_value = []
            mock_gen.return_value = {}
            mock_recipe = tmp_path / "output" / "recipe.json"
            mock_recipe.parent.mkdir(parents=True, exist_ok=True)
            mock_recipe.write_text("{}", encoding="utf-8")
            mock_asm.return_value = mock_recipe
            with mock.patch("json.load", return_value={}):
                result = convert(
                    str(csv_path),
                    str(tmp_path / "output"),
                    fmt="polymer_fingerprint",
                    property_name="P",
                )

        assert result["method"] == "polymer_fingerprint"
        assert "recipe_path" in result
        assert "systems_dir" in result
        assert "n_recipes" in result


# ---------------------------------------------------------------------------
# polymer_fingerprint_to_npy end-to-end with real RDKit
# ---------------------------------------------------------------------------


class TestPolymerFingerprintToNpyRealRDKit:
    def test_two_row_csv_no_end_groups(self, tmp_path: Path) -> None:
        """Two rows, no end groups (no Mn needed for end-share computation)."""
        pytest.importorskip("rdkit")

        smi_a = "C"   # methane
        smi_b = "O"   # water
        csv_path = tmp_path / "poly.csv"
        _write_european_csv(
            csv_path,
            [
                {
                    "SMILES_repeating_unitA": smi_a,
                    "SMILES_repeating_unitB": smi_b,
                    "SMILES_repeating_unitC": "",
                    "SMILES_repeating_unitD": "",
                    "SMILES_repeating_unitE": "",
                    "molpercent_repeating_unitA": "0.6",
                    "molpercent_repeating_unitB": "0.4",
                    "molpercent_repeating_unitC": "",
                    "molpercent_repeating_unitD": "",
                    "molpercent_repeating_unitE": "",
                    "SMILES_start_group": "",
                    "SMILES_end_group": "",
                    "Mn": "20000",
                    "Tg": "120",
                },
                {
                    "SMILES_repeating_unitA": smi_a,
                    "SMILES_repeating_unitB": "",
                    "SMILES_repeating_unitC": "",
                    "SMILES_repeating_unitD": "",
                    "SMILES_repeating_unitE": "",
                    "molpercent_repeating_unitA": "1.0",
                    "molpercent_repeating_unitB": "",
                    "molpercent_repeating_unitC": "",
                    "molpercent_repeating_unitD": "",
                    "molpercent_repeating_unitE": "",
                    "SMILES_start_group": "",
                    "SMILES_end_group": "",
                    "Mn": "10000",
                    "Tg": "50",
                },
            ],
        )

        out_dir = tmp_path / "out"
        result = polymer_fingerprint_to_npy(
            csv_path, out_dir, property_name="Tg", seed=0
        )

        assert result.recipe_path.is_file()
        assert result.systems_dir.is_dir()
        assert result.n_unique_smiles == 2  # methane and water
        assert result.n_recipes == 2

        # recipe.json structure uses groups schema
        data = json.loads(result.recipe_path.read_text())
        assert len(data) == 2
        for recipe in data.values():
            assert "label" in recipe
            assert "groups" in recipe
            groups = recipe["groups"]
            assert "repeating" in groups
            rep_weights = [c["weight"] for c in groups["repeating"] if c["weight"] > 0]
            assert all(w > 0 for w in rep_weights)

        # system dirs have expected files
        for sys_dir in result.systems_dir.iterdir():
            if sys_dir.is_dir():
                assert (sys_dir / "set.000" / "coord.npy").is_file()
