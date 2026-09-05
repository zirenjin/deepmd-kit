#!/usr/bin/env python3
"""Convert the public CaSiO3 davemaoite figure data to audit-friendly CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zenodo_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Fig. 4b is the published continuous Delta-G curve at 50 GPa.
    raw = pd.read_excel(args.zenodo_dir / "Fig4.xlsx", sheet_name="Fig4b", header=None)
    curve = raw.iloc[1:, [0, 1, 4]].copy()
    curve.columns = ["temperature_K", "delta_g_tet_minus_cub_eV_per_atom", "delta_g_fit_eV_per_atom"]
    curve = curve.dropna(subset=["temperature_K", "delta_g_tet_minus_cub_eV_per_atom"])
    curve = curve.sort_values("temperature_K").reset_index(drop=True)
    curve.to_csv(args.output / "delta_g_50GPa.csv", index=False)

    # Keep the complete workbook table losslessly for the multi-pressure boundary.
    boundary = pd.read_excel(args.zenodo_dir / "Fig5.xlsx", sheet_name="Fig5", header=None)
    boundary.to_csv(args.output / "phase_boundary_fig5_raw.csv", index=False, header=False)

    metadata = {
        "source": "Wu et al., Deep-learning-based prediction of the tetragonal-cubic transition in davemaoite",
        "doi": "10.5281/zenodo.10460440",
        "functional_note": "Figure data report DP-LDA and DP-GGA results; no DFT-TI G table is supplied.",
        "delta_g_definition": "G_tetragonal - G_cubic",
        "delta_g_source": "Fig4.xlsx / Fig4b",
        "delta_g_pressure_GPa": 50.0,
        "boundary_source": "Fig5.xlsx / Fig5",
        "units": {"temperature": "K", "pressure": "GPa", "delta_g": "eV/atom"},
        "files": ["delta_g_50GPa.csv", "phase_boundary_fig5_raw.csv"],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
