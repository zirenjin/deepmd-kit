#!/usr/bin/env python3
"""Extract and validate the common PBE-D3(BJ) silica free-energy surface."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


PHASE_FILES = {
    "quartz_beta": "quartz_beta_pbe_d3bj.csv",
    "cristobalite_beta": "cristobalite_beta_pbe_d3bj.csv",
    "tridymite_p63mmc": "tridymite_beta_pbe_d3bj.csv",
    "tridymite_c2221": "tridymite_ortho_pbe_d3bj.csv",
}


def read_table(archive: zipfile.ZipFile, name: str) -> dict[int, float]:
    member = f"direct_upsampling_data_rungs_1-3/thermodynamic_properties/{name}"
    with archive.open(member) as stream:
        rows = csv.DictReader(line.decode("utf-8") for line in stream)
        result = {int(row["temperature"]): float(row["Gibbs_energy"]) for row in rows}
    if len(result) < 2:
        raise ValueError(f"too few rows in {member}")
    return result


def crossings(temperatures: list[int], values: list[float]) -> list[float]:
    result = []
    for t0, t1, y0, y1 in zip(temperatures, temperatures[1:], values, values[1:]):
        if y0 == 0:
            result.append(float(t0))
        elif y0 * y1 < 0:
            result.append(t0 + (t1 - t0) * (-y0) / (y1 - y0))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as archive:
        tables = {phase: read_table(archive, member) for phase, member in PHASE_FILES.items()}
    common = sorted(set.intersection(*(set(table) for table in tables.values())))
    if not common:
        raise ValueError("the four phase tables have no common temperature")
    data = {phase: [tables[phase][temperature] for temperature in common] for phase in tables}

    with (args.output / "phase_g_common.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["temperature", *tables])
        writer.writerows(zip(common, *(data[phase] for phase in tables)))

    pair_info = {}
    with (args.output / "pair_delta_g_common.csv").open("w", newline="") as stream:
        pairs = [(left, right) for left in tables for right in tables if left < right]
        writer = csv.writer(stream)
        writer.writerow(["temperature", *(f"{left}_minus_{right}" for left, right in pairs)])
        for index, temperature in enumerate(common):
            writer.writerow(
                [
                    temperature,
                    *[data[left][index] - data[right][index] for left, right in pairs],
                ]
            )
        for left, right in pairs:
            delta = [x - y for x, y in zip(data[left], data[right])]
            pair_info[f"{left}_minus_{right}"] = {
                "left": left,
                "right": right,
                "crossings_K": crossings(common, delta),
                "delta_min_eV_per_atom": min(delta),
                "delta_max_eV_per_atom": max(delta),
            }

    ranking = [
        sorted(tables, key=lambda phase: data[phase][index])
        for index in range(len(common))
    ]
    with (args.output / "phase_ranking_common.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["temperature", "ranked_phases"])
        writer.writerows((temperature, ">".join(order)) for temperature, order in zip(common, ranking))

    metadata = {
        "functional": "PBE-D3(BJ)",
        "pressure_assumption": "0-pressure table; source has no pressure column",
        "phase_files": PHASE_FILES,
        "phases": list(tables),
        "common_temperature_K": [common[0], common[-1]],
        "n_common_temperatures": len(common),
        "pair_info": pair_info,
        "ranking_at_endpoints": {str(common[0]): ranking[0], str(common[-1]): ranking[-1]},
    }
    (args.output / "reference_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
