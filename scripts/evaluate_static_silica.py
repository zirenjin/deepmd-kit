#!/usr/bin/env python3
import csv
import glob
import itertools
import json
from pathlib import Path

import numpy as np


def main(root: Path, reference: Path, output: Path) -> None:
    phases = ["quartz_beta", "cristobalite_beta", "tridymite_p63mmc", "tridymite_c2221"]
    natoms = {"quartz_beta": 243, "cristobalite_beta": 192,
              "tridymite_p63mmc": 216, "tridymite_c2221": 24}
    with reference.open() as f:
        rows = list(csv.DictReader(f))
    temperature = np.array([float(row["temperature"]) for row in rows])
    ref = {p: np.array([float(row[p]) for row in rows]) for p in phases}
    pred = {}
    for phase in phases:
        files = sorted(glob.glob(str(root / f"{phase}.property.out.*")),
                       key=lambda path: int(path.rsplit(".", 1)[1]))
        values = []
        for path in files:
            lines = [line for line in Path(path).read_text().splitlines()
                     if line and not line.startswith("#")]
            values.append(float(lines[-1].split()[1]) / natoms[phase])
        pred[phase] = np.array(values)
    if any(len(pred[p]) != len(temperature) for p in phases):
        raise RuntimeError({p: len(pred[p]) for p in phases})

    result = {"n_points": len(temperature), "pairs": {}, "ranking": {},
              "cycle_inconsistency": {}}
    for a, b in itertools.combinations(phases, 2):
        key = f"{a}__{b}"
        delta = pred[a] - pred[b]
        target = ref[a] - ref[b]
        crossings = []
        predicted_crossings = []
        for values, found in ((target, crossings), (delta, predicted_crossings)):
            for i in range(len(temperature) - 1):
                if values[i] * values[i + 1] < 0:
                    found.append(float(temperature[i] - values[i] *
                                       (temperature[i + 1] - temperature[i]) /
                                       (values[i + 1] - values[i])))
        result["pairs"][key] = {
            "mae_eV_per_atom": float(np.mean(abs(delta - target))),
            "rmse_eV_per_atom": float(np.sqrt(np.mean((delta - target) ** 2))),
            "sign_accuracy": float(np.mean(np.sign(delta) == np.sign(target))),
            "reference_crossings_K": crossings,
            "predicted_crossings_K": predicted_crossings,
        }

    pred_matrix = np.vstack([pred[p] for p in phases])
    ref_matrix = np.vstack([ref[p] for p in phases])
    pred_rank = np.argmin(pred_matrix, axis=0)
    ref_rank = np.argmin(ref_matrix, axis=0)
    result["ranking"] = {
        "accuracy": float(np.mean(pred_rank == ref_rank)),
        "reference_stable_sequence": [phases[i] for i in np.unique(ref_rank)],
        "predicted_stable_sequence": [phases[i] for i in np.unique(pred_rank)],
    }
    cycle_values = []
    for a, b, c in itertools.combinations(phases, 3):
        cycle_values.append((pred[a] - pred[b]) + (pred[b] - pred[c]) -
                            (pred[a] - pred[c]))
    cycle = np.concatenate(cycle_values)
    result["cycle_inconsistency"] = {"max_eV_per_atom": float(np.max(abs(cycle))),
                                      "mean_eV_per_atom": float(np.mean(abs(cycle)))}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.reference, args.output)
