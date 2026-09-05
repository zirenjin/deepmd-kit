#!/usr/bin/env python3
"""Build fixed-geometry DeepMD data from one representative per silica phase."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PHASES = ("quartz_beta", "cristobalite_beta", "tridymite_p63mmc", "tridymite_c2221")


def load_structure(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".cif":
        from ase.io import read

        atoms = read(path.as_posix())
        coords = np.asarray(atoms.get_positions(), dtype=np.float64)
        box = np.asarray(atoms.cell.array, dtype=np.float64)
        symbols = atoms.get_chemical_symbols()
        type_map = {"Si": 0, "O": 1}
        atype = np.asarray([type_map[symbol] for symbol in symbols], dtype=np.int32)
        return coords, box, atype

    import dpdata

    system = dpdata.LabeledSystem(path.as_posix(), fmt="vasp/outcar")
    return (
        np.asarray(system["coords"][0], dtype=np.float64),
        np.asarray(system["cells"][0], dtype=np.float64),
        np.asarray(system["atom_types"], dtype=np.int32),
    )


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, help="phase_g_common.csv")
    parser.add_argument("representatives", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--set-size", type=int, default=256)
    args = parser.parse_args()

    with args.reference.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    temperatures = np.asarray([float(row["temperature"]) for row in rows], dtype=np.float64)
    energies = {phase: np.asarray([float(row[phase]) for row in rows]) for phase in PHASES}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "type.raw").write_text("0 1\n")
    (args.output / "type_map.raw").write_text("Si\nO\n")

    metadata = {
        "protocol": "static-only",
        "state_input": "fparam=[temperature_K, pressure_GPa]",
        "pressure_GPa": 0.0,
        "temperature_count": len(temperatures),
        "temperature_range_K": [float(temperatures[0]), float(temperatures[-1])],
        "phases": {},
    }
    for phase in PHASES:
        source = args.representatives / f"{phase}.cif"
        if not source.exists():
            source = args.representatives / f"{phase}.OUTCAR"
        coords, box, atype = load_structure(source)
        phase_dir = args.output / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(temperatures), args.set_size):
            stop = min(start + args.set_size, len(temperatures))
            set_dir = phase_dir / f"set.{start // args.set_size:03d}"
            set_dir.mkdir()
            np.save(
                set_dir / "coord.npy",
                np.repeat(coords.reshape(1, -1), stop - start, axis=0),
            )
            np.save(set_dir / "box.npy", np.repeat(box.reshape(1, 9), stop - start, axis=0))
            np.save(
                set_dir / "fparam.npy",
                np.column_stack((temperatures[start:stop], np.zeros(stop - start))),
            )
            # DeepMD energy labels are extensive; the reference table is per atom.
            np.save(
                set_dir / "free_energy.npy",
                (energies[phase][start:stop] * len(atype))[:, None],
            )
        metadata["phases"][phase] = {
            "source": source.as_posix(),
            "n_atoms": int(len(atype)),
            "label_unit": "eV_per_structure (converted from eV_per_atom)",
            "coord_sha256": digest(coords),
            "box_sha256": digest(box),
            "atype_sha256": digest(atype),
            "coord_constant_across_temperature": True,
            "box_constant_across_temperature": True,
            "pressure_constant_across_temperature": True,
        }
        np.savetxt(phase_dir / "type.raw", atype, fmt="%d")

    (args.output / "static_protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
