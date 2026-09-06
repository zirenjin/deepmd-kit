#!/usr/bin/env python3
"""Prepare revision-2157 phase-LOPO configs from the fixed SiO2 split."""
from __future__ import annotations
import argparse, json
from pathlib import Path

PHASES = ["quartz_beta", "cristobalite_beta", "tridymite_p63mmc", "tridymite_c2221"]
SOURCE = {
    "A_DeepProperty": "A_DeepProperty_formal_seed11",
    "B_residual_absolute": "B_residual_absolute_formal_seed11",
    "C_paired_direct": "C_paired_direct_formal_seed11",
    "C_coef": "C_coef_formal_seed11",
    "D_anchored_quadratic": "D_anchored_quadratic_clean2_seed11",
    "D_anchored_tlogT": "D_anchored_tlogT_formal_seed11",
    "D_anchored_cubic": "D_anchored_cubic_clean2_seed11",
}

def remove_phase(value, heldout):
    if isinstance(value, list):
        return [v for v in value if not (isinstance(v, str) and v.rstrip("/").split("/")[-1] == heldout)]
    if isinstance(value, dict):
        return {k: remove_phase(v, heldout) for k, v in value.items()}
    return value

def rewrite(value, old, new):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [rewrite(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: rewrite(v, old, new) for k, v in value.items()}
    return value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--methods", nargs="+", choices=sorted(SOURCE), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out_root = root / "sio2_lopo"
    for heldout in PHASES:
        for method in args.methods:
            old = SOURCE[method]
            source = json.loads((root / "sio2" / old / "config.json").read_text())
            for seed in args.seeds:
                name = f"{method}_holdout_{heldout}_seed{seed}"
                out = out_root / heldout / name
                out.mkdir(parents=True, exist_ok=True)
                cfg = remove_phase(source, heldout)
                cfg = rewrite(cfg, old, name)
                cfg["training"]["seed"] = seed
                cfg["model"]["fitting_net"]["seed"] = seed
                cfg["training"]["save_ckpt"] = str(out / "model.ckpt")
                cfg["training"]["disp_file"] = str(out / "train.out")
                cfg["training"]["stat_file"] = str(out / "fes_stats.hdf5")
                cfg["training"]["training_data"]["pair_systems"] = cfg["training"]["training_data"]["systems"]
                cfg["training"]["validation_data"]["pair_systems"] = cfg["training"]["validation_data"]["systems"]
                (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
                (out / "split_manifest.json").write_text(json.dumps({
                    "protocol": "revision2157", "split": "phase_lopo",
                    "heldout_phase": heldout, "train_phases": [p for p in PHASES if p != heldout],
                    "validation_phases": [p for p in PHASES if p != heldout],
                    "test_phase": heldout, "seed": seed, "static_only": True,
                }, indent=2) + "\n")
if __name__ == "__main__":
    main()

