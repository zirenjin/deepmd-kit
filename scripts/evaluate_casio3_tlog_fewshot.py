#!/usr/bin/env python3
"""Evaluate CaSiO3 continuous t-log diagnostic runs."""
import argparse, csv, glob, json, subprocess
from pathlib import Path
import numpy as np

LABELS = {1: [0], 2: [0, 9], 4: [0, 2, 4, 9], 8: [0, 1, 2, 3, 4, 6, 7, 9]}
PHASES = {"tetragonal": 10, "cubic": 5}

def crossings(x, y, atol=1e-10):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = []
    for i in range(len(y) - 1):
        if abs(y[i]) <= atol:
            out.append(float(x[i]))
        elif y[i] * y[i + 1] < 0:
            out.append(float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])))
    return out

def read_property(path, natoms):
    lines = [s for s in Path(path).read_text().splitlines() if s.strip() and not s.startswith("#")]
    return float(lines[-1].split()[1]) / natoms

def test_phase(model, phase, data, out):
    d = out / "test" / phase
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["dp", "--pt", "test", "-m", str(model), "-s", str(data / phase), "-d", str(d)],
                   check=True, stdout=(out / f"test_{phase}.log").open("w"), stderr=subprocess.STDOUT)
    files = sorted(glob.glob(str(out / "**" / f"{phase}.property.out.*"), recursive=True),
                   key=lambda p: int(p.rsplit(".", 1)[1]))
    if not files:
        raise RuntimeError(f"no outputs for {phase}: {d}")
    return np.array([read_property(p, PHASES[phase]) for p in files])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 51, 67])
    a = ap.parse_args()
    rows = list(csv.DictReader(a.reference.open()))
    T = np.array([float(r["temperature_K"]) for r in rows])
    ref = np.array([float(r["delta_g_tet_minus_cub_eV_per_atom"]) for r in rows])
    ref_cross = crossings(T, ref)
    results = {}
    for n, idx in LABELS.items():
        results[str(n)] = {}
        for seed in a.seeds:
            run = a.root / f"casiO3_tlog_fewshot{n}_seed{seed}"
            ck = run / "model.ckpt-3000.pt"
            if not ck.exists():
                results[str(n)][str(seed)] = {"status": "missing_checkpoint"}
                continue
            ev, model = run / "eval", run / "eval" / "model.pth"
            ev.mkdir(exist_ok=True)
            if not model.exists():
                subprocess.run(["dp", "--pt", "freeze", "-c", str(ck), "-o", str(model)],
                               check=True, stdout=(ev / "freeze.log").open("w"), stderr=subprocess.STDOUT)
            pt = test_phase(model, "tetragonal", a.data_root, ev)
            pc = test_phase(model, "cubic", a.data_root, ev)
            pred = pt - pc
            mask = np.ones(T.size, dtype=bool)
            mask[idx] = False
            err = pred[mask] - ref[mask]
            crossings_pred = crossings(T, pred)
            results[str(n)][str(seed)] = {
                "status": "ok", "label_indices": idx, "label_temperatures_K": T[idx].tolist(),
                "heldout_mae_eV_per_atom": float(np.mean(np.abs(err))),
                "heldout_rmse_eV_per_atom": float(np.sqrt(np.mean(err ** 2))),
                "heldout_sign_accuracy": float(np.mean(np.sign(pred[mask]) == np.sign(ref[mask]))),
                "reference_crossings_K": ref_cross, "predicted_crossings_K": crossings_pred,
                "crossing_error_K": (min(abs(x - ref_cross[0]) for x in crossings_pred)
                                     if crossings_pred and ref_cross else None),
            }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

