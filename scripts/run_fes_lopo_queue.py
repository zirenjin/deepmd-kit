#!/usr/bin/env python3
"""Run prepared revision-2157 phase-LOPO jobs sequentially on one GPU."""
from __future__ import annotations
import argparse, json, os, subprocess
from pathlib import Path

PHASES = ["quartz_beta", "cristobalite_beta", "tridymite_p63mmc", "tridymite_c2221"]
METHODS = ["A_DeepProperty", "B_residual_absolute", "C_paired_direct", "C_coef",
           "D_anchored_quadratic", "D_anchored_tlogT", "D_anchored_cubic"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dp", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    args = ap.parse_args()
    root = Path(args.root)
    env = os.environ.copy()
    env["PYTHONPATH"] = args.repo
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    for heldout in PHASES:
        for method in args.methods:
            for seed in args.seeds:
                name = f"{method}_holdout_{heldout}_seed{seed}"
                out = root / "sio2_lopo" / heldout / name
                log_path = out / "console_queue.log"
                log = log_path.open("w")
                result = subprocess.run(
                    [args.dp, "--pt", "train", "config.json"],
                    cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                    check=False,
                )
                log.close()
                (out / "exit_code.txt").write_text(str(result.returncode) + "\n")
                if result.returncode:
                    (out / "failure.json").write_text(json.dumps({
                        "protocol": "revision2157", "heldout_phase": heldout,
                        "method": method, "seed": seed, "exit_code": result.returncode,
                        "log": str(log_path),
                    }, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

