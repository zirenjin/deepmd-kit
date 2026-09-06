#!/usr/bin/env python3
"""Run independent FES seed queues without distributed training."""
from __future__ import annotations
import argparse, json, os, subprocess
from pathlib import Path

SOURCES = {
    "A_DeepProperty": "A_DeepProperty_formal_seed11",
    "B_residual_absolute": "B_residual_absolute_formal_seed11",
    "C_paired_direct": "C_paired_direct_formal_seed11",
    "C_coef": "C_coef_formal_seed11",
    "D_anchored_quadratic": "D_anchored_quadratic_clean2_seed11",
    "D_anchored_tlogT": "D_anchored_tlogT_formal_seed11",
    "D_anchored_cubic": "D_anchored_cubic_clean2_seed11",
}

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
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dp", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    sio2 = root / "sio2"
    env = os.environ.copy()
    env["PYTHONPATH"] = args.repo
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    for method in args.methods:
        source_name = SOURCES[method]
        source_config = sio2 / source_name / "config.json"
        source = json.loads(source_config.read_text())
        for seed in args.seeds:
            name = f"{method}_seed{seed}"
            out = sio2 / name
            out.mkdir(parents=True, exist_ok=True)
            config = rewrite(source, source_name, name)
            config["training"]["seed"] = seed
            config["model"]["fitting_net"]["seed"] = seed
            config["training"]["save_ckpt"] = str(out / "model.ckpt")
            config["training"]["disp_file"] = str(out / "train.out")
            config["training"]["stat_file"] = str(out / "fes_stats.hdf5")
            (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (out / "queue_meta.json").write_text(json.dumps({
                "method": method, "seed": seed, "source_config": str(source_config),
                "protocol": "revision2157", "distributed_training": False,
            }, indent=2) + "\n")
            log = (out / "console_queue.log").open("w")
            result = subprocess.run(
                [args.dp, "--pt", "train", "config.json"],
                cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                check=False,
            )
            log.close()
            (out / "exit_code.txt").write_text(str(result.returncode) + "\n")
            if result.returncode:
                (out / "failure.json").write_text(json.dumps({
                    "method": method, "seed": seed, "exit_code": result.returncode,
                    "log": str(out / "console_queue.log"),
                }, indent=2) + "\n")
                return result.returncode
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

