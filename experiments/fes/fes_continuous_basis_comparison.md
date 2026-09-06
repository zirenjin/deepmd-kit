# SiO2 Continuous Temperature-Basis Comparison

This controlled comparison uses the frozen DPA-3.1 descriptor, fixed static representatives, four phases, the paired gauge-free objective, identical training data and optimizer settings, and five seeds (11, 23, 37, 51, 67). Evaluation uses 1649 temperatures at 0 GPa.

| Basis | Seeds | Mean all-pair MAE (eV/atom) | Mean all-pair RMSE (eV/atom) | Ranking accuracy | Cycle max (eV/atom) |
|---|---:|---:|---:|---:|---:|
| Existing piecewise knots (reference) | 1 | 0.013100 | 0.014247 | 0.423 | 0.00e+00 |
| Continuous cubic polynomial | 5 | 0.024253 | 0.028343 | 0.232 | 0.00e+00 |
| Continuous T-log polynomial | 5 | 0.015743 | 0.018317 | 0.095 | 0.00e+00 |

The continuous cubic basis is seed-stable but weaker in-domain than the existing piecewise reference. The T-log basis lowers average pairwise error relative to the cubic basis, but has worse phase-ranking accuracy. Both continuous bases preserve exact scalar-energy cycle consistency. The piecewise reference is a one-seed comparison and is not a matched five-seed baseline.

Persistent artifacts are under `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/results/continuous_sio2/`, including per-seed configs, frozen models, metrics, `controlled_comparison.json`, `summary.json`, and `manifest.json`.
