# Matched five-seed piecewise baseline

This records the matched SiO2 four-phase static-only baseline evaluated with the
same data, labels, 3000 training steps, and five seeds as the continuous-basis
runs.

- Remote run root: `/dev/shm/fes_sio2_piecewise`
- Persistent metrics: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/results/continuous_sio2/piecewise/`
- DPA checkpoint: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/models/DPA-3.1-3M.pt`
- Phases: quartz_beta, cristobalite_beta, tridymite_p63mmc, tridymite_c2221
- Seeds: 11, 23, 37, 51, 67
- Static-only evaluation: 1649 temperatures, 0 GPa; all pairwise differences are formed from scalar phase predictions.

| metric | mean | std |
|---|---:|---:|
| all-pair MAE (eV/atom) | 0.0154027 | 0.0187197 |
| all-pair RMSE (eV/atom) | 0.0167959 | 0.0201672 |
| phase-ranking accuracy | 0.358035 | 0.338517 |
| cycle inconsistency max (eV/atom) | 0 | 0 |

Per-pair means:

| pair | MAE (eV/atom) | RMSE (eV/atom) | sign accuracy |
|---|---:|---:|---:|
| quartz-cristobalite | 0.00115371 | 0.00136438 | 0.8500 |
| quartz-tridymite-P63/mmc | 0.00103970 | 0.00131553 | 0.8164 |
| quartz-tridymite-C2221 | 0.0299884 | 0.0325782 | 0.5757 |
| cristobalite-tridymite-P63/mmc | 0.000509975 | 0.000550047 | 0.7510 |
| cristobalite-tridymite-C2221 | 0.0296339 | 0.0322393 | 0.5578 |
| tridymite-P63/mmc-tridymite-C2221 | 0.0300905 | 0.0327279 | 0.5572 |

The previous aggregate script expected top-level keys that the evaluator does
not emit; `summary_corrected.json` is the authoritative recomputation from
the per-seed pair metrics. The zero cycle inconsistency is an exact consequence
of deriving every pair from one scalar prediction per phase.
