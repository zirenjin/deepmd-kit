# Static-only four-phase SiO2 FES

This is the corrected four-phase run using the PBE-D3(BJ) DaRUS tables. The
tables are in eV/atom and are converted to extensive `free_energy.npy` labels
because DeepMD energy labels are per structure. The model is the frozen DPA3.1
descriptor with mean/std/max phase pooling, piecewise-linear temperature basis,
no explicit volume in the correction, and a quartz gauge for the paired loss.

All 1649 temperatures use one fixed coordinate/cell/type array per phase. The
static input audit is stored in `static_protocol.json`; no temperature-dependent
structure is passed to the model. The C2221 representative is the public COD
structure used by the builder, while the other three representatives are fixed
single OUTCAR geometries from the DaRUS training archive. This provenance is
kept explicit and should not be described as a 0 K relaxation.

## All-pair evaluation

| Pair | MAE (eV/atom) | RMSE (eV/atom) | Sign acc. | Reference crossing (K) | Predicted crossing(s) (K) |
| --- | ---: | ---: | ---: | ---: | --- |
| quartz-cristobalite | 0.001411 | 0.001916 | 74.4% | 1542.7 | 1964.6 |
| quartz-tridymite P63/mmc | 0.001142 | 0.001558 | 77.4% | 1582.3 | 1954.3 |
| quartz-tridymite C2221 | 0.025850 | 0.028027 | 92.8% | 2229.6 | 2111.3 |
| cristobalite-tridymite P63/mmc | 0.000374 | 0.000427 | 34.8% | none | 1049.0, 2124.8 |
| cristobalite-tridymite C2221 | 0.024763 | 0.026595 | 77.6% | none | 2129.8 |
| tridymite P63/mmc-C2221 | 0.025060 | 0.026961 | 77.6% | none | 2129.7 |

Phase-ranking accuracy over the full common range is 42.3%. The reference
stable sequence is quartz -> cristobalite; the predicted sequence is
quartz -> cristobalite -> P63/mmc tridymite -> C2221 tridymite. Scalar phase
outputs give exactly zero cycle inconsistency at floating-point precision.

Artifacts:

- data: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/fes_static_only_4phase_extensive`
- checkpoint: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/fes_static_only_4phase_extensive_a100/model.ckpt-3000.pt`
- frozen model: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/fes_static_only_4phase_extensive_a100/fes_static_only_4phase_extensive.pth`
- metrics: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/fes_static_only_4phase_extensive_a100/all_pair_metrics.json`
