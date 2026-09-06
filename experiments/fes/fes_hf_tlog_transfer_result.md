# Hf Continuous T-Log Transfer Results

## Aggregate (5 seeds)

| Labels | MAE (eV/atom) | RMSE (eV/atom) | Sign accuracy | Ranking accuracy | Predicted crossing (K) |
|---:|---:|---:|---:|---:|---|
| 1 | 0.078755 +/- 0.000001 | 0.097901 +/- 0.000001 | 0.5324 | 0.5324 | 1335.4 +/- 0.0 |
| 2 | 0.024668 +/- 0.000001 | 0.033155 +/- 0.000001 | 0.7272 | 0.7272 | 1579.6 +/- 0.0 |
| 4 | 0.016771 +/- 0.000002 | 0.023038 +/- 0.000003 | 0.7893 | 0.7893 | 1657.4 +/- 0.0 |
| 8 | 0.004641 +/- 0.000166 | 0.006213 +/- 0.000256 | 0.9613 +/- 0.0045 | 0.9613 +/- 0.0045 | 1872.5 +/- 5.8 |

Reference crossing: 1919 K (zero plateau). The 8-label crossing error is approximately 46.5 K on average.

## Comparison With Locked Piecewise Head

| Labels | Piecewise MAE | T-log MAE | Piecewise crossing | T-log crossing |
|---:|---:|---:|---|---|
| 1 | 0.100585 | 0.078755 | missing (5/5) | 1335 K (5/5) |
| 2 | 0.044113 | 0.024668 | missing (5/5) | 1580 K (5/5) |
| 4 | 0.042051 | 0.016771 | missing (5/5) | 1657 K (5/5) |
| 8 | 0.037276 | 0.004641 | false crossings near 1823-1906 K | 1866-1879 K (5/5) |

The system-independent continuous T-log basis substantially improves material transfer on Hf, especially in the 8-label setting. It does not fully remove the cross-material error: the transition is systematically underpredicted by about 40-53 K. The result supports the interpretation that the old SiO2-specific knot parameterization was a major failure mode, while leaving a residual question about transfer of the thermodynamic correction itself.

## Reproducibility

- Raw results: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/reference/hf_hcp_bcc_pbe/hf_tlog_fewshot_results.json`
- Configs: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/hf_tlog_fes_fewshot{1,2,4,8}_seed{11,23,37,51,67}.json`
- Checkpoint root: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/hf_tlog_fes_fewshot{1,2,4,8}_seed{11,23,37,51,67}`
- Evaluator: `scripts/evaluate_hf_tlog_fewshot.py`
- Reference: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/reference/hf_hcp_bcc_pbe/free_energy_common.csv`


## Interpolation Baseline

Using the same one-sided labels and piecewise linear interpolation with constant extrapolation outside the labeled range:

| Labels | Interpolation MAE (eV/atom) | Interpolation RMSE (eV/atom) | Sign accuracy | Crossing |
|---:|---:|---:|---:|---|
| 1 | 0.023835 | 0.026673 | 0.6723 | Missing |
| 2 | 0.015381 | 0.018799 | 0.6720 | Missing |
| 4 | 0.011958 | 0.015383 | 0.6715 | Missing |
| 8 | 0.010584 | 0.013948 | 0.6704 | Missing |

The continuous T-log head is worse than this interpolation baseline at 1, 2, and 4 labels, but is better at 8 labels and is the only method here that produces a single crossing in all 5 seeds. The result should therefore be interpreted as partial transfer, not a general few-shot win over interpolation.


## Phase-ID + Temperature MLP Baseline

A two-layer MLP receives only normalized temperature and a binary phase ID; it does not receive DPA descriptors or structure information. It is trained on the same absolute G labels and evaluated on the same held-out grid.

| Labels | MAE (eV/atom) | RMSE (eV/atom) | Sign / ranking accuracy | Predicted crossing range (K) |
|---:|---:|---:|---:|---|
| 1 | 0.028589 | 0.031599 | 67.5% | one seed: 1526 K; four missing |
| 2 | 0.006810 | 0.008741 | 83.9% | 1658-1847 K |
| 4 | 0.008366 | 0.011280 | 82.5% | 1662-1728 K |
| 8 | 0.008149 | 0.011197 | 83.5% | 1672-1743 K |

The phase-ID baseline outperforms the T-log FES at 2 and 4 labels, but not at 8 labels. Its predicted crossings are less stable and remain systematically below the 1919 K reference. This baseline is a temperature/function-fitting control rather than evidence of transferable structural information.


## Simple Pooled-DPA MLP Baseline

The baseline uses the frozen DPA descriptor exported by `dp embed`, pooled per phase with `[mean, std, max]`, plus normalized temperature and a binary phase indicator. It has no FES-specific piecewise or T-log basis and is trained directly on absolute G labels.

| Labels | MAE (eV/atom) | RMSE (eV/atom) | Sign / ranking accuracy | Predicted crossing range (K) |
|---:|---:|---:|---:|---|
| 1 | 0.024943 | 0.027965 | 67.2% | Missing (5/5) |
| 2 | 0.016557 | 0.018117 | 66.6% | 1484-1723 K; one missing |
| 4 | 0.018674 | 0.021529 | 67.2% | 1502-1711 K; one missing |
| 8 | 0.009546 | 0.011038 | 89.7% | 1671-2007 K |

At 8 labels this simple pooled-DPA baseline is better than interpolation and phase-ID + T in crossing coverage, but remains worse than the structured continuous T-log FES (MAE 0.004641, sign/ranking accuracy 96.1%). The comparison isolates the value of the structured thermodynamic parameterization from the value of the DPA representation alone.
## Native DeepProperty Baseline

Native DeepProperty was trained as an absolute property model on the same fixed Hf structures, temperature inputs, label subsets, validation grid, DPA checkpoint, and five seeds. It uses `intensive=false` so the target is total free energy; errors below are reported in eV/atom.

| Labels | MAE (eV/atom) | RMSE (eV/atom) | Sign accuracy | Ranking accuracy | First crossing error (K) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.011383 +/- 0.000000 | 0.014492 +/- 0.000000 | 8.3% +/- 7.4% | 59.6% +/- 2.5% | 840.8 +/- 0.8 |
| 2 | 0.011374 +/- 0.000000 | 0.014484 +/- 0.000000 | 33.2% +/- 26.9% | 59.0% +/- 8.8% | 824.3 +/- 23.4 |
| 4 | 0.011356 +/- 0.000000 | 0.014465 +/- 0.000000 | 27.7% +/- 21.5% | 55.5% +/- 11.5% | 705.8 +/- 244.2 |
| 8 | 0.011318 +/- 0.000000 | 0.014427 +/- 0.000000 | 36.4% +/- 20.3% | 57.0% +/- 13.4% | 693.8 +/- 239.3 |

The absolute property baseline produces many spurious crossings in the one-sided setting; the first-crossing statistic is therefore not a reliable transition estimate. Its nearly label-count-independent error and unstable signs show that absolute property fitting does not recover the phase-relative thermodynamics here. This contrasts with the paired, gauge-free structured head.

## Native DeepProperty Reproducibility

- Raw results: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/reference/hf_hcp_bcc_pbe/hf_property_fewshot_results.json`
- Per-seed results: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/reference/hf_hcp_bcc_pbe/hf_property_seed{11,23,37,51,67}.json`
- Configs: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/hf_property_fewshot{1,2,4,8}_seed{11,23,37,51,67}.json`
- Evaluator: `scripts/evaluate_hf_property_fewshot.py`
