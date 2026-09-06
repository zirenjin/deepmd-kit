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
