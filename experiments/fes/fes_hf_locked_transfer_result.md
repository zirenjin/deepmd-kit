# Hf Locked Cross-Material Transfer

## Scope

This is a locked transfer diagnostic for the public PBE Hf hcp/bcc free-energy tables. It uses the existing SiO2 piecewise FES head without changing its architecture or hyperparameters.

- Reference: Hf hcp and bcc Gibbs free energies from the machine-readable DaRUS tables.
- Thermodynamic grid: 1252 synchronized points, 1076-2327 K, 0.1 MPa.
- Fixed representatives: one static initial crystal frame per phase; no temperature-dependent structures or MD snapshots are used at inference.
- DPA: frozen DPA-3.1-3M, `Domains_Alloy` branch.
- Head: frozen DPA descriptor, mean/std/max pooling, no explicit volume correction, paired phase-relative objective, piecewise knots `[1000, 1200, 1900, 2200] K`.
- Labels: one-sided temperature selections, indices `[0]`, `[0,200]`, `[0,100,200,300]`, and `[0,50,100,150,200,250,300,350]`.
- Seeds: 11, 23, 37, 51, 67.
- Evaluation: held-out points exclude selected labels; crossing detection treats the reference zero plateau as one crossing.

## Aggregate Results

| Labels | Delta-G MAE (eV/atom) | Delta-G RMSE (eV/atom) | Sign accuracy | Ranking accuracy | Predicted crossing behavior |
|---:|---:|---:|---:|---:|---|
| 1 | 0.100585 +/- 0.000000 | 0.115414 +/- 0.000000 | 0.6723 | 0.6747 | Missing in all 5 seeds |
| 2 | 0.044113 +/- 0.000000 | 0.071250 +/- 0.000000 | 0.6720 | 0.6744 | Missing in all 5 seeds |
| 4 | 0.042051 +/- 0.001938 | 0.071290 +/- 0.001126 | 0.6715 | 0.6739 | Missing in all 5 seeds |
| 8 | 0.037276 +/- 0.000076 | 0.070770 +/- 0.000166 | 0.6207 +/- 0.0174 | 0.6232 +/- 0.0174 | False crossings near 1823-1872 K and 1902-1906 K; reference crossing is 1919 K |

The reference crossing is the zero plateau centered at 1919 K. The old SiO2-specific temperature basis does not transfer reliably to Hf: increasing the number of one-sided labels reduces MAE but does not recover the transition, and the 8-label setting introduces multiple spurious crossings.

This result is a diagnostic, not evidence that DPA structural representations lack transferable information. It tests a material-specific piecewise temperature parameterization under material OOD.

## Reproducibility Paths

- Raw per-seed results: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/reference/hf_hcp_bcc_pbe/hf_fewshot_results.json`
- Hf reference: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/reference/hf_hcp_bcc_pbe/free_energy_common.csv`
- Static-only dataset: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/fes_static_only_hf_hcp_bcc_pbe`
- Configs: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/hf_fes_fewshot{1,2,4,8}_seed{11,23,37,51,67}.json`
- Evaluator: `scripts/evaluate_hf_fewshot.py`
