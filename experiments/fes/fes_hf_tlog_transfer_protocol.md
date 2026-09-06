# Hf Continuous T-Log Transfer Protocol

This is the follow-up to the locked Hf piecewise transfer diagnostic. It changes only the temperature parameterization to the system-independent continuous `tlog_polynomial` basis; DPA, pooling, loss, optimizer, steps, labels, and seeds are unchanged.

- Basis: `a + b*tau + c*tau*log(tau) + d*tau^2`, `tau = T / 1000 K`.
- No dataset-specific temperature knots.
- Frozen DPA-3.1-3M, `Domains_Alloy` branch.
- Fixed static Hf hcp/bcc representatives.
- PBE reference: 1252 synchronized points, 1076-2327 K, 0.1 MPa.
- Label counts: 1, 2, 4, 8; seeds: 11, 23, 37, 51, 67.
- One-sided labels remain unchanged from the piecewise diagnostic.
- Training configs: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/hf_tlog_fes_fewshot{1,2,4,8}_seed{11,23,37,51,67}.json`.
- Batch launcher: `/dev/shm/run_hf_tlog_all.sh`.
- Output root: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/hf_tlog_fes_fewshot{1,2,4,8}_seed{11,23,37,51,67}`.

This is a controlled parameterization-transfer experiment, not a new architecture search.
