## Full-curve evaluation: mean/std/max no-volume FES

Model: DPA3.1 frozen PES + piecewise-linear FES correction with aggregate descriptor mean/std/max phase pooling and no explicit volume input.

- Reference data: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/fes/full_curve/`
- Temperature range: `851-2499 K`, 1649 points per phase
- Full-range paired ΔG MAE: `3.499684e-4 eV/atom`
- Full-range paired ΔG RMSE: `6.918927e-4 eV/atom`
- Original target window `1400-1700 K` paired ΔG MAE: `8.924880e-5 eV/atom`
- Original target window paired ΔG RMSE: `9.412693e-5 eV/atom`
- Predicted crossing: `1530.18 K`; DFT-TI reference: `1542.71 K`

Interpretation: the head is strong in the intended interpolation/crossing window but is not reliable for unrestricted extrapolation to the full temperature range.
