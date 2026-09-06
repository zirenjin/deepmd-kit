# Cross-material benchmark audit

## Candidate: Ti bcc/hcp

- Source: [Calorine free-energy tutorial data](https://zenodo.org/records/21198312/files/free_energy_calculations.zip)
- Public artifacts: YAML free-energy tables, reversible-scaling/TI CSV files, input files, and structures.
- Common grid: 700--1400 K in 50 K steps.
- bcc-hcp crossing from the supplied NEP-level TI tables: approximately 1139.4 K.
- Assessment: reproducible secondary diagnostic with a useful 1000--1400 K overlap with SiO2, but **not DFT-level**. It must not be presented as the primary DFT cross-material benchmark.
- Local audit artifacts: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/external/titanium_calorine/` and `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/reference/titanium_nep_ti/`.

## Rejected for the primary benchmark

- MgSiO3 and HfO2 from the finite-temperature crystal-structure-prediction work: useful temperature ranges, but the numerical data are available on request rather than as an open download.
- Carbon polymorph data: the public supplement contains structural/energy tables and plotted curves, but no machine-readable finite-temperature free-energy table.
- NPT-TI public examples: reproducible, but their temperature ranges are mostly below the SiO2 overlap regime.
- Fe: current public extraction does not provide a clean synchronous bcc/fcc/hcp grid for this protocol.

The primary Task B benchmark remains open until a public numerical DFT/QHA/TI dataset with overlapping thermodynamic range is verified.
