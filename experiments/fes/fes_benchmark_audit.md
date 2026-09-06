# Cross-material benchmark audit

## Candidate: Hf hcp/bcc (strong primary candidate)

- Source: [DaRUS dataset 10.18419/DARUS-3582](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2FDARUS-3582).
- Public artifacts: machine-readable PBE Gibbs tables for Hf hcp and bcc, plus public Hf hcp/bcc structure and effective-QH archives.
- Common grid: 1252 synchronized points, 1076--2327 K, at 0.1 MPa.
- bcc-hcp crossing from the supplied tables: approximately 1920 K by linear interpolation.
- Assessment: strongest current candidate for the primary DFT-level cross-material diagnostic because its thermodynamic range overlaps the SiO2 evaluation range and the reference includes a real crossing.
- Local audit artifacts: `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/external/darus_ti_zr_hf/` and `/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj/reference/hf_hcp_bcc_pbe/`.
- Status: thermodynamic tables verified; static-structure archive download and representative-structure extraction still pending.

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
