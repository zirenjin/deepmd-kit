# Few-shot phase-relative free-energy benchmark

This directory defines the benchmark for **few-shot adaptation of pretrained
interatomic representations for phase-relative free energies**. It is separate
from the historical single-material SiO2 head sweep.

## Protocol

- A phase is represented by one fixed relaxed structure `X0` and one cached DPA
  representation. The same `X0` is used at every temperature and pressure.
- The only state variables supplied to an adapter are `T` and `P`.
- MD frames, thermal snapshots, equilibrium-volume structures, and
  temperature-dependent relaxed structures are forbidden in the formal test.
- A material/phase pair is held out completely from adapter pretraining. Its
  reference labels are exposed only at `0, 1, 2, 4, or 8` adaptation points.
- Every few-shot result uses fixed, published label-selection seeds and reports
  the complete curve evaluation, not only the sampled points.

The primary output is a label-efficiency curve: number of expensive free-energy
labels versus `Delta G` MAE/RMSE and transition-temperature error.

## Candidate registry

The machine-readable registry is `manifest.yaml`. Candidate status is explicit:
`ready` means structures and a continuous reference curve have been staged;
`stage-1` means the reference is public but needs extraction/normalization;
`boundary-only` means it is useful for pressure/phase-boundary evaluation but
does not yet meet the continuous-curve requirement.

## Methods

The formal structured method is the existing SiO2 best configuration:

`frozen DPA PES -> [mean, std, max] phase pooling -> piecewise temperature basis -> gauge-free paired Delta G correction`

No architecture search is part of this benchmark. Required comparisons are
linear interpolation/spline, phase-ID plus `T/P` MLP, native DeepProperty,
pooled-DPA MLP, and the structured FES head. The fixed-structure protocol is
the only formal score; temperature-dependent structures are a leakage
diagnostic only.

## Source datasets

- SiO2: Forslund et al., DaRUS-4999.
- WB, HfO2, MgSiO3, and Al: T-USPEX finite-temperature crystal-structure
  prediction data and supplementary tables.
- Ice Ih/Ic: Materials Cloud Archive 2018.0020, including free-energy workflow
  inputs and the revPBE0-D3 reference data.

The source publications use thermodynamic integration and/or thermodynamic
perturbation corrections. Their MD snapshots are reference-generation data and
are never used as the adapter input.
