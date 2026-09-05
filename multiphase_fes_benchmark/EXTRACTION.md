# Reference extraction checklist

For each candidate, create a directory with:

```text
structures/<phase>/representative.extxyz
reference/curve.csv          # T_K, P_GPa, G_phase_eV_per_atom or delta_G_eV_per_atom
metadata.json                # source, units, method, hashes, representative provenance
splits/label_points_seed*.json
```

The representative provenance must state that the structure is a single
relaxed/ideal branch representative and must not be selected from a finite-T
trajectory. If the source only provides equilibrium structures at different
temperatures, it cannot be used for the formal static-only score.

Before training, validate:

1. both phase structures are fixed byte-for-byte over all state points;
2. composition and atom counts are recorded;
3. `G` units and per-atom normalization are explicit;
4. reference curves have at least four candidate adaptation points and a
   disjoint evaluation grid;
5. the transition root is bracketed, or the case is labeled no-crossing and is
   evaluated with sign accuracy instead of an invented `Tc`.
