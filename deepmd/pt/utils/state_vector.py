# SPDX-License-Identifier: LGPL-3.0-or-later
"""Thermodynamic state vector for the free energy surface (FES) head.

The FES head conditions its per-atom correction on a frame-level state vector

    s = [T, P, v(X), c(X)]

where ``T`` and ``P`` are supplied externally through ``fparam.npy`` while the
specific volume ``v`` and the composition ``c`` are *functions of the structure*
and are therefore derived here from ``box`` and ``atype`` rather than being
asked of the user.

``box`` reaches the model in ``forward_common`` but is dropped before
``forward_atomic``, so the augmentation happens at the model level and the
fitting net only ever sees a finished state vector.  The very same builder is
used by the data-statistics path (so ``fparam_avg``/``fparam_inv_std`` are
computed over the augmented vector) and by downstream inference, which keeps the
three code paths bit-for-bit consistent.
"""

import torch

VOLUME_MODES = ("per_atom", "total", "both", "none")


def volume_dim(volume_mode: str) -> int:
    """Number of volume columns contributed by ``volume_mode``."""
    if volume_mode == "none":
        return 0
    if volume_mode in ("per_atom", "total"):
        return 1
    if volume_mode == "both":
        return 2
    raise ValueError(
        f"volume_mode must be one of {list(VOLUME_MODES)}; got {volume_mode!r}."
    )


def state_vector_dim(
    numb_state_fparam: int,
    ntypes: int,
    volume_mode: str = "per_atom",
    use_composition: bool = True,
) -> int:
    """Full width of the state vector seen by the fitting net."""
    return (
        int(numb_state_fparam)
        + volume_dim(volume_mode)
        + (int(ntypes) if use_composition else 0)
    )


def cell_volume(box: torch.Tensor) -> torch.Tensor:
    """Absolute cell volume from a ``nf x 9`` or ``nf x 3 x 3`` box.

    Uses the scalar triple product rather than ``torch.linalg.det`` so the whole
    builder stays TorchScript-friendly for ``dp freeze``.
    """
    bb = box.reshape(-1, 3, 3)
    return torch.abs(
        (torch.cross(bb[:, 0, :], bb[:, 1, :], dim=-1) * bb[:, 2, :]).sum(dim=-1)
    )


def build_state_vector(
    box: torch.Tensor | None,
    atype: torch.Tensor,
    fparam: torch.Tensor | None,
    ntypes: int,
    volume_mode: str = "per_atom",
    use_composition: bool = True,
    numb_state_fparam: int = 0,
) -> torch.Tensor:
    """Assemble ``[T, P, v, c]`` for every frame.

    Parameters
    ----------
    box
        Simulation cell, ``nf x 9`` or ``nf x 3 x 3``.  Required unless
        ``volume_mode`` is ``"none"``.
    atype
        Atom types, ``nf x natoms``.  Negative entries mark virtual/padding
        atoms and are excluded from both the atom count and the composition.
    fparam
        Externally supplied frame parameters, ``nf x numb_state_fparam``
        (``[T, P]`` by default).  May be ``None`` when ``numb_state_fparam`` is 0.
    ntypes
        Number of atom types; sets the width of the composition block.
    volume_mode
        ``"per_atom"`` (V/N, the default and the intensive choice), ``"total"``
        (V), ``"both"``, or ``"none"``.
    use_composition
        Whether to append the per-type fractions.
    numb_state_fparam
        Expected width of ``fparam``; validated when > 0.

    Returns
    -------
    torch.Tensor
        ``nf x state_vector_dim(...)``, in the model's global float precision.
    """
    nf = atype.shape[0]

    blocks: list[torch.Tensor] = []
    dtype = torch.float64
    device = atype.device
    if fparam is not None:
        dtype = fparam.dtype
        device = fparam.device
    elif box is not None:
        dtype = box.dtype
        device = box.device

    atype_work = atype.to(device=device)
    real = atype_work >= 0
    n_real = real.sum(dim=1)

    if numb_state_fparam > 0:
        if fparam is None:
            raise ValueError(
                f"the FES head needs {numb_state_fparam} frame parameter(s) "
                "(T, P by default) but fparam is None; provide fparam.npy or "
                "set default_fparam."
            )
        fparam = fparam.reshape(nf, -1)
        if fparam.shape[1] != numb_state_fparam:
            raise ValueError(
                f"expected {numb_state_fparam} fparam column(s), got {fparam.shape[1]}."
            )
        blocks.append(fparam.to(device=device, dtype=dtype))

    n_real_f = n_real.to(device=device, dtype=dtype).clamp_min(1.0)

    if volume_mode != "none":
        if box is None:
            raise ValueError(
                "volume_mode " + volume_mode + " needs a simulation box, but "
                "box is None. Non-periodic frames have no cell volume; use "
                'volume_mode="none" for those.'
            )
        vol = cell_volume(box.to(device=device, dtype=dtype))
        if volume_mode == "per_atom":
            blocks.append((vol / n_real_f).reshape(nf, 1))
        elif volume_mode == "total":
            blocks.append(vol.reshape(nf, 1))
        elif volume_mode == "both":
            blocks.append(
                torch.stack([vol / n_real_f, vol], dim=-1).reshape(nf, 2),
            )
        else:
            raise ValueError(
                "volume_mode must be one of per_atom / total / both / none; "
                "got " + volume_mode + "."
            )

    if use_composition:
        # one_hot demands int64; the data pipeline hands out int32 in places.
        onehot = torch.nn.functional.one_hot(
            atype_work.clamp_min(0).to(torch.long), ntypes
        ).to(dtype)
        onehot = onehot * real.unsqueeze(-1).to(dtype)
        blocks.append(onehot.sum(dim=1) / n_real_f.reshape(nf, 1))

    if len(blocks) == 0:
        return torch.zeros((nf, 0), dtype=dtype, device=device)
    return torch.cat(blocks, dim=-1)
