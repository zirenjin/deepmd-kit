# SPDX-License-Identifier: LGPL-3.0-or-later
"""Atomic model for the free energy surface (FES) head."""

import functools
import logging
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import torch

from deepmd.pt.model.task.free_energy import (
    FreeEnergyFittingNet,
)
from deepmd.pt.utils.state_vector import (
    build_state_vector,
)
from deepmd.utils.path import (
    DPPath,
)

from .dp_atomic_model import (
    DPAtomicModel,
)

log = logging.getLogger(__name__)


class DPFreeEnergyAtomicModel(DPAtomicModel):
    """Atomic model wiring a :class:`FreeEnergyFittingNet` to a descriptor.

    Two behaviours differ from the standard atomic model:

    1. **No label-statistics output bias.**  ``E_DPA`` arrives fully formed from
       the pre-trained checkpoint (its per-type offset is folded into the
       baseline's ``bias_atom_e`` when finetuning), and the correction net
       learns its own offset.  Fitting an additional bias against the ``G``
       labels would double-count both.
    2. **Augmented fparam statistics.**  ``fparam.npy`` holds only the
       externally supplied ``[T, P]``, while the correction net is conditioned
       on the full ``[T, P, v, c]`` state vector.  The statistics sampler is
       wrapped so ``fparam_avg``/``fparam_inv_std`` are computed over the same
       vector the net actually sees at train time.
    """

    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any, **kwargs: Any
    ) -> None:
        if not isinstance(fitting, FreeEnergyFittingNet):
            raise TypeError(
                "fitting must be an instance of FreeEnergyFittingNet for "
                "DPFreeEnergyAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def get_intensive(self) -> bool:
        """G is extensive; per-atom outputs are summed."""
        return False

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the atomic outputs untouched.

        The per-type offsets already live inside the two sub-fitting nets
        (``baseline.bias_atom_e`` and ``correction.bias_atom_e``), so the
        atomic-model level ``out_bias`` stays zero for this model.
        """
        return ret

    def change_out_bias(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
        bias_adjust_mode: str = "change-by-statistic",
    ) -> None:
        """No-op: see :meth:`apply_out_stat`.

        Also keeps ``dp train --finetune`` from overwriting the loaded
        pre-trained baseline with a bias fitted against free-energy labels.
        """
        return None

    def _augment_state_fparam(self, sampled: list[dict]) -> list[dict]:
        """Return copies of *sampled* whose ``fparam`` is the ``[T, P, v, c]`` vector.

        Shallow-copies each sample rather than mutating in place: the same
        sampled list is shared with the descriptor statistics, and some
        descriptors (DPA3 with charge/spin embedding) read ``fparam`` for their
        own purposes.
        """
        fitting = self.fitting_net
        augmented = []
        for sample in sampled:
            if sample.get("box") is None and fitting.volume_mode != "none":
                raise ValueError(
                    "the FES head needs a simulation box to derive the cell "
                    'volume; set volume_mode="none" for non-periodic data.'
                )
            new_sample = dict(sample)
            new_sample["fparam"] = build_state_vector(
                sample.get("box"),
                sample["atype"],
                sample.get("fparam"),
                fitting.ntypes,
                volume_mode=fitting.volume_mode,
                use_composition=fitting.use_composition,
                numb_state_fparam=fitting.numb_state_fparam,
            )
            augmented.append(new_sample)
        return augmented

    def compute_fitting_input_stat(
        self,
        sample_merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
    ) -> None:
        if callable(sample_merged):

            @functools.lru_cache
            def augmented() -> list[dict]:
                return self._augment_state_fparam(sample_merged())

            merged: Callable[[], list[dict]] | list[dict] = augmented
        else:
            merged = self._augment_state_fparam(sample_merged)
        super().compute_fitting_input_stat(merged, stat_file_path)
