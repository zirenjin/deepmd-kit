# SPDX-License-Identifier: LGPL-3.0-or-later
"""Loss for the free energy surface (FES) head."""

import logging
from typing import (
    Any,
)

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.model.task.free_energy import (
    BASELINE_NAME,
    CORRECTION_NAME,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

log = logging.getLogger(__name__)


class FreeEnergyLoss(TaskLoss):
    """Supervise the predicted Gibbs free energy per atom.

    ``G`` is extensive, so both prediction and label are divided by the real
    atom count before the loss is taken.  That matches the quantity the project
    ultimately cares about -- the per-atom free energies whose difference gives
    ``dG = G_a/N_a - G_b/N_b`` and hence the transition temperature -- and keeps
    systems of different size on the same footing.

    Two diagnostics are reported alongside the metrics and do **not** enter the
    optimized loss:

    ``baseline_mae``
        Error of the frozen ``E_DPA`` alone against the free energy label.  The
        number the correction has to beat.
    ``correction_rms``
        RMS magnitude of the learned per-atom correction.  A value stuck near
        zero means the head has collapsed onto the baseline.
    """

    def __init__(
        self,
        var_name: str = "free_energy",
        task_dim: int = 1,
        loss_func: str = "mse",
        metric: list[str] | None = None,
        beta: float = 1.00,
        delta_g_pref: float = 0.0,
        **kwargs: Any,
    ) -> None:
        r"""Construct a layer to compute the free energy loss.

        Parameters
        ----------
        var_name : str
            Name of the fitted property; must match the label file.
        task_dim : int
            Output dimension of the FES fitting net (1).
        loss_func : str
            One of "mse", "mae", "smooth_mae", "rmse".
        metric : list[str]
            Extra metrics to print, from "mae", "mse", "rmse", "smooth_mae".
        beta : float
            The 'beta' parameter of "smooth_mae".
        delta_g_pref : float
            Weight of the paired per-atom ``dG`` objective. Requires a
            ``pair_systems`` data loader, which emits phase A followed by phase
            B in each batch. The target is ``G_B/N_B - G_A/N_A``.
        """
        super().__init__()
        self.var_name = var_name
        self.task_dim = task_dim
        self.loss_func = loss_func
        self.metric = list(metric) if metric is not None else ["mae", "rmse"]
        self.beta = beta
        self.delta_g_pref = delta_g_pref

    @staticmethod
    def _per_atom(
        value: torch.Tensor,
        model_pred: dict[str, torch.Tensor],
        natoms: int,
    ) -> torch.Tensor:
        if "mask" in model_pred:
            real_natoms = (
                torch.sum(model_pred["mask"], dim=-1)
                .to(dtype=value.dtype)
                .reshape(-1, 1)
            )
            return value / real_natoms
        return value / natoms

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float = 0.0,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Return the free energy loss.

        Parameters
        ----------
        input_dict : dict[str, torch.Tensor]
            Model inputs.
        model : torch.nn.Module
            Model to be used to output the predictions.
        label : dict[str, torch.Tensor]
            Labels.
        natoms : int
            The local atom number.
        learning_rate : float
            Unused; kept for the common loss signature.
        mae : bool
            Unused; the reported metrics are selected via ``metric``.

        Returns
        -------
        model_pred : dict[str, torch.Tensor]
            Model predictions.
        loss : torch.Tensor
            Loss for model to minimize.
        more_loss : dict[str, torch.Tensor]
            Other losses for display.
        """
        model_pred = self._inject_atom_mask(model(**input_dict), input_dict)
        var_name = self.var_name
        nbz = model_pred[var_name].shape[0]
        if model_pred[var_name].shape != (nbz, self.task_dim):
            raise ValueError(
                f"expected prediction of shape {(nbz, self.task_dim)}, got "
                f"{tuple(model_pred[var_name].shape)}."
            )
        if label[var_name].shape != (nbz, self.task_dim):
            raise ValueError(
                f"expected label {var_name!r} of shape {(nbz, self.task_dim)}, "
                f"got {tuple(label[var_name].shape)}."
            )

        pred = self._per_atom(model_pred[var_name], model_pred, natoms)
        ref = self._per_atom(label[var_name], model_pred, natoms)

        if self.loss_func == "mse":
            loss = F.mse_loss(pred, ref, reduction="mean")
        elif self.loss_func == "mae":
            loss = F.l1_loss(pred, ref, reduction="mean")
        elif self.loss_func == "smooth_mae":
            loss = F.smooth_l1_loss(pred, ref, reduction="mean", beta=self.beta)
        elif self.loss_func == "rmse":
            loss = torch.sqrt(F.mse_loss(pred, ref, reduction="mean"))
        else:
            raise RuntimeError(f"Unknown loss function : {self.loss_func}")

        more_loss = {}
        if self.delta_g_pref > 0.0:
            if "pair_batch_size" not in label:
                raise ValueError(
                    "delta_g_pref > 0 requires training_data.pair_systems"
                )
            pair_batch_size = int(label["pair_batch_size"].reshape(-1)[0].item())
            if pred.shape[0] != 2 * pair_batch_size:
                raise ValueError(
                    "paired FES batch must contain phase A followed by phase B"
                )
            delta_pred = pred[pair_batch_size:] - pred[:pair_batch_size]
            delta_ref = ref[pair_batch_size:] - ref[:pair_batch_size]
            delta_loss = F.mse_loss(delta_pred, delta_ref, reduction="mean")
            loss = loss + self.delta_g_pref * delta_loss
            more_loss["delta_mae"] = F.l1_loss(
                delta_pred, delta_ref, reduction="mean"
            ).detach()
            more_loss["delta_rmse"] = torch.sqrt(delta_loss).detach()
        if "mae" in self.metric:
            more_loss["mae"] = F.l1_loss(pred, ref, reduction="mean").detach()
        if "mse" in self.metric:
            more_loss["mse"] = F.mse_loss(pred, ref, reduction="mean").detach()
        if "rmse" in self.metric:
            more_loss["rmse"] = torch.sqrt(
                F.mse_loss(pred, ref, reduction="mean")
            ).detach()
        if "smooth_mae" in self.metric:
            more_loss["smooth_mae"] = F.smooth_l1_loss(
                pred, ref, reduction="mean", beta=self.beta
            ).detach()

        # Diagnostics: how far the frozen baseline already gets, and whether the
        # correction is doing anything at all.
        if BASELINE_NAME in model_pred:
            baseline = self._per_atom(model_pred[BASELINE_NAME], model_pred, natoms)
            more_loss["baseline_mae"] = F.l1_loss(
                baseline, ref, reduction="mean"
            ).detach()
        if CORRECTION_NAME in model_pred:
            correction = self._per_atom(model_pred[CORRECTION_NAME], model_pred, natoms)
            more_loss["correction_rms"] = torch.sqrt(torch.mean(correction**2)).detach()

        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        return [
            DataRequirementItem(
                self.var_name,
                ndof=self.task_dim,
                atomic=False,
                must=True,
                high_prec=True,
            )
        ]

    def serialize(self) -> dict:
        """Serialize the loss module."""
        return {
            "@class": "FreeEnergyLoss",
            "@version": 1,
            "var_name": self.var_name,
            "task_dim": self.task_dim,
            "loss_func": self.loss_func,
            "metric": self.metric,
            "beta": self.beta,
            "delta_g_pref": self.delta_g_pref,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "FreeEnergyLoss":
        """Deserialize the loss module."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        return cls(**data)


__all__ = ["FreeEnergyLoss"]
