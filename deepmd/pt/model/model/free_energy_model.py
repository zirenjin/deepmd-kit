# SPDX-License-Identifier: LGPL-3.0-or-later
"""Free energy surface (FES) model.

Assembles the thermodynamic state vector at the model level -- ``box`` reaches
``forward_common`` but is dropped before ``forward_atomic``, so the volume and
composition columns have to be derived here -- and hands the finished vector to
the FES fitting net through the ordinary ``fparam`` channel.

Consequently the head keeps two fparam widths:

* :meth:`get_dim_fparam` (2 by default) is what a *user* supplies in
  ``fparam.npy`` and what ``DeepEval`` reshapes against: ``[T, P]``.
* The fitting net's own ``numb_fparam`` is the full ``[T, P, v, c]`` width.
"""

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.output_def import (
    OutputVariableDef,
)
from deepmd.pt.model.atomic_model.free_energy_atomic_model import (
    DPFreeEnergyAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.task.free_energy import (
    BASELINE_NAME,
    CORRECTION_NAME,
)
from deepmd.pt.utils.state_vector import (
    build_state_vector,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPFreeEnergyModel_ = make_model(DPFreeEnergyAtomicModel)


@BaseModel.register("fes")
class FreeEnergyModel(DPModelCommon, DPFreeEnergyModel_):
    model_type = "fes"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPFreeEnergyModel_.__init__(self, *args, **kwargs)

    def translated_output_def(self) -> dict[str, OutputVariableDef]:
        out_def_data = self.model_output_def().get_data()
        var_name = self.get_var_name()
        output_def = {
            f"atom_{var_name}": out_def_data[var_name],
            var_name: out_def_data[f"{var_name}_redu"],
            BASELINE_NAME: out_def_data[f"{BASELINE_NAME}_redu"],
            CORRECTION_NAME: out_def_data[f"{CORRECTION_NAME}_redu"],
        }
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def

    def build_state_fparam(
        self,
        box: torch.Tensor | None,
        atype: torch.Tensor,
        fparam: torch.Tensor | None,
    ) -> torch.Tensor:
        """Expand the user's ``[T, P]`` into the full ``[T, P, v, c]`` vector."""
        fitting = self.get_fitting_net()
        numb_state_fparam = fitting.get_dim_state_fparam()
        if fparam is None and numb_state_fparam > 0:
            default = fitting.get_default_fparam()
            if default is None:
                raise ValueError(
                    f"the FES head needs {numb_state_fparam} frame parameter(s) "
                    "(T, P by default); provide fparam.npy or set default_fparam."
                )
            fparam = default.reshape(1, -1).expand(atype.shape[0], -1)
        return build_state_vector(
            box,
            atype,
            fparam,
            fitting.ntypes,
            volume_mode=fitting.volume_mode,
            use_composition=fitting.use_composition,
            numb_state_fparam=numb_state_fparam,
        )

    def _translate(self, model_ret: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Literal keys, not the module constants: TorchScript cannot close over
        # module-level strings (see FreeEnergyFittingNet.forward).
        var_name = self.get_var_name()
        model_predict = {
            "atom_" + var_name: model_ret[var_name],
            var_name: model_ret[var_name + "_redu"],
            "fes_baseline": model_ret["fes_baseline_redu"],
            "fes_correction": model_ret["fes_correction_redu"],
        }
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state_fparam = self.build_state_fparam(box, atype, fparam)
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=state_fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )
        return self._translate(model_ret)

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Lower-level forward. ``fparam`` must be the full state vector.

        There is no ``box`` at this level, so the volume and composition columns
        cannot be derived here; the caller has to pass the same vector
        :meth:`build_state_fparam` would produce.
        """
        state_dim = self.get_fitting_net().get_dim_fparam()
        if fparam is None or fparam.shape[-1] != state_dim:
            got = "None" if fparam is None else str(fparam.shape[-1])
            raise ValueError(
                f"forward_lower needs the full FES state vector of width "
                f"{state_dim} in fparam (got {got}); the cell volume cannot be "
                "derived without a box. Use forward() with a box, or build the "
                "vector with FreeEnergyModel.build_state_fparam."
            )
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
            charge_spin=charge_spin,
        )
        return self._translate(model_ret)

    @torch.jit.export
    def get_dim_fparam(self) -> int:
        """Width of ``fparam.npy``: the externally supplied ``[T, P]``.

        Overrides the ``make_model`` delegation, which would report the fitting
        net's full state-vector width and make the data pipeline demand volume
        and composition columns the user never has to write down.
        """
        return self.get_fitting_net().get_dim_state_fparam()

    @torch.jit.export
    def get_task_dim(self) -> int:
        """Output dimension of the FES fitting net."""
        return self.get_fitting_net().dim_out

    @torch.jit.export
    def get_intensive(self) -> bool:
        """G is extensive."""
        return False

    @torch.jit.export
    def get_var_name(self) -> str:
        """Name of the fitted property."""
        return self.get_fitting_net().var_name
