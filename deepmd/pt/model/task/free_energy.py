# SPDX-License-Identifier: LGPL-3.0-or-later
"""Free energy surface (FES) fitting network.

Predicts the finite-temperature Gibbs free energy of a static structure as a
frozen pre-trained potential energy plus a learned, thermodynamically
conditioned per-atom correction::

    G(X, T, P) = E_DPA(X) + sum_i dr(h_i, T, P, v(X), c(X))

The head owns two stock deepmd fitting nets over one shared descriptor pass:

``baseline``
    An :class:`EnergyFittingNet` carrying the pre-trained PES weights.  Frozen
    by default, so ``E_DPA`` is a fixed reference rather than a moving target.
``correction``
    An :class:`InvarFitting` conditioned on the thermodynamic state vector,
    which the model layer assembles (see
    :mod:`deepmd.pt.utils.state_vector`) and hands over through ``fparam``.
"""

import logging
from typing import (
    Any,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.pt.utils.state_vector import (
    state_vector_dim,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

log = logging.getLogger(__name__)

CORRECTION_NAME = "fes_correction"
BASELINE_NAME = "fes_baseline"


def _activation(name: str) -> torch.nn.Module:
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name in {"linear", "none"}:
        return torch.nn.Identity()
    raise ValueError(f"Unsupported activation_function: {name!r}")


@Fitting.register("fes")
@fitting_check_output
class FreeEnergyFittingNet(Fitting):
    """Free energy surface head: frozen PES baseline + conditioned correction.

    Parameters
    ----------
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    property_name : str
        Name of the fitted quantity; must match the label file
        (``free_energy.npy`` by default).
    numb_state_fparam : int
        Width of the *externally supplied* frame parameters, i.e. the number of
        columns in ``fparam.npy``.  Defaults to 2 for ``[T, P]``.
    volume_mode : str
        How cell volume enters the state vector: ``per_atom`` (V/N, default),
        ``total``, ``both``, or ``none``.
    use_composition : bool
        Whether to append the per-type composition fractions to the state vector.
    neuron : list[int]
        Hidden widths of the correction net.
    fparam_neuron : list[int]
        Hidden widths of the state-vector encoder.  The encoded state is
        concatenated to the descriptor so that a handful of thermodynamic
        variables are not numerically drowned by a 128+ dimensional embedding;
        the raw (normalized) state vector is still concatenated by the
        correction net itself, so the encoder acts as an added wide path rather
        than a replacement.  An empty list disables the encoder.
    resnet_dt : bool
        Using a time-step in the ResNet construction of the correction net.
    activation_function : str
        Activation function of the correction net and the state encoder.
    precision : str
        Numerical precision.
    baseline : dict, optional
        Constructor arguments for the baseline :class:`EnergyFittingNet`.  These
        must match the pre-trained PES fitting net exactly (``neuron``,
        ``resnet_dt``, ``precision``, ``mixed_types``, ``dim_case_embd``),
        otherwise loading the checkpoint fails with a shape mismatch.
    freeze_baseline : bool
        Whether to freeze the baseline weights.  True keeps ``E_DPA`` fixed,
        which is the intended two-stage workflow.
    trainable : bool
        Whether the correction net and the state encoder are trainable.
    default_fparam : list[float], optional
        Fallback for the *external* frame parameters when ``fparam.npy`` is
        absent.  Length must equal ``numb_state_fparam``.
    dim_case_embd : int
        Dimension of the case embedding, for multi-task training.
    seed : int, optional
        Random seed for parameter initialization.
    type_map : list[str], optional
        Name of each atom type.
    exclude_types : list[int]
        Atom types whose contribution is set to zero.
    mixed_types : bool
        Whether one fitting net serves all atom types.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        property_name: str = "free_energy",
        numb_state_fparam: int = 2,
        volume_mode: str = "per_atom",
        use_composition: bool = True,
        temperature_basis: str = "mlp",
        temperature_scale: float = 1000.0,
        curvature_scale: float = 1.0e-2,
        temperature_knots: list[float] | None = None,
        phase_gauge_neuron: list[int] | None = None,
        phase_gauge_pooling: str = "mean",
        phase_gauge_basis: str = "piecewise_linear",
        phase_gauge_only: bool = False,
        center_local_correction: bool = False,
        neuron: list[int] | None = None,
        fparam_neuron: list[int] | None = None,
        resnet_dt: bool = True,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        baseline: dict[str, Any] | None = None,
        freeze_baseline: bool = True,
        trainable: bool = True,
        default_fparam: list[float] | None = None,
        dim_case_embd: int = 0,
        seed: int | list[int] | None = None,
        type_map: list[str] | None = None,
        exclude_types: list[int] | None = None,
        mixed_types: bool = True,
        # Injected unconditionally by _get_standard_model_components for every
        # fitting type; "type" already selected this class via the registry.
        type: str = "fes",
        **kwargs: Any,
    ) -> None:
        del type
        super().__init__()
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.var_name = property_name
        self.dim_out = 1
        self.task_dim = 1
        self.numb_state_fparam = int(numb_state_fparam)
        self.volume_mode = volume_mode
        self.use_composition = bool(use_composition)
        if temperature_basis not in (
            "mlp",
            "linear_zero_anchor",
            "affine",
            "concave",
            "concave_log",
            "entropy_affine",
            "concave_entropy",
            "piecewise_linear",
        ):
            raise ValueError(
                "temperature_basis must be 'mlp', 'linear_zero_anchor', 'affine', "
                "or 'concave', 'concave_log', 'entropy_affine', or "
                "'concave_entropy', or 'piecewise_linear'"
            )
        if temperature_scale <= 0.0:
            raise ValueError("temperature_scale must be positive")
        if curvature_scale <= 0.0:
            raise ValueError("curvature_scale must be positive")
        self.temperature_basis = temperature_basis
        self.temperature_scale = float(temperature_scale)
        self.curvature_scale = float(curvature_scale)
        self.temperature_knots = list(
            temperature_knots or [1000.0, 1200.0, 1900.0, 2200.0]
        )
        if len(self.temperature_knots) != 4 or any(
            self.temperature_knots[ii] >= self.temperature_knots[ii + 1]
            for ii in range(3)
        ):
            raise ValueError("temperature_knots must contain four increasing values")
        self.phase_gauge_neuron = list(phase_gauge_neuron or [])
        if phase_gauge_pooling not in (
            "mean",
            "mean_max",
            "mean_std",
            "mean_std_max",
            "type_mean",
            "deep_mean",
        ):
            raise ValueError(
                "phase_gauge_pooling must be 'mean', 'mean_max', 'mean_std', "
                "or 'mean_std_max', 'type_mean', or 'deep_mean'"
            )
        self.phase_gauge_pooling = phase_gauge_pooling
        if phase_gauge_basis not in ("piecewise_linear", "concave"):
            raise ValueError(
                "phase_gauge_basis must be 'piecewise_linear' or 'concave'"
            )
        self.phase_gauge_basis = phase_gauge_basis
        self.phase_gauge_only = bool(phase_gauge_only)
        if self.phase_gauge_only and not self.phase_gauge_neuron:
            raise ValueError("phase_gauge_only requires phase_gauge_neuron")
        if self.phase_gauge_only and self.temperature_basis != "piecewise_linear":
            raise ValueError("phase_gauge_only requires temperature_basis='piecewise_linear'")
        if self.phase_gauge_neuron and self.temperature_basis != "piecewise_linear":
            raise ValueError(
                "phase_gauge_neuron currently requires temperature_basis="
                "'piecewise_linear'"
            )
        self.center_local_correction = bool(center_local_correction)
        if self.center_local_correction and not self.phase_gauge_neuron:
            raise ValueError(
                "center_local_correction requires phase_gauge_neuron"
            )
        self.neuron = list(neuron or [128, 128, 128])
        self.fparam_neuron = list(fparam_neuron if fparam_neuron is not None else [])
        self.resnet_dt = resnet_dt
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.freeze_baseline = bool(freeze_baseline)
        self.trainable = bool(trainable)
        self.default_fparam = default_fparam
        self.dim_case_embd = int(dim_case_embd)
        self.seed = seed
        self.type_map = list(type_map or [])
        self.exclude_types = list(exclude_types or [])
        self.mixed_types = mixed_types

        if self.default_fparam is not None and len(self.default_fparam) != (
            self.numb_state_fparam
        ):
            raise ValueError(
                f"default_fparam has {len(self.default_fparam)} entries but "
                f"numb_state_fparam is {self.numb_state_fparam}; default_fparam "
                "covers only the externally supplied frame parameters (T, P), "
                "not the volume/composition columns derived from the structure."
            )

        # Full width seen by the correction net: [T, P] + v + c.
        self.state_dim = state_vector_dim(
            self.numb_state_fparam,
            self.ntypes,
            self.volume_mode,
            self.use_composition,
        )
        self.correction_state_dim = self.state_dim
        if self.temperature_basis in (
            "linear_zero_anchor",
            "affine",
            "concave",
            "concave_log",
            "entropy_affine",
            "concave_entropy",
            "piecewise_linear",
        ):
            if self.numb_state_fparam < 1:
                raise ValueError(
                    "temperature-dependent FES bases require temperature in fparam[:, 0]"
                )
            self.correction_state_dim -= 1

        baseline_cfg = dict(baseline or {})
        for forbidden in ("numb_fparam", "numb_aparam"):
            if baseline_cfg.pop(forbidden, 0):
                raise ValueError(
                    f"baseline.{forbidden} is not supported: the pre-trained PES "
                    "baseline consumes the descriptor only, while the "
                    "thermodynamic state vector is routed to the correction net."
                )
        baseline_cfg.setdefault("neuron", [128, 128, 128])
        baseline_cfg.setdefault("resnet_dt", resnet_dt)
        baseline_cfg.setdefault("precision", precision)
        baseline_cfg.setdefault("activation_function", activation_function)
        baseline_cfg.setdefault("mixed_types", mixed_types)
        baseline_cfg.setdefault("exclude_types", self.exclude_types)
        baseline_cfg.setdefault("type_map", self.type_map)
        baseline_cfg.setdefault("seed", seed)
        self.baseline_cfg = baseline_cfg
        self.baseline = EnergyFittingNet(
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            **baseline_cfg,
        )

        fparam_out_dim = self.fparam_neuron[-1] if self.fparam_neuron else 0
        self.correction = InvarFitting(
            var_name=CORRECTION_NAME,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt + fparam_out_dim,
            dim_out=1,
            neuron=self.neuron,
            resnet_dt=resnet_dt,
            numb_fparam=self.correction_state_dim,
            numb_aparam=0,
            dim_case_embd=self.dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            exclude_types=self.exclude_types,
            type_map=self.type_map,
            trainable=trainable,
        )
        knot_neuron = self.neuron if self.temperature_basis == "piecewise_linear" else [1]
        knot_trainable = trainable and self.temperature_basis == "piecewise_linear"
        self.knot_correction_1 = InvarFitting(
            var_name="fes_knot_1",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt + fparam_out_dim,
            dim_out=1,
            neuron=knot_neuron,
            resnet_dt=resnet_dt,
            numb_fparam=self.correction_state_dim,
            numb_aparam=0,
            dim_case_embd=self.dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            exclude_types=self.exclude_types,
            type_map=self.type_map,
            trainable=knot_trainable,
        )
        self.knot_correction_2 = InvarFitting(
            var_name="fes_knot_2",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt + fparam_out_dim,
            dim_out=1,
            neuron=knot_neuron,
            resnet_dt=resnet_dt,
            numb_fparam=self.correction_state_dim,
            numb_aparam=0,
            dim_case_embd=self.dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            exclude_types=self.exclude_types,
            type_map=self.type_map,
            trainable=knot_trainable,
        )
        self.knot_correction_3 = InvarFitting(
            var_name="fes_knot_3",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt + fparam_out_dim,
            dim_out=1,
            neuron=knot_neuron,
            resnet_dt=resnet_dt,
            numb_fparam=self.correction_state_dim,
            numb_aparam=0,
            dim_case_embd=self.dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            exclude_types=self.exclude_types,
            type_map=self.type_map,
            trainable=knot_trainable,
        )
        # Keep the optional slope module present in every scripted instance;
        # use a tiny inactive net outside affine mode to preserve the default
        # unrestricted FES behaviour without optional-module TorchScript state.
        slope_neuron = (
            self.neuron
            if self.temperature_basis in (
                "affine",
                "concave",
                "entropy_affine",
                "concave_entropy",
                "concave_log",
            )
            else [1]
        )
        self.slope_correction = InvarFitting(
            var_name="fes_slope",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt + fparam_out_dim,
            dim_out=1,
            neuron=slope_neuron,
            resnet_dt=resnet_dt,
            numb_fparam=self.correction_state_dim,
            numb_aparam=0,
            dim_case_embd=self.dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            exclude_types=self.exclude_types,
            type_map=self.type_map,
            trainable=trainable
            and self.temperature_basis
            in (
                "affine",
                "concave",
                "concave_log",
                "entropy_affine",
                "concave_entropy",
            ),
        )

        # A positive curvature coefficient gives a thermodynamically concave
        # free-energy curve: d2G/dT2 = -2*c / temperature_scale**2 <= 0,
        # consistent with non-negative heat capacity at fixed pressure.
        curvature_neuron = (
            self.neuron
            if self.temperature_basis in ("concave", "concave_log", "concave_entropy")
            else [1]
        )
        self.curvature_correction = InvarFitting(
            var_name="fes_curvature",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt + fparam_out_dim,
            dim_out=1,
            neuron=curvature_neuron,
            resnet_dt=resnet_dt,
            numb_fparam=self.correction_state_dim,
            numb_aparam=0,
            dim_case_embd=self.dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            exclude_types=self.exclude_types,
            type_map=self.type_map,
            trainable=trainable
            and self.temperature_basis in (
                "concave",
                "concave_log",
                "concave_entropy",
            ),
        )

        self.fparam_network = self._build_fparam_network(seed)
        self.phase_gauge_atom_network = self._build_phase_gauge_atom_network()
        self.phase_gauge_network = self._build_phase_gauge_network()

        self._set_trainable()

    def _build_fparam_network(self, seed: int | list[int] | None) -> torch.nn.Module:
        """State-vector encoder for the correction conditioning variables."""
        if not self.fparam_neuron:
            return torch.nn.Identity()

        def build() -> list[torch.nn.Module]:
            dims = [self.correction_state_dim, *self.fparam_neuron]
            layers: list[torch.nn.Module] = []
            for ii in range(len(dims) - 1):
                layers.append(
                    torch.nn.Linear(
                        dims[ii],
                        dims[ii + 1],
                        dtype=self.prec,
                        device=env.DEVICE,
                    )
                )
                layers.append(_activation(self.activation_function))
            return layers

        if seed is None:
            layers = build()
        else:
            # Scope the seed to this net without disturbing the caller's global
            # RNG stream (nn.Linear.reset_parameters takes no generator).
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed if isinstance(seed, int) else seed[0])
                layers = build()
        return torch.nn.Sequential(*layers).to(env.DEVICE)

    def _build_phase_gauge_network(self) -> torch.nn.Module:
        """Build a global, permutation-invariant phase-gauge predictor."""
        if not self.phase_gauge_neuron:
            return torch.nn.Identity()
        dims = [
            self.dim_descrpt
            * (
                self.ntypes
                if self.phase_gauge_pooling == "type_mean"
                else 3
                if self.phase_gauge_pooling == "mean_std_max"
                else 2
                if self.phase_gauge_pooling in ("mean_max", "mean_std")
                else 1
            )
            + self.correction_state_dim,
            *self.phase_gauge_neuron,
            # Keep four outputs for checkpoint compatibility with the
            # independent-knot gauge; the concave map below combines them into
            # an intercept, slope, and non-negative curvature.
            4,
        ]
        layers: list[torch.nn.Module] = []
        for ii in range(len(dims) - 1):
            layers.append(
                torch.nn.Linear(
                    dims[ii],
                    dims[ii + 1],
                    dtype=self.prec,
                    device=env.DEVICE,
                )
            )
            if ii < len(dims) - 2:
                layers.append(_activation(self.activation_function))
        return torch.nn.Sequential(*layers).to(env.DEVICE)

    def _build_phase_gauge_atom_network(self) -> torch.nn.Module:
        """Build an atomwise map before permutation-invariant pooling."""
        if self.phase_gauge_pooling != "deep_mean":
            return torch.nn.Identity()
        return torch.nn.Sequential(
            torch.nn.Linear(
                self.dim_descrpt,
                self.dim_descrpt,
                dtype=self.prec,
                device=env.DEVICE,
            ),
            _activation(self.activation_function),
        ).to(env.DEVICE)

    def _center_local_correction(
        self, value: torch.Tensor, atype: torch.Tensor
    ) -> torch.Tensor:
        """Remove the structure mean so only the global head carries G."""
        atom_mask = (atype >= 0).to(value.dtype)
        atom_count = torch.clamp(torch.sum(atom_mask, dim=1), min=1.0)
        mean = torch.sum(value[:, :, 0] * atom_mask, dim=1) / atom_count
        return value - mean.unsqueeze(1).unsqueeze(2)

    def _set_trainable(self) -> None:
        for param in self.baseline.parameters():
            param.requires_grad = not self.freeze_baseline
        for param in self.correction.parameters():
            param.requires_grad = self.trainable
        for param in self.knot_correction_1.parameters():
            param.requires_grad = self.trainable and self.temperature_basis == "piecewise_linear"
        for param in self.knot_correction_2.parameters():
            param.requires_grad = self.trainable and self.temperature_basis == "piecewise_linear"
        for param in self.knot_correction_3.parameters():
            param.requires_grad = self.trainable and self.temperature_basis == "piecewise_linear"
        for param in self.slope_correction.parameters():
            param.requires_grad = self.trainable and self.temperature_basis in (
                "affine",
                "concave",
                "entropy_affine",
                "concave_entropy",
            )
        for param in self.curvature_correction.parameters():
            param.requires_grad = self.trainable and self.temperature_basis in (
                "concave",
                "concave_log",
            )
        for param in self.fparam_network.parameters():
            param.requires_grad = self.trainable
        for param in self.phase_gauge_network.parameters():
            param.requires_grad = self.trainable and bool(self.phase_gauge_neuron)
        for param in self.phase_gauge_atom_network.parameters():
            param.requires_grad = self.trainable and self.phase_gauge_pooling == "deep_mean"

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        return_atomic_feature: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return the per-atom baseline, correction and their sum.

        ``fparam`` must already be the full state vector ``[T, P, v, c]``; the
        model layer assembles it because ``box`` does not reach the fitting net.
        """
        baseline = self.baseline(descriptor, atype, gr, g2, h2, None, None)["energy"]

        corr_descriptor = descriptor
        temperature_scale = torch.ones(
            (descriptor.shape[0], 1), dtype=descriptor.dtype, device=descriptor.device
        )
        if fparam is None:
            raise ValueError("the FES head requires a state vector in fparam")
        full_state = fparam.reshape(descriptor.shape[0], self.state_dim).to(self.prec)
        correction_fparam = full_state
        if self.temperature_basis in (
            "linear_zero_anchor",
            "affine",
            "concave",
            "concave_log",
            "entropy_affine",
            "concave_entropy",
            "piecewise_linear",
        ):
            temperature_scale = full_state[:, :1] / self.temperature_scale
            correction_fparam = full_state[:, 1:]
        if self.fparam_neuron:
            state = correction_fparam.reshape(
                descriptor.shape[0], self.correction_state_dim
            ).to(self.prec)
            # Reuse the correction net's fparam statistics so the encoder and
            # the raw concatenation path see the identically normalized vector.
            avg = self.correction.fparam_avg
            inv_std = self.correction.fparam_inv_std
            if avg is not None and inv_std is not None:
                state = (state - avg) * inv_std
            encoded = self.fparam_network(state)
            encoded = encoded.unsqueeze(1).expand(-1, descriptor.shape[1], -1)
            corr_descriptor = torch.cat([descriptor.to(self.prec), encoded], dim=-1)

        # Literal keys, not the module constants: TorchScript cannot close over
        # module-level strings.  ``test_fes_output_names`` pins them to
        # BASELINE_NAME/CORRECTION_NAME so the two cannot drift apart.
        if self.phase_gauge_only:
            correction = torch.zeros(
                (*descriptor.shape[:2], 1), dtype=descriptor.dtype, device=descriptor.device
            )
        else:
            correction = self.correction(
                corr_descriptor,
                atype,
                gr,
                g2,
                h2,
                correction_fparam,
                aparam,
            )["fes_correction"]
        phase_gauge = torch.zeros(
            (descriptor.shape[0], 4), dtype=correction.dtype, device=descriptor.device
        )
        if self.phase_gauge_neuron:
            atom_mask = (atype >= 0).to(self.prec)
            atom_count = torch.clamp(torch.sum(atom_mask, dim=1), min=1.0)
            pooled_input = descriptor.to(self.prec)
            if self.phase_gauge_pooling == "deep_mean":
                pooled_input = self.phase_gauge_atom_network(pooled_input)
            pooled_descriptor = torch.sum(
                pooled_input * atom_mask.unsqueeze(-1), dim=1
            ) / atom_count.unsqueeze(-1)
            if self.phase_gauge_pooling == "type_mean":
                type_means = []
                for type_idx in range(self.ntypes):
                    type_mask = (atype == type_idx).to(self.prec)
                    type_count = torch.clamp(torch.sum(type_mask, dim=1), min=1.0)
                    type_means.append(
                        torch.sum(
                            descriptor.to(self.prec) * type_mask.unsqueeze(-1), dim=1
                        )
                        / type_count.unsqueeze(-1)
                    )
                pooled_descriptor = torch.cat(type_means, dim=-1)
            elif self.phase_gauge_pooling == "mean_max":
                masked_descriptor = torch.where(
                    atom_mask.unsqueeze(-1) > 0.0,
                    descriptor.to(self.prec),
                    torch.full_like(descriptor, -torch.inf),
                )
                pooled_descriptor = torch.cat(
                    [pooled_descriptor, torch.max(masked_descriptor, dim=1).values],
                    dim=-1,
                )
            elif self.phase_gauge_pooling == "mean_std":
                centered = descriptor.to(self.prec) - pooled_descriptor.unsqueeze(1)
                variance = torch.sum(
                    centered * centered * atom_mask.unsqueeze(-1), dim=1
                ) / atom_count.unsqueeze(-1)
                pooled_descriptor = torch.cat(
                    [pooled_descriptor, torch.sqrt(torch.clamp(variance, min=1.0e-12))],
                    dim=-1,
                )
            elif self.phase_gauge_pooling == "mean_std_max":
                centered = descriptor.to(self.prec) - pooled_descriptor.unsqueeze(1)
                variance = torch.sum(
                    centered * centered * atom_mask.unsqueeze(-1), dim=1
                ) / atom_count.unsqueeze(-1)
                std_descriptor = torch.sqrt(torch.clamp(variance, min=1.0e-12))
                masked_descriptor = torch.where(
                    atom_mask.unsqueeze(-1) > 0.0,
                    descriptor.to(self.prec),
                    torch.full_like(descriptor, -torch.inf),
                )
                pooled_descriptor = torch.cat(
                    [
                        pooled_descriptor,
                        std_descriptor,
                        torch.max(masked_descriptor, dim=1).values,
                    ],
                    dim=-1,
                )
            gauge_state = correction_fparam.reshape(
                descriptor.shape[0], self.correction_state_dim
            ).to(self.prec)
            avg = self.correction.fparam_avg
            inv_std = self.correction.fparam_inv_std
            if avg is not None and inv_std is not None:
                gauge_state = (gauge_state - avg) * inv_std
            gauge_state = gauge_state.to(self.prec)
            gauge_input = torch.cat([pooled_descriptor, gauge_state], dim=-1)
            gauge_input = gauge_input.to(self.phase_gauge_network[0].weight.dtype)
            phase_gauge = self.phase_gauge_network(
                gauge_input
            )
            if self.phase_gauge_basis == "concave":
                # The global phase correction is concave in temperature, as
                # required by d2G/dT2 = -Cp/T <= 0 for positive heat capacity.
                curvature = torch.nn.functional.softplus(phase_gauge[:, 2:3])
                slope = phase_gauge[:, 1:2] + phase_gauge[:, 3:4]
                phase_gauge = torch.cat(
                    [
                        phase_gauge[:, 0:1]
                        + slope
                        * (torch.full_like(full_state[:, :1], knot)
                           / self.temperature_scale)
                        - curvature
                        * (torch.full_like(full_state[:, :1], knot)
                           / self.temperature_scale) ** 2
                        for knot in self.temperature_knots
                    ],
                    dim=1,
                )
        if self.temperature_basis == "piecewise_linear":
            if self.phase_gauge_only:
                knot_1 = torch.zeros_like(correction)
                knot_2 = torch.zeros_like(correction)
                knot_3 = torch.zeros_like(correction)
            else:
                knot_1 = self.knot_correction_1(
                    corr_descriptor, atype, gr, g2, h2, correction_fparam, aparam
                )["fes_knot_1"]
                knot_2 = self.knot_correction_2(
                    corr_descriptor, atype, gr, g2, h2, correction_fparam, aparam
                )["fes_knot_2"]
                knot_3 = self.knot_correction_3(
                    corr_descriptor, atype, gr, g2, h2, correction_fparam, aparam
                )["fes_knot_3"]
            if self.center_local_correction:
                correction = self._center_local_correction(correction, atype)
                knot_1 = self._center_local_correction(knot_1, atype)
                knot_2 = self._center_local_correction(knot_2, atype)
                knot_3 = self._center_local_correction(knot_3, atype)
            if self.phase_gauge_neuron:
                atom_count = torch.clamp(
                    torch.sum((atype >= 0).to(correction.dtype), dim=1), min=1.0
                ).reshape(-1, 1, 1)
                correction = correction + phase_gauge[:, 0:1].unsqueeze(1) / atom_count
                knot_1 = knot_1 + phase_gauge[:, 1:2].unsqueeze(1) / atom_count
                knot_2 = knot_2 + phase_gauge[:, 2:3].unsqueeze(1) / atom_count
                knot_3 = knot_3 + phase_gauge[:, 3:4].unsqueeze(1) / atom_count
            t = full_state[:, :1]
            k0 = self.temperature_knots[0]
            k1 = self.temperature_knots[1]
            k2 = self.temperature_knots[2]
            k3 = self.temperature_knots[3]
            w0 = torch.clamp((k1 - t) / (k1 - k0), 0.0, 1.0)
            w1 = torch.where(
                t <= k1,
                torch.clamp((t - k0) / (k1 - k0), 0.0, 1.0),
                torch.clamp((k2 - t) / (k2 - k1), 0.0, 1.0),
            )
            w2 = torch.where(
                t <= k2,
                torch.clamp((t - k1) / (k2 - k1), 0.0, 1.0),
                torch.clamp((k3 - t) / (k3 - k2), 0.0, 1.0),
            )
            w3 = torch.clamp((t - k2) / (k3 - k2), 0.0, 1.0)
            correction = (
                correction * w0.unsqueeze(1)
                + knot_1 * w1.unsqueeze(1)
                + knot_2 * w2.unsqueeze(1)
                + knot_3 * w3.unsqueeze(1)
            )
        elif self.temperature_basis == "linear_zero_anchor":
            correction = correction * temperature_scale.unsqueeze(1)
        elif self.temperature_basis == "affine":
            slope = self.slope_correction(
                corr_descriptor,
                atype,
                gr,
                g2,
                h2,
                correction_fparam,
                aparam,
            )["fes_slope"]
            correction = correction + slope * temperature_scale.unsqueeze(1)

        elif self.temperature_basis == "concave":
            slope = self.slope_correction(
                corr_descriptor,
                atype,
                gr,
                g2,
                h2,
                correction_fparam,
                aparam,
            )["fes_slope"]
            raw_curvature = self.curvature_correction(
                corr_descriptor,
                atype,
                gr,
                g2,
                h2,
                correction_fparam,
                aparam,
            )["fes_curvature"]
            curvature = torch.nn.functional.softplus(raw_curvature)
            correction = (
                correction
                + slope * temperature_scale.unsqueeze(1)
                - self.curvature_scale
                * curvature
                * temperature_scale.unsqueeze(1).square()
            )
        elif self.temperature_basis == "concave_log":
            slope = self.slope_correction(
                corr_descriptor,
                atype,
                gr,
                g2,
                h2,
                correction_fparam,
                aparam,
            )["fes_slope"]
            raw_curvature = self.curvature_correction(
                corr_descriptor,
                atype,
                gr,
                g2,
                h2,
                correction_fparam,
                aparam,
            )["fes_curvature"]
            curvature = torch.nn.functional.softplus(raw_curvature)
            x = temperature_scale.unsqueeze(1)
            correction = correction + slope * x - self.curvature_scale * curvature * x * torch.log(x)
        elif self.temperature_basis == "entropy_affine":
            raw_entropy = self.slope_correction(
                corr_descriptor,
                atype,
                gr,
                g2,
                h2,
                correction_fparam,
                aparam,
            )["fes_slope"]
            entropy = torch.nn.functional.softplus(raw_entropy)
            correction = correction - entropy * temperature_scale.unsqueeze(1)
        elif self.temperature_basis == "concave_entropy":
            raw_entropy = self.slope_correction(
                corr_descriptor,
                atype,
                gr,
                g2,
                h2,
                correction_fparam,
                aparam,
            )["fes_slope"]
            raw_curvature = self.curvature_correction(
                corr_descriptor,
                atype,
                gr,
                g2,
                h2,
                correction_fparam,
                aparam,
            )["fes_curvature"]
            x = temperature_scale.unsqueeze(1)
            correction = (
                correction
                - torch.nn.functional.softplus(raw_entropy) * x
                - self.curvature_scale
                * torch.nn.functional.softplus(raw_curvature)
                * x.square()
            )

        return {
            "fes_baseline": baseline,
            "fes_correction": correction,
            self.var_name: baseline + correction,
        }

    def output_def(self) -> FittingOutputDef:
        # free_energy is the trained/served output; the other two are kept as
        # separate variables so training logs can show how much of G comes from
        # the frozen baseline and how much the correction actually moves.
        # Written out longhand rather than through a local helper: this runs
        # inside TorchScript (via fitting_output_def -> do_grad_r), which does
        # not support nested function definitions.
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=False,
                ),
                OutputVariableDef(
                    "fes_baseline",
                    [self.dim_out],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=False,
                ),
                OutputVariableDef(
                    "fes_correction",
                    [self.dim_out],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=False,
                ),
            ]
        )

    # --- introspection -------------------------------------------------

    @torch.jit.export
    def get_dim_fparam(self) -> int:
        """Width of the state vector consumed by the correction net."""
        return self.state_dim

    @torch.jit.export
    def get_dim_state_fparam(self) -> int:
        """Width of the frame parameters the *user* supplies (``fparam.npy``)."""
        return self.numb_state_fparam

    @torch.jit.export
    def has_default_fparam(self) -> bool:
        return self.default_fparam is not None

    @torch.jit.export
    def get_default_fparam(self) -> torch.Tensor | None:
        if self.default_fparam is None:
            return None
        return torch.tensor(
            self.default_fparam,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )

    @torch.jit.export
    def get_dim_aparam(self) -> int:
        return 0

    @torch.jit.export
    def get_task_dim(self) -> int:
        return self.dim_out

    @torch.jit.export
    def get_intensive(self) -> bool:
        """G is extensive: the per-atom outputs are summed, not averaged."""
        return False

    @torch.jit.export
    def get_type_map(self) -> list[str]:
        return self.type_map if self.type_map is not None else []

    @torch.jit.export
    def get_sel_type(self) -> list[int]:
        # Explicit loop, not a filtered comprehension: TorchScript does not
        # support comprehension `if` clauses (same shape as GeneralFitting's).
        sel_type: list[int] = []
        for ii in range(self.ntypes):
            if ii not in self.exclude_types:
                sel_type.append(ii)
        return sel_type

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        self.baseline.change_type_map(type_map, model_with_new_type_stat)
        self.correction.change_type_map(type_map, model_with_new_type_stat)
        self.knot_correction_1.change_type_map(type_map, model_with_new_type_stat)
        self.knot_correction_2.change_type_map(type_map, model_with_new_type_stat)
        self.knot_correction_3.change_type_map(type_map, model_with_new_type_stat)
        self.slope_correction.change_type_map(type_map, model_with_new_type_stat)
        self.curvature_correction.change_type_map(type_map, model_with_new_type_stat)
        self.type_map = list(type_map)
        self.ntypes = len(type_map)

    def set_case_embd(self, case_idx: int) -> None:
        """Distinguish branches that share a descriptor in multi-task training.

        Only the correction net carries a case embedding; the baseline stays a
        single frozen reference potential across branches.
        """
        self.correction.set_case_embd(case_idx)
        if self.temperature_basis == "piecewise_linear":
            self.knot_correction_1.set_case_embd(case_idx)
            self.knot_correction_2.set_case_embd(case_idx)
            self.knot_correction_3.set_case_embd(case_idx)
        if self.temperature_basis in (
            "affine",
            "concave",
            "concave_log",
            "entropy_affine",
            "concave_entropy",
        ):
            self.slope_correction.set_case_embd(case_idx)
        if self.temperature_basis in ("concave", "concave_log", "concave_entropy"):
            self.curvature_correction.set_case_embd(case_idx)

    def compute_input_stats(
        self,
        merged: Any,
        protection: float = 1e-2,
        stat_file_path: Any | None = None,
    ) -> None:
        """Delegate fparam statistics to the correction net.

        ``merged`` must already carry the augmented state vector under
        ``fparam``; :class:`DPFreeEnergyAtomicModel` wraps the sampler so that
        the statistics are taken over ``[T, P, v, c]`` rather than the two raw
        columns found in ``fparam.npy``.
        """
        if self.temperature_basis in (
            "linear_zero_anchor",
            "affine",
            "concave",
            "concave_log",
            "entropy_affine",
            "concave_entropy",
            "piecewise_linear",
        ):
            def reduce_samples(samples: list[dict]) -> list[dict]:
                reduced_samples = []
                for sample in samples:
                    item = dict(sample)
                    item["fparam"] = sample["fparam"][..., 1:]
                    reduced_samples.append(item)
                return reduced_samples

            # Keep the sampler lazy. Paired loaders intentionally do not
            # expose the ordinary loader internals used during statistics;
            # when a stats file exists, the fitting nets can load it without
            # sampling at all.
            reduced = (
                (lambda: reduce_samples(merged()))
                if callable(merged)
                else reduce_samples(merged)
            )
            self.correction.compute_input_stats(reduced, protection, stat_file_path)
            if self.temperature_basis in ("affine", "entropy_affine"):
                self.slope_correction.compute_input_stats(
                    reduced, protection, stat_file_path
                )
            if self.temperature_basis in ("concave", "concave_log", "concave_entropy"):
                self.slope_correction.compute_input_stats(
                    reduced, protection, stat_file_path
                )
                self.curvature_correction.compute_input_stats(
                    reduced, protection, stat_file_path
                )
            if self.temperature_basis == "piecewise_linear":
                self.knot_correction_1.compute_input_stats(
                    reduced, protection, stat_file_path
                )
                self.knot_correction_2.compute_input_stats(
                    reduced, protection, stat_file_path
                )
                self.knot_correction_3.compute_input_stats(
                    reduced, protection, stat_file_path
                )
        else:
            self.correction.compute_input_stats(merged, protection, stat_file_path)

    # --- (de)serialization ---------------------------------------------

    def serialize(self) -> dict[str, Any]:
        fparam_layers = [
            {
                "matrix": to_numpy_array(layer.weight),
                "bias": to_numpy_array(layer.bias),
            }
            for layer in self.fparam_network.modules()
            if isinstance(layer, torch.nn.Linear)
        ]
        phase_gauge_layers = [
            {
                "matrix": to_numpy_array(layer.weight),
                "bias": to_numpy_array(layer.bias),
            }
            for layer in self.phase_gauge_network.modules()
            if isinstance(layer, torch.nn.Linear)
        ]
        phase_gauge_atom_layers = [
            {
                "matrix": to_numpy_array(layer.weight),
                "bias": to_numpy_array(layer.bias),
            }
            for layer in self.phase_gauge_atom_network.modules()
            if isinstance(layer, torch.nn.Linear)
        ]
        return {
            "@class": "Fitting",
            "@version": 1,
            "type": "fes",
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "property_name": self.var_name,
            "numb_state_fparam": self.numb_state_fparam,
            "volume_mode": self.volume_mode,
            "use_composition": self.use_composition,
            "temperature_basis": self.temperature_basis,
            "temperature_scale": self.temperature_scale,
            "curvature_scale": self.curvature_scale,
            "temperature_knots": self.temperature_knots,
            "phase_gauge_neuron": self.phase_gauge_neuron,
            "phase_gauge_pooling": self.phase_gauge_pooling,
            "phase_gauge_basis": self.phase_gauge_basis,
            "phase_gauge_only": self.phase_gauge_only,
            "center_local_correction": self.center_local_correction,
            "neuron": self.neuron,
            "fparam_neuron": self.fparam_neuron,
            "resnet_dt": self.resnet_dt,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "baseline": self.baseline_cfg,
            "freeze_baseline": self.freeze_baseline,
            "trainable": self.trainable,
            "default_fparam": self.default_fparam,
            "dim_case_embd": self.dim_case_embd,
            "type_map": self.type_map,
            "exclude_types": self.exclude_types,
            "mixed_types": self.mixed_types,
            "@variables": {
                "baseline": self.baseline.serialize(),
                "correction": self.correction.serialize(),
                "knot_correction_1": self.knot_correction_1.serialize(),
                "knot_correction_2": self.knot_correction_2.serialize(),
                "knot_correction_3": self.knot_correction_3.serialize(),
                "slope_correction": self.slope_correction.serialize(),
                "curvature_correction": self.curvature_correction.serialize(),
                "fparam_network": fparam_layers,
            "phase_gauge_network": phase_gauge_layers,
                "phase_gauge_atom_network": phase_gauge_atom_layers,
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "FreeEnergyFittingNet":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        variables = data.pop("@variables")
        obj = cls(**data)
        obj.baseline = EnergyFittingNet.deserialize(variables["baseline"])
        obj.correction = InvarFitting.deserialize(variables["correction"])
        if "knot_correction_1" in variables:
            obj.knot_correction_1 = InvarFitting.deserialize(variables["knot_correction_1"])
        if "knot_correction_2" in variables:
            obj.knot_correction_2 = InvarFitting.deserialize(variables["knot_correction_2"])
        if "knot_correction_3" in variables:
            obj.knot_correction_3 = InvarFitting.deserialize(variables["knot_correction_3"])
        if "slope_correction" in variables:
            obj.slope_correction = InvarFitting.deserialize(variables["slope_correction"])
        if "curvature_correction" in variables:
            obj.curvature_correction = InvarFitting.deserialize(
                variables["curvature_correction"]
            )
        linears = [
            layer
            for layer in obj.fparam_network.modules()
            if isinstance(layer, torch.nn.Linear)
        ]
        for layer, saved in zip(linears, variables["fparam_network"], strict=True):
            layer.weight.data.copy_(to_torch_tensor(saved["matrix"]))
            layer.bias.data.copy_(to_torch_tensor(saved["bias"]))
        if "phase_gauge_network" in variables:
            gauge_linears = [
                layer
                for layer in obj.phase_gauge_network.modules()
                if isinstance(layer, torch.nn.Linear)
            ]
            for layer, saved in zip(
                gauge_linears, variables["phase_gauge_network"], strict=True
            ):
                layer.weight.data.copy_(to_torch_tensor(saved["matrix"]))
                layer.bias.data.copy_(to_torch_tensor(saved["bias"]))
        if "phase_gauge_atom_network" in variables:
            atom_linears = [
                layer
                for layer in obj.phase_gauge_atom_network.modules()
                if isinstance(layer, torch.nn.Linear)
            ]
            for layer, saved in zip(
                atom_linears, variables["phase_gauge_atom_network"], strict=True
            ):
                layer.weight.data.copy_(to_torch_tensor(saved["matrix"]))
                layer.bias.data.copy_(to_torch_tensor(saved["bias"]))
        obj._set_trainable()
        return obj

    # make jit happy with torch 2.0.0
    exclude_types: list[int]
