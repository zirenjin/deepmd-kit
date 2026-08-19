# SPDX-License-Identifier: LGPL-3.0-or-later
"""Schema for the ``fes`` fitting and loss types.

``dp --pt train`` runs dargs in strict mode on every ``input.json``, so a field
the head supports but the schema omits is rejected outright, and a field the
schema accepts but the head ignores passes validation while silently doing
nothing.  Both failure modes are cheap to introduce here because the ``fes``
schema is derived from the ``property`` one.

The distinction worth protecting is ``numb_state_fparam`` vs ``numb_fparam``:
the FES head keeps two fparam widths (the ``[T, P]`` a user writes into
``fparam.npy``, and the ``[T, P, v, c]`` the correction net consumes), and
naming the user-facing one differently is what keeps that visible in the input
file.
"""

import pytest
from dargs import (
    Argument,
)

from deepmd.utils.argcheck import (
    fitting_fes,
    loss_fes,
    normalize,
)


def _check(args: list[Argument], data: dict) -> dict:
    base = Argument("fitting_net", dict, args)
    base.check_value(data, strict=True)
    return data


def _fitting(**overrides) -> dict:
    """Fitting body *without* ``type``: it selects the variant and is not one of
    ``fitting_fes()``'s own arguments.
    """
    return dict(overrides)


def _base_config(fitting: dict, loss: dict | None = None) -> dict:
    return {
        "model": {
            "type_map": ["Si", "O"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [20, 40],
                "rcut": 6.0,
                "neuron": [4, 8],
                "axis_neuron": 4,
            },
            "fitting_net": {"type": "fes", **fitting},
        },
        "learning_rate": {
            "type": "exp",
            "start_lr": 1e-3,
            "stop_lr": 1e-5,
            "decay_steps": 100,
        },
        "loss": loss if loss is not None else {"type": "fes"},
        "training": {
            "training_data": {"systems": ["sys"], "batch_size": 1},
            "numb_steps": 10,
            "seed": 1,
        },
    }


def test_fes_defaults_are_the_documented_ones():
    normalized = normalize(_base_config(_fitting()))
    fitting = normalized["model"]["fitting_net"]
    assert fitting["property_name"] == "free_energy"
    assert fitting["numb_state_fparam"] == 2
    assert fitting["volume_mode"] == "per_atom"
    assert fitting["use_composition"] is True
    assert fitting["freeze_baseline"] is True
    # a non-empty encoder by default: a 5-wide state vector concatenated raw to
    # a 128+ wide embedding gets numerically drowned
    assert fitting["fparam_neuron"] == [64, 64]


def test_fes_accepts_its_own_options():
    _check(
        fitting_fes(),
        _fitting(
            property_name="gibbs",
            numb_state_fparam=3,
            volume_mode="both",
            use_composition=False,
            fparam_neuron=[32],
            neuron=[64, 64],
            baseline={"neuron": [240, 240, 240], "resnet_dt": True},
            freeze_baseline=False,
            default_fparam=[300.0, 1.0, 0.0],
            trainable=True,
            dim_case_embd=2,
            precision="float64",
            activation_function="gelu",
            resnet_dt=False,
            seed=1,
        ),
    )


@pytest.mark.parametrize(
    "unsupported",
    [
        # replaced by numb_state_fparam; accepting it would hide which of the
        # two fparam widths the user meant
        "numb_fparam",
        # the conditioning is frame-level, there is no per-atom channel
        "numb_aparam",
        # G is always extensive
        "intensive",
        # no label-statistics output bias to distinguish types for
        "distinguish_types",
        # single scalar output
        "task_dim",
        "some_future_typo",
    ],
)
def test_fes_rejects_unsupported_options(unsupported):
    with pytest.raises(Exception):
        _check(fitting_fes(), _fitting(**{unsupported: 1}))


def test_baseline_is_a_free_form_dict():
    """It is forwarded verbatim to EnergyFittingNet, whose own schema already
    owns those keys; re-declaring them here would only let the two drift apart.
    """
    normalized = normalize(
        _base_config(_fitting(baseline={"neuron": [240, 240], "resnet_dt": False}))
    )
    assert normalized["model"]["fitting_net"]["baseline"] == {
        "neuron": [240, 240],
        "resnet_dt": False,
    }


def test_fes_loss_defaults():
    normalized = normalize(_base_config(_fitting(), {"type": "fes"}))
    loss = normalized["loss"]
    assert loss["loss_func"] == "mse"
    assert loss["metric"] == ["mae", "rmse"]
    assert loss["delta_g_pref"] == 0.0


def test_fes_loss_accepts_its_options():
    _check(
        loss_fes(),
        {
            "loss_func": "smooth_mae",
            "metric": ["mae"],
            "beta": 2.0,
            "delta_g_pref": 0.0,
        },
    )
