# SPDX-License-Identifier: LGPL-3.0-or-later
"""Free energy surface (FES) head.

The head predicts ``G = E_DPA(X) + sum_i dr(h_i, T, P, v, c)`` with the
pre-trained energy net frozen inside it.  What has to hold for that decomposition
to mean anything:

* the frozen baseline really is frozen, and with a zero correction the model
  reproduces ``E_DPA`` bit-for-bit -- otherwise "baseline" is just a name;
* ``v`` and ``c`` are derived from ``box``/``atype`` and are numerically the
  values a hand computation gives, on all three code paths that build them
  (forward, data statistics, inference);
* the two fparam widths stay separate: users write ``[T, P]`` into
  ``fparam.npy`` while the correction net consumes ``[T, P, v, c]``.  Collapsing
  them silently would make the data pipeline demand columns nobody has.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("deepmd.lib")

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.task.free_energy import (
    BASELINE_NAME,
    CORRECTION_NAME,
    FreeEnergyFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.state_vector import (
    build_state_vector,
    state_vector_dim,
)

TYPE_MAP = ["Si", "O"]
NTYPES = len(TYPE_MAP)
NATOMS = 6
ATYPE_ROW = [0, 0, 1, 1, 1, 1]
CELL = 8.0


def _tensor(data, dtype):
    return torch.tensor(data, dtype=dtype, device=env.DEVICE)


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _model_params(**fitting_overrides):
    fitting = {
        "type": "fes",
        "numb_state_fparam": 2,
        "fparam_neuron": [8, 8],
        "neuron": [16, 16],
        "baseline": {"neuron": [16, 16]},
        "precision": "float64",
        "seed": 1,
    }
    fitting.update(fitting_overrides)
    return {
        "type_map": list(TYPE_MAP),
        "descriptor": {
            "type": "se_e2_a",
            "sel": [20, 40],
            "rcut": 6.0,
            "rcut_smth": 0.5,
            "neuron": [4, 8],
            "axis_neuron": 4,
            "seed": 1,
        },
        "fitting_net": fitting,
    }


def _make_model(**fitting_overrides):
    return get_model(_model_params(**fitting_overrides))


def _batch(nframes=2, seed=0):
    rng = np.random.default_rng(seed)
    coord = _tensor(rng.uniform(0, 5, (nframes, NATOMS, 3)), dtype=torch.float64)
    atype = _tensor([ATYPE_ROW] * nframes, dtype=torch.long)
    box = _tensor(
        np.tile(np.eye(3).reshape(1, 9) * CELL, (nframes, 1)), dtype=torch.float64
    )
    fparam = _tensor(
        [[300.0 + 300.0 * ii, 1.0] for ii in range(nframes)], dtype=torch.float64
    )
    return coord, atype, box, fparam


# --- state vector ----------------------------------------------------


def test_state_vector_matches_hand_computation():
    _, atype, box, fparam = _batch()
    state = build_state_vector(
        box, atype, fparam, NTYPES, "per_atom", True, numb_state_fparam=2
    )
    assert state.shape == (2, state_vector_dim(2, NTYPES, "per_atom", True))
    # [T, P, V/N, c_Si, c_O]
    np.testing.assert_allclose(_numpy(state[:, :2]), _numpy(fparam))
    np.testing.assert_allclose(_numpy(state[:, 2]), CELL**3 / NATOMS)
    np.testing.assert_allclose(_numpy(state[:, 3]), 2 / NATOMS)
    np.testing.assert_allclose(_numpy(state[:, 4]), 4 / NATOMS)


def test_state_vector_ignores_virtual_atoms():
    """Padding atoms (atype < 0) must not count toward N or the composition."""
    box = _tensor(np.eye(3).reshape(1, 9) * CELL, dtype=torch.float64)
    atype = _tensor([[0, 0, 1, 1, -1, -1]], dtype=torch.long)
    state = build_state_vector(box, atype, None, NTYPES, "per_atom", True)
    np.testing.assert_allclose(_numpy(state[:, 0]), CELL**3 / 4)  # N = 4, not 6
    np.testing.assert_allclose(_numpy(state[:, 1]), 0.5)
    np.testing.assert_allclose(_numpy(state[:, 2]), 0.5)


@pytest.mark.parametrize(
    ("volume_mode", "use_composition", "expected"),
    [
        ("per_atom", True, 2 + 1 + NTYPES),
        ("total", True, 2 + 1 + NTYPES),
        ("both", True, 2 + 2 + NTYPES),
        ("none", True, 2 + 0 + NTYPES),
        ("per_atom", False, 2 + 1),
    ],
)
def test_state_vector_dim(volume_mode, use_composition, expected):
    assert state_vector_dim(2, NTYPES, volume_mode, use_composition) == expected
    _, atype, box, fparam = _batch(nframes=1)
    state = build_state_vector(
        box, atype, fparam, NTYPES, volume_mode, use_composition, numb_state_fparam=2
    )
    assert state.shape[1] == expected


def test_state_vector_needs_box_for_volume():
    _, atype, _, fparam = _batch(nframes=1)
    with pytest.raises(ValueError, match="needs a simulation box"):
        build_state_vector(None, atype, fparam, NTYPES, "per_atom", True, 2)
    # ...but "none" is the documented escape hatch for non-periodic frames.
    state = build_state_vector(None, atype, fparam, NTYPES, "none", True, 2)
    assert state.shape[1] == 2 + NTYPES


# --- baseline / correction decomposition -----------------------------


def test_output_names_match_module_constants():
    """``forward`` writes the keys as literals (TorchScript cannot close over
    module-level strings); pin them so the two cannot drift apart.
    """
    model = _make_model()
    coord, atype, box, fparam = _batch()
    out = model(coord, atype, box=box, fparam=fparam)
    assert BASELINE_NAME in out
    assert CORRECTION_NAME in out
    assert set(model.get_fitting_net().output_def().keys()) == {
        "free_energy",
        BASELINE_NAME,
        CORRECTION_NAME,
    }


def test_baseline_is_frozen_and_correction_is_not():
    fitting = _make_model().get_fitting_net()
    assert not any(p.requires_grad for p in fitting.baseline.parameters())
    assert all(p.requires_grad for p in fitting.correction.parameters())
    assert all(p.requires_grad for p in fitting.fparam_network.parameters())


def test_freeze_baseline_false_unfreezes():
    fitting = _make_model(freeze_baseline=False).get_fitting_net()
    assert all(p.requires_grad for p in fitting.baseline.parameters())


def test_zero_correction_reproduces_baseline_exactly():
    model = _make_model()
    fitting = model.get_fitting_net()
    coord, atype, box, fparam = _batch()

    before = model(coord, atype, box=box, fparam=fparam)
    with torch.no_grad():
        for param in fitting.correction.parameters():
            param.zero_()
        fitting.correction.bias_atom_e.zero_()
    after = model(coord, atype, box=box, fparam=fparam)

    assert torch.allclose(before[BASELINE_NAME], after[BASELINE_NAME])
    assert torch.count_nonzero(after[CORRECTION_NAME]) == 0
    # bit-for-bit, not allclose: G must *be* E_DPA when the correction vanishes.
    assert torch.equal(after["free_energy"], after[BASELINE_NAME])


def test_free_energy_is_baseline_plus_correction():
    model = _make_model()
    coord, atype, box, fparam = _batch()
    out = model(coord, atype, box=box, fparam=fparam)
    assert torch.allclose(out["free_energy"], out[BASELINE_NAME] + out[CORRECTION_NAME])


def test_correction_responds_to_temperature():
    """The whole point of the head: G must move when T moves, at fixed structure."""
    model = _make_model()
    coord, atype, box, _ = _batch(nframes=2)
    coord[1] = coord[0]  # identical structures, different T
    cold = _tensor([[300.0, 1.0], [300.0, 1.0]], dtype=torch.float64)
    hot = _tensor([[300.0, 1.0], [2000.0, 1.0]], dtype=torch.float64)

    out_cold = model(coord, atype, box=box, fparam=cold)
    out_hot = model(coord, atype, box=box, fparam=hot)
    # frame 0 is unchanged, frame 1 saw a different temperature
    assert torch.allclose(out_cold["free_energy"][0], out_hot["free_energy"][0])
    assert not torch.allclose(out_cold["free_energy"][1], out_hot["free_energy"][1])
    # the baseline is structure-only and must be blind to T
    assert torch.allclose(out_cold[BASELINE_NAME], out_hot[BASELINE_NAME])


def test_gradients_reach_only_the_correction():
    model = _make_model()
    coord, atype, box, fparam = _batch()
    model(coord, atype, box=box, fparam=fparam)["free_energy"].sum().backward()
    fitting = model.get_fitting_net()
    assert all(p.grad is None for p in fitting.baseline.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in fitting.correction.parameters()
    )


# --- the two fparam widths -------------------------------------------


def test_model_reports_external_fparam_width_only():
    """``get_dim_fparam`` drives ``fparam.npy``'s ndof and DeepEval's reshape.

    It must report the 2 columns a user writes, not the 5 the correction net
    consumes -- otherwise the data pipeline demands volume/composition columns
    that are derived, not supplied.
    """
    model = _make_model()
    assert model.get_dim_fparam() == 2
    assert model.get_fitting_net().get_dim_fparam() == state_vector_dim(
        2, NTYPES, "per_atom", True
    )
    assert model.get_fitting_net().get_dim_state_fparam() == 2


def test_default_fparam_fills_in_for_missing_fparam():
    model = _make_model(default_fparam=[300.0, 1.0])
    coord, atype, box, fparam = _batch()
    explicit = model(
        coord,
        atype,
        box=box,
        fparam=_tensor([[300.0, 1.0]] * 2, dtype=torch.float64),
    )
    implicit = model(coord, atype, box=box, fparam=None)
    assert torch.allclose(explicit["free_energy"], implicit["free_energy"])


def test_missing_fparam_without_default_is_an_error():
    model = _make_model()
    coord, atype, box, _ = _batch()
    with pytest.raises(ValueError, match="frame parameter"):
        model(coord, atype, box=box, fparam=None)


def test_default_fparam_length_is_checked_against_state_fparam():
    with pytest.raises(ValueError, match="default_fparam"):
        _make_model(default_fparam=[300.0, 1.0, 42.0])


def test_forward_lower_requires_the_full_state_vector():
    model = _make_model()
    coord, atype, box, fparam = _batch()
    from deepmd.pt.utils.nlist import (
        extend_input_and_build_neighbor_list,
    )

    ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
        coord, atype, model.get_rcut(), model.get_sel(), mixed_types=True, box=box
    )
    # the user-facing [T, P] is not enough: volume cannot be derived without a box
    with pytest.raises(ValueError, match="full FES state vector"):
        model.forward_lower(ext_coord, ext_atype, nlist, mapping, fparam=fparam)

    full = model.build_state_fparam(box, atype, fparam)
    lower = model.forward_lower(ext_coord, ext_atype, nlist, mapping, fparam=full)
    upper = model(coord, atype, box=box, fparam=fparam)
    assert torch.allclose(lower["free_energy"], upper["free_energy"])


# --- statistics -------------------------------------------------------


def _stat_sample(nframes=4):
    rng = np.random.default_rng(0)
    return {
        "coord": _tensor(rng.uniform(0, 5, (nframes, NATOMS, 3)), dtype=torch.float64),
        "atype": _tensor([ATYPE_ROW] * nframes, dtype=torch.long),
        "box": _tensor(
            np.tile(np.eye(3).reshape(1, 9) * CELL, (nframes, 1)), dtype=torch.float64
        ),
        "fparam": _tensor(
            [[300.0, 1.0], [600.0, 1.0], [900.0, 1.0], [1200.0, 1.0]],
            dtype=torch.float64,
        ),
        "natoms": _tensor([[NATOMS, NATOMS, 2, 4]] * nframes, dtype=torch.long),
    }


def test_fparam_stats_cover_the_augmented_state_vector():
    model = _make_model()
    sample = _stat_sample()
    model.atomic_model.compute_fitting_input_stat(lambda: [sample])
    avg = model.get_fitting_net().correction.fparam_avg
    assert avg.shape == (state_vector_dim(2, NTYPES, "per_atom", True),)
    np.testing.assert_allclose(avg[0].item(), 750.0)  # mean T
    np.testing.assert_allclose(avg[1].item(), 1.0)  # P
    np.testing.assert_allclose(avg[2].item(), CELL**3 / NATOMS)  # V/N
    np.testing.assert_allclose(avg[3].item(), 2 / NATOMS)


def test_stat_augmentation_does_not_mutate_the_shared_sample():
    """The sampled dicts are shared with the descriptor statistics; some
    descriptors read ``fparam`` for charge/spin, so widening it in place would
    corrupt them.
    """
    model = _make_model()
    sample = _stat_sample()
    model.atomic_model.compute_fitting_input_stat(lambda: [sample])
    assert sample["fparam"].shape == (4, 2)


def test_out_bias_stays_zero():
    """The per-type offsets live inside the two sub-fitting nets; an atomic-model
    level bias fitted against G labels would double-count the baseline's own.
    """
    model = _make_model()
    sample = _stat_sample()
    sample["free_energy"] = torch.full(
        (4, 1), -123.0, dtype=torch.float64, device=env.DEVICE
    )
    model.atomic_model.change_out_bias([sample], bias_adjust_mode="set-by-statistic")
    assert torch.count_nonzero(model.atomic_model.out_bias) == 0


# --- (de)serialization and TorchScript --------------------------------


def test_serialize_roundtrip_preserves_predictions():
    model = _make_model()
    coord, atype, box, fparam = _batch()
    before = model(coord, atype, box=box, fparam=fparam)

    restored = FreeEnergyFittingNet.deserialize(model.get_fitting_net().serialize())
    model.atomic_model.fitting_net = restored
    after = model(coord, atype, box=box, fparam=fparam)

    for key in ("free_energy", BASELINE_NAME, CORRECTION_NAME):
        assert torch.allclose(before[key], after[key])
    assert not any(p.requires_grad for p in restored.baseline.parameters())


def test_jit_script_matches_eager():
    """``dp freeze`` scripts the model; the FES path must survive it."""
    model = _make_model()
    scripted = torch.jit.script(model)
    coord, atype, box, fparam = _batch()
    eager_out = model(coord, atype, box=box, fparam=fparam)
    scripted_out = scripted(coord, atype, box=box, fparam=fparam)
    for key, value in eager_out.items():
        assert torch.allclose(value, scripted_out[key])
    assert scripted.get_dim_fparam() == 2
    assert scripted.get_var_name() == "free_energy"


# --- construction guards ----------------------------------------------


def test_baseline_may_not_take_frame_or_atomic_parameters():
    for bad in ("numb_fparam", "numb_aparam"):
        with pytest.raises(ValueError, match=f"baseline.{bad}"):
            _make_model(baseline={"neuron": [16, 16], bad: 2})


def test_empty_fparam_neuron_keeps_the_raw_concatenation_path():
    model = _make_model(fparam_neuron=[])
    fitting = model.get_fitting_net()
    assert len(list(fitting.fparam_network.parameters())) == 0
    # the correction net still receives the state vector through its own fparam
    assert fitting.correction.numb_fparam == state_vector_dim(
        2, NTYPES, "per_atom", True
    )
    assert fitting.correction.dim_descrpt == fitting.dim_descrpt
    coord, atype, box, fparam = _batch()
    out = model(coord, atype, box=box, fparam=fparam)
    assert torch.isfinite(out["free_energy"]).all()


# --- finetune contract -------------------------------------------------


def test_baseline_state_dict_keys_mirror_a_plain_energy_head():
    """``dp train --finetune pes.pt`` loads the pre-trained energy fitting net
    into the baseline slot by stripping one ``baseline.`` level from each key
    (see collect_single_finetune_params in deepmd/pt/train/training.py).

    That remap is a string rewrite, so it only works while the baseline's own
    parameter names match a stand-alone EnergyFittingNet's exactly.
    """
    from deepmd.pt.model.task.ener import (
        EnergyFittingNet,
    )

    fitting = _make_model().get_fitting_net()
    baseline_keys = set(fitting.baseline.state_dict().keys())
    reference = EnergyFittingNet(
        ntypes=NTYPES,
        dim_descrpt=fitting.dim_descrpt,
        neuron=[16, 16],
        # se_e2_a is not mixed_types, so the head builds one network per type;
        # the pre-trained PES model shares the descriptor and therefore this
        # same layout, which is what makes the key rewrite valid.
        mixed_types=fitting.mixed_types,
    )
    assert baseline_keys == set(reference.state_dict().keys())

    # ...and they live under a "baseline." prefix inside the FES head, which is
    # exactly the level the remap strips.
    prefixed = {k for k in fitting.state_dict() if k.startswith("baseline.")}
    assert prefixed == {"baseline." + k for k in baseline_keys}


def test_correction_and_baseline_do_not_share_keys():
    """The remap routes ``baseline.*`` to the checkpoint and everything else to
    random init; overlapping names would make that routing ambiguous.
    """
    fitting = _make_model().get_fitting_net()
    keys = list(fitting.state_dict().keys())
    assert any(k.startswith("baseline.") for k in keys)
    assert any(k.startswith("correction.") for k in keys)
    assert any(k.startswith("fparam_network.") for k in keys)
    assert not any(k.startswith("correction.baseline.") for k in keys)
