# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.pt.loss.free_energy import FreeEnergyLoss


class _FixedFreeEnergy(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        del kwargs
        return {"free_energy": self.values}


def test_multi_phase_delta_loss_uses_phase_zero_as_anchor() -> None:
    # Three phases: predicted relative values are [2, 5] and references are
    # [1, 2], so the MSE over the two non-anchor phases is (1 + 4) / 2.
    loss = FreeEnergyLoss(
        absolute_g_pref=0.0,
        delta_g_pref=1.0,
        metric=["mae", "rmse"],
    )
    label = {
        "free_energy": torch.tensor([[1.0], [2.0], [4.0]]),
        "pair_batch_size": torch.tensor(1),
        "pair_phase_count": torch.tensor(3),
    }
    _, value, diagnostics = loss({}, _FixedFreeEnergy([1.0, 3.0, 6.0]), label, 1)
    assert torch.isclose(value, torch.tensor(2.5))
    assert torch.isclose(diagnostics["delta_mae"], torch.tensor(1.5))
