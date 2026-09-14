import pytest
import torch

from bsde_diffusion import AssociatedBSDE


def _zero_field(_time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(state)


def test_problem_accepts_general_tensor_state() -> None:
    state = torch.zeros(1, 2, 3)
    problem = AssociatedBSDE(state, torch.linspace(0, 1, 5), _zero_field, _zero_field, lambda x: x)
    assert problem.initial_state.shape == (1, 2, 3)


@pytest.mark.parametrize(
    ("state", "grid", "message"),
    [
        (torch.zeros(2, 3), torch.linspace(0, 1, 3), "one shared anchor"),
        (torch.zeros(1, 3), torch.tensor([0.0]), "at least two"),
        (torch.zeros(1, 3), torch.tensor([0.0, 0.5, 0.5]), "strictly increasing"),
    ],
)
def test_problem_rejects_invalid_shapes(state: torch.Tensor, grid: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AssociatedBSDE(state, grid, _zero_field, _zero_field, lambda x: x)

