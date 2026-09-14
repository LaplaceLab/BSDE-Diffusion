import torch
from torch import nn

from bsde_diffusion import AssociatedBSDE, DeepBSDESolver, SolverConfig


class ZeroControl(nn.Module):
    def forward(self, _time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(state)


def _zero_field(_time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(state)


def test_solver_learns_constant_terminal_value() -> None:
    torch.manual_seed(7)
    problem = AssociatedBSDE(
        initial_state=torch.zeros(1, 2),
        time_grid=torch.linspace(0, 1, 4),
        drift=_zero_field,
        diffusion=_zero_field,
        terminal=lambda state: torch.full_like(state, 2.0),
    )
    solver = DeepBSDESolver(problem, ZeroControl())
    solution = solver.fit(SolverConfig(iterations=160, batch_size=8, learning_rate=0.08))

    torch.testing.assert_close(solution.initial_value, torch.full((1, 2), 2.0), atol=2e-3, rtol=0)
    assert solution.final_loss < 1e-5
    assert solution.final_loss < solution.loss_history[0]


def test_rollout_preserves_arbitrary_event_shape() -> None:
    problem = AssociatedBSDE(
        initial_state=torch.zeros(1, 2, 4, 4),
        time_grid=torch.linspace(0, 1, 3),
        drift=_zero_field,
        diffusion=_zero_field,
        terminal=lambda state: state,
    )
    state, value, target = DeepBSDESolver(problem, ZeroControl()).rollout(batch_size=3)
    assert state.shape == value.shape == target.shape == (3, 2, 4, 4)

