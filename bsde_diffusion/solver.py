"""Neural solver for an associated backward stochastic differential equation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .problem import AssociatedBSDE


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Optimization settings for the Deep-BSDE discretization."""

    iterations: int = 1_000
    batch_size: int = 32
    learning_rate: float = 3e-4
    gradient_clip: float | None = 1.0

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.batch_size <= 0:
            raise ValueError("iterations and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive when provided")


@dataclass(frozen=True, slots=True)
class BSDESolution:
    """Learned inverse anchor and terminal-loss trace."""

    initial_value: torch.Tensor
    loss_history: tuple[float, ...]

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1]


class DeepBSDESolver(nn.Module):
    """Learn ``Y_0`` and ``Z_t`` so the simulated process reaches ``g(X_T)``."""

    def __init__(
        self,
        problem: AssociatedBSDE,
        control: nn.Module,
        *,
        initial_value: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.control = control

        with torch.no_grad():
            target = problem.terminal(problem.initial_state)
        if target.shape != problem.initial_state.shape:
            raise ValueError("terminal must preserve the state shape")

        value = torch.zeros_like(problem.initial_state) if initial_value is None else initial_value
        if value.shape != problem.initial_state.shape:
            raise ValueError("initial_value must match initial_state")
        self.initial_value = nn.Parameter(value.detach().clone())

    @staticmethod
    def _field_like(field: torch.Tensor, state: torch.Tensor, name: str) -> torch.Tensor:
        try:
            return torch.broadcast_to(field, state.shape)
        except RuntimeError as error:
            raise ValueError(f"{name} must be broadcastable to the state shape") from error

    def rollout(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate the coupled Euler-Maruyama and BSDE discretization."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        state = self.problem.initial_state.expand(batch_size, *self.problem.initial_state.shape[1:]).clone()
        value = self.initial_value.expand_as(state)
        grid = self.problem.time_grid.to(device=state.device, dtype=state.dtype)

        for time, next_time in zip(grid[:-1], grid[1:]):
            dt = next_time - time
            noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
            brownian_increment = noise * torch.sqrt(dt)

            control = self.control(time, state)
            if control.shape != state.shape:
                raise ValueError("control must return a tensor with the state shape")
            driver = self.problem.driver(time, state, value, control)
            if driver.shape != value.shape:
                raise ValueError("driver must return a tensor with the value shape")

            value = value - driver * dt + control * brownian_increment
            with torch.no_grad():
                drift = self._field_like(self.problem.drift(time, state), state, "drift")
                diffusion = self._field_like(self.problem.diffusion(time, state), state, "diffusion")
                state = state + drift * dt + diffusion * brownian_increment

        with torch.no_grad():
            target = self.problem.terminal(state)
        if target.shape != value.shape:
            raise ValueError("terminal must preserve the state shape")
        return state, value, target

    def fit(
        self,
        config: SolverConfig = SolverConfig(),
        *,
        generator: torch.Generator | None = None,
        callback: Callable[[int, float], None] | None = None,
    ) -> BSDESolution:
        """Minimize the Monte Carlo terminal mismatch."""
        optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        history: list[float] = []
        self.train()

        for iteration in range(config.iterations):
            optimizer.zero_grad(set_to_none=True)
            _, terminal_value, target = self.rollout(config.batch_size, generator=generator)
            loss = torch.mean((terminal_value - target) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("terminal loss became non-finite")
            loss.backward()
            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(self.parameters(), config.gradient_clip)
            optimizer.step()

            loss_value = float(loss.detach())
            history.append(loss_value)
            if callback is not None:
                callback(iteration, loss_value)

        return BSDESolution(initial_value=self.initial_value.detach().clone(), loss_history=tuple(history))


def solve_associated_bsde(
    problem: AssociatedBSDE,
    control: nn.Module,
    config: SolverConfig = SolverConfig(),
    *,
    initial_value: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    callback: Callable[[int, float], None] | None = None,
) -> BSDESolution:
    """Construct and fit a :class:`DeepBSDESolver` in one call."""
    solver = DeepBSDESolver(problem, control, initial_value=initial_value).to(problem.initial_state.device)
    return solver.fit(config, generator=generator, callback=callback)

