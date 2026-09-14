# BSDE Diffusion

A compact PyTorch framework for terminal-conditioned inversion with pretrained stochastic diffusion priors.

The framework solves the associated system

\[
dX_t=b(t,X_t)\,dt+\sigma(t,X_t)\,dW_t,
\qquad
dY_t=-f(t,X_t,Y_t,Z_t)\,dt+Z_t\,dW_t,
\qquad
Y_T=\Psi(X_T;y),
\]

where the application supplies the frozen prior dynamics and the terminal map $\Psi$. The core is independent of the measurement modality, forward operator, score-model architecture, and data shape.

## What you provide

- `initial_state`: one prior-space anchor with a batch dimension.
- `time_grid`: a strictly increasing discretization from inversion time 0 to terminal time $T$.
- `drift(t, x)` and `diffusion(t, x)`: the frozen prior dynamics.
- `terminal(x)`: the task-specific feasibility map, typically decode $\rightarrow$ enforce the observation $\rightarrow$ encode.
- `control(t, x)`: a trainable PyTorch module for the martingale integrand $Z_t$.
- Optionally, `driver(t, x, y, z)` for a nonzero BSDE driver.

## Minimal use

```python
import torch

from bsde_diffusion import (
    AssociatedBSDE,
    SolverConfig,
    TimeConditionedMLP,
    solve_associated_bsde,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anchor = torch.zeros(1, 16, device=device)


def prior_drift(time, state):
    return torch.zeros_like(state)  # replace with the frozen score-SDE drift


def prior_diffusion(time, state):
    return torch.ones_like(state) * 0.2  # replace with the prior diffusion


measurement = torch.ones_like(anchor)


def terminal_map(state):
    return measurement.expand_as(state)  # replace with decode-project-encode


problem = AssociatedBSDE(
    initial_state=anchor,
    time_grid=torch.linspace(0, 1, 33, device=device),
    drift=prior_drift,
    diffusion=prior_diffusion,
    terminal=terminal_map,
)
control = TimeConditionedMLP(anchor.shape[1:]).to(device)
solution = solve_associated_bsde(
    problem,
    control,
    SolverConfig(iterations=1_000, batch_size=32),
)

recovered_anchor = solution.initial_value
print(solution.final_loss)
```

For images, volumes, time series, or other structured states, replace `TimeConditionedMLP` with a modality-appropriate `torch.nn.Module` whose `forward(time, state)` returns a tensor with the same shape as `state`.

## Install and verify

```bash
git clone https://github.com/LaplaceLab/BSDE-Diffusion.git
cd BSDE-Diffusion

python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
ruff check .
```

## Agent-assisted integration

[`SKILL.md`](SKILL.md) gives a coding agent a complete workflow for adapting the framework to a new inverse problem while keeping the prior frozen and task-specific physics outside the core package.

## Paper

Zihao Wang. “Backward SDEs-based Diffusion for Physics-Constrained Generation.” *Proceedings of the 43rd International Conference on Machine Learning*, PMLR 306, 2026.

Project page: <https://laplacelab.github.io/BSDEDiffusion/>
