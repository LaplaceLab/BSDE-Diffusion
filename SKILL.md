---
name: bsde-inverse-problem
description: Adapts the BSDE Diffusion framework to a new physics-constrained or measurement-conditioned inverse problem. Use when implementing a terminal map, connecting a pretrained score-SDE prior, choosing a control network, training the associated BSDE, or validating feasibility and uncertainty for a new modality.
---

# Deploy BSDE Diffusion on an inverse problem

Use this repository as a framework. Keep application physics and model adapters outside `bsde_diffusion/`; the core package must remain modality-agnostic.

## Integration contract

Define these application-owned components:

1. A prior-space `initial_state` with shape `(1, *event_shape)`.
2. A strictly increasing `time_grid` on the same device and dtype.
3. Frozen `drift(time, state)` and `diffusion(time, state)` callables for the pretrained stochastic prior.
4. A `terminal(state)` map that returns the same shape as `state` and encodes the observation or physics constraint.
5. A trainable `torch.nn.Module` with `forward(time, state) -> control`, where `control.shape == state.shape`.
6. A BSDE `driver(time, state, value, control)` only when the formulation requires a nonzero driver. Otherwise use the default zero driver.

Read `README.md`, `bsde_diffusion/problem.py`, and `bsde_diffusion/solver.py` before implementing an adapter.

## Workflow

### 1. Formalize the inverse problem

Write down the unknown $x$, observation $y$, forward operator $\mathcal{A}$, feasible set $\mathcal{S}(y)$, and the prior's state space. Decide the terminal time $\tau$ and discretization before writing code.

Do not insert measurement gradients or projections into `drift` or `diffusion`. The framework's defining separation is:

- prior dynamics in `drift` and `diffusion`;
- task information in `terminal`;
- learned adapted correction in the control network.

### 2. Wrap and freeze the prior

Create an application adapter that exposes:

```python
def drift(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor: ...
def diffusion(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor: ...
```

Freeze all pretrained parameters with `requires_grad_(False)` and put the prior in evaluation mode. Preserve gradients only inside the trainable control network and `DeepBSDESolver.initial_value`.

### 3. Implement the terminal map

Prefer this structure when the prior and measurement spaces differ:

```python
def terminal(state):
    signal = decode(state)
    feasible_signal = enforce_measurement(signal, observation)
    return encode(feasible_signal)
```

`enforce_measurement` may be a closed-form projection, an iterative operator solve, or another domain-valid feasibility map. It must be deterministic for a fixed observation during one solver run and must preserve the batch dimension.

### 4. Choose the control network

Use `TimeConditionedMLP` for vectors or small tensors. For structured, high-dimensional states, implement a compact time-conditioned CNN, U-Net, transformer, or domain-native network. Its output is $Z_t$, not a reconstruction, and must exactly match the state shape.

### 5. Solve

```python
problem = AssociatedBSDE(
    initial_state=initial_state,
    time_grid=time_grid,
    drift=drift,
    diffusion=diffusion,
    terminal=terminal,
    driver=driver,  # omit for the zero driver
)

solver = DeepBSDESolver(problem, control).to(initial_state.device)
solution = solver.fit(
    SolverConfig(
        iterations=iterations,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_clip=1.0,
    )
)
recovered_anchor = solution.initial_value
```

Keep experiment logging, checkpointing, datasets, and visualization in the application project. Do not add them to the framework package.

### 6. Validate before claiming success

Add application tests for all of the following:

- output shapes, devices, and dtypes;
- finite prior fields and terminal values across the full time grid;
- zero trainable parameters in the frozen prior;
- decreasing terminal loss on a small deterministic fixture;
- terminal feasibility measured in observation space;
- reproducibility under a fixed random generator;
- stability across at least three discretizations or seeds.

Report both terminal mismatch and the domain metric $\|\mathcal{A}(x)-y\|$ (or its nonlinear analogue). A visually plausible output is not evidence of feasibility.

## Neighborhood sampling

After learning the anchor, characterize local uncertainty by perturbing it with an application-chosen scale and running the frozen generative dynamics. Report the perturbation distribution, scale, sample count, and both fidelity and diversity metrics. Do not present neighborhood samples as posterior samples without a separate justification.

## Release hygiene

Before sharing an adapter:

- scan tracked text and Git history for credentials, emails, absolute paths, hostnames, private dataset identifiers, and experiment notes;
- exclude data, checkpoints, outputs, caches, environment files, and generated figures;
- keep only the adapter, terminal map, tests, and concise usage documentation;
- run `pytest -q`, `ruff check .`, and `git diff --check`;
- inspect `git ls-files` and verify that every listed file is intentional.

