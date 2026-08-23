"""A small, explicit PINN implementation reused by Videos 1 and 3."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.func import jacfwd, vmap


@dataclass
class PinnResult:
    model: nn.Module
    horizon: float
    x0: np.ndarray
    final_loss: float


class TimeNetwork(nn.Module):
    """A small fully connected tanh network from normalized time to R^2."""

    def __init__(self, width=32, depth=3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, normalized_time):
        return self.network(normalized_time)


def _hard_initial_condition(model, t, horizon, x0):
    """N_theta(t) = x0 + t Ntilde_theta(2t/T - 1)."""
    normalized_time = 2.0 * t / horizon - 1.0
    return x0 + t * model(normalized_time)


def train_damped_oscillator_pinn(
    horizon,
    *,
    gamma=0.05,
    x0=(1.0, 0.0),
    epochs=4000,
    points_per_unit=16,
    width=32,
    depth=3,
    seed=0,
    learning_rate=1e-3,
    lbfgs_steps=0,
):
    """Minimize the residual on uniformly sampled collocation times."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    dtype = torch.float64
    model = TimeNetwork(width=width, depth=depth).to(dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_collocation = int(points_per_unit * horizon) + 1
    collocation_generator = torch.Generator().manual_seed(seed)
    x0_tensor = torch.tensor(x0, dtype=dtype)

    def state_at_time(single_time):
        """Evaluate the hard-IC network at one time with shape (1,)."""
        return _hard_initial_condition(model, single_time, horizon, x0_tensor)

    batched_state = vmap(state_at_time)
    batched_time_jacobian = vmap(jacfwd(state_at_time))

    def residual_loss(t):
        state = batched_state(t)
        # Each Jacobian has shape (state dimension, time dimension) = (2, 1).
        dstate_dt = batched_time_jacobian(t).squeeze(-1)
        q, p = state[:, 0], state[:, 1]
        dq_dt, dp_dt = dstate_dt[:, 0], dstate_dt[:, 1]
        return torch.mean((dq_dt - p) ** 2 + (dp_dt + q + gamma * p) ** 2)

    final_loss = float("nan")
    for _ in range(epochs):
        # Fresh Monte Carlo points approximate the residual integral each epoch.
        t = horizon * torch.rand(
            (n_collocation, 1), dtype=dtype, generator=collocation_generator
        )
        optimizer.zero_grad()
        loss = residual_loss(t)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    if lbfgs_steps:
        # Deterministic full-batch polishing. Video 1 applies the same optimizer
        # to both horizons; Video 3 leaves this off and varies Adam epochs alone.
        # L-BFGS needs the closure to return the same objective during its line
        # search, so one final random batch is sampled and then held fixed.
        t = horizon * torch.rand(
            (n_collocation, 1), dtype=dtype, generator=collocation_generator
        )
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_steps,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = residual_loss(t)
            loss.backward()
            return loss

        optimizer.step(closure)
        final_loss = float(closure().detach())

    return PinnResult(model, float(horizon), np.asarray(x0), final_loss)


def evaluate_pinn(result, times):
    """Evaluate a trained hard-IC network without building a gradient graph."""
    times = np.asarray(times, dtype=float)
    dtype = next(result.model.parameters()).dtype
    t = torch.tensor(times, dtype=dtype).reshape(-1, 1)
    x0 = torch.tensor(result.x0, dtype=dtype).reshape(1, 2)
    with torch.no_grad():
        state = _hard_initial_condition(result.model, t, result.horizon, x0)
    return state.cpu().numpy()
