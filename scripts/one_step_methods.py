"""Small one-step integration helpers shared by the video experiments."""

import numpy as np


def explicit_euler_step(rhs, t, x, h, args=()):
    return x + h * rhs(t, x, *args)


def integrate_fixed_step(step, rhs, x0, t_final, h, *, args=()):
    """Compose a one-step map on a uniform grid ending exactly at t_final."""
    n_steps = round(t_final / h)
    if not np.isclose(n_steps * h, t_final):
        raise ValueError("t_final must be an integer multiple of h")

    times = np.linspace(0.0, t_final, n_steps + 1)
    states = np.empty((n_steps + 1, len(x0)), dtype=float)
    states[0] = x0
    for n in range(n_steps):
        states[n + 1] = step(rhs, times[n], states[n], h, args)
    return times, states
