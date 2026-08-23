"""ODE right-hand sides used in more than one video."""

import numpy as np


def damped_oscillator(_t, x, gamma=0.05):
    """q' = p, p' = -q - gamma p."""
    q, p = x
    return np.array([p, -q - gamma * p])


def harmonic_oscillator(_t, x):
    """q' = p, p' = -q."""
    q, p = x
    return np.array([p, -q])

