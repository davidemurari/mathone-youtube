"""High-accuracy reference integration shared by the experiments."""

import numpy as np
from scipy.integrate import solve_ivp


def reference_solution(rhs, x0, t_eval, *, args=()):
    """Evaluate a tight-tolerance DOP853 solution at requested times."""
    t_eval = np.asarray(t_eval, dtype=float)
    solution = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        np.asarray(x0, dtype=float),
        args=args,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    if not solution.success:
        raise RuntimeError(f"Reference integration failed: {solution.message}")
    return solution.y.T

