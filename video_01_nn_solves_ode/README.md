# Can a Neural Network Solve a Differential Equation?

This experiment asks whether a small neural network can represent the solution
of one fixed initial-value problem by minimizing its differential-equation
residual globally in time.

For the mildly damped oscillator

\[
q'=p, \qquad p'=-q-0.05p, \qquad (q(0),p(0))=(1,0),
\]

the script trains the same 3-hidden-layer, width-32 tanh network on $T=10$
and $T=40$. The initial condition is exact because the prediction has the
form $x_0+t\widehat N_\theta(t)$. Both runs use Adam for 4000 epochs,
followed by at most 400 full-batch L-BFGS iterations, with seed 7 and 16
collocation points per time unit. Collocation times are sampled uniformly on
$[0,T]$ at every Adam epoch; L-BFGS uses one fixed final random batch because its line search requires a deterministic objective. A tight-tolerance SciPy `DOP853` solution is the reference; errors are evaluated on a separate dense grid (not merely at the collocation points).

## Run

From the repository root, install dependencies and run:

A system LaTeX installation is required for the repository's Matplotlib style.

```bash
python -m pip install -r requirements.txt
python video_01_nn_solves_ode/run.py
```

The script reports residual, RMS, and maximum state errors and creates:

- `figures/short_position.png` and `figures/short_phase.png`;
- `figures/long_position.png` and `figures/long_phase.png`.

The result illustrates that this straightforward global residual-minimization formulation becomes harder as the horizon grows. It does **not** show that neural networks cannot solve long-time ODEs. Accuracy is sensitive to network, optimizer, sampling, and training choices.

The plots showed in the video used $\gamma=0.04$ instead of the written $\gamma=0.05$. The results do not change qualitatively, and now $0.05$ is the default in the code.

The YouTube URL will be added when the video is published.
