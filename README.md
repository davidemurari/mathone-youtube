# Mathone video experiments

Minimal, reproducible computational experiments accompanying the
[Mathone YouTube channel](https://www.youtube.com/@MathoneDavide).

Install the small scientific Python stack once:

Python 3.10--3.12 is recommended (and was used for the validated runs).
A working LaTeX installation is also required because the repository plot style uses Matplotlib's `text.usetex=True`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run each experiment from the repository root.

| Video | Working title | Folder | Experiment |
|---:|---|---|---|
| 1 | Can a Neural Network Solve a Differential Equation? | [`video_01_nn_solves_ode/`](video_01_nn_solves_ode/) | A residual-trained neural solution of one damped-oscillator IVP on short and long intervals. |


YouTube links will be added to the individual READMEs when the videos are
published.
