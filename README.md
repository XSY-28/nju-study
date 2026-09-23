# nju-study

**English** | [简体中文](README.zh-CN.md)

**Learn online learning through small NumPy implementations, worked examples, and reproducible regret experiments.**

This repository implements Follow the Leader (FTL) for squared loss and compares Online Gradient Descent (OGD) with FTL on alternating linear losses. It connects the update rules to executable code, hand calculations, and tests that check the online timing: make a decision first, then observe the current loss.

| Experiment | What you can explore |
|---|---|
| [FTL with squared loss](#ftl-with-squared-loss) | Why minimizing past squared losses gives the historical mean, and how to compute static regret |
| [OGD vs. FTL with linear loss](#ogd-vs-ftl-with-linear-loss) | How FTL can incur linear regret on an alternating sequence while projected OGD has a square-root regret bound |

These are learning experiments, not a general-purpose optimization library. The documentation is self-contained; no separate course notes are required. Source comments and command-line output currently remain in Chinese. Function names, command-line options, and the English explanations below provide the corresponding usage and interpretation.

## Quick start

Clone the repository and run commands from its root. The setup below uses a POSIX shell and keeps the virtual environment outside the repository:

```bash
git clone https://github.com/XSY-28/nju-study.git
cd nju-study
python3 -m venv ~/.venvs/nju-study
source ~/.venvs/nju-study/bin/activate
python -m pip install numpy==2.5.2 pytest==9.1.1
python -m pytest -q
python -m experiments.run_ftl_square --rounds 400 --seed 42
python -m experiments.run_ogd_linear --rounds 400
```

Previously tested environment: macOS 26.6.2, Python 3.13.14, NumPy 2.5.2, and pytest 9.1.1. This is a reproducibility reference, not a claim of testing on every platform.

Expected reference results for these commands:

| Experiment | Quantity | Value |
|---|---|---:|
| Squared-loss FTL, 400 rounds, seed 42 | Static regret | ≈ 2.108443 |
| Linear-loss OGD, 400 rounds | Static regret | 20.45 |
| Linear-loss FTL, 400 rounds | Static regret | 399.5 |
| Linear-loss OGD, 400 rounds | Theoretical regret bound | 40 |

The squared-loss experiment samples binary observations using the specified seed. The linear-loss experiment is deterministic and needs no seed. Numerical results help check the implementation; they do not replace a proof.

## Repository layout

| Path | Contents |
|---|---|
| [src/ftl.py](src/ftl.py) | Squared-loss FTL predictions and cumulative static regret |
| [src/ogd.py](src/ogd.py) | Linear-loss OGD/FTL decisions and static regret |
| [src/_validation.py](src/_validation.py) | Shared input validation |
| [tests/](tests/) | Worked examples, boundary cases, input validation, and online-timing checks |
| [experiments/](experiments/) | Reproducible command-line experiments |

## FTL with squared loss

At round t, predict a value, then observe the outcome. The loss is

$$
f_t(x_t)=(x_t-y_t)^2.
$$

The decision domain is the real line. If the observations and initial prediction are all in [0, 1], the predictions stay in that interval.

### Update rule

FTL selects a decision that minimizes the accumulated **past** losses. Differentiating the sum of past squared losses and setting the derivative to zero gives the historical mean:

$$
x_t=\frac{1}{t-1}\sum_{s=1}^{t-1}y_s,\qquad t\ge 2.
$$

There are no past observations at the first round, so the default initial prediction is 0.5. With zero-based indexing, the implementation updates the mean as follows:

```python
x[i + 1] = x[i] + (y[i] - x[i]) / (i + 1)
```

After observing `y[i]`, this updates the **next** prediction. It does not use the current answer to make the current prediction. Each update takes constant time; computing and storing all T predictions takes O(T) time and O(T) space.

### Static regret and API

The best fixed decision in hindsight is the mean of the entire sequence:

$$
u=\frac{1}{T}\sum_{t=1}^{T}y_t.
$$

Cumulative static regret compares the algorithm's loss with the loss of that single fixed decision:

$$
R_T=\sum_{t=1}^{T}(x_t-y_t)^2-\sum_{t=1}^{T}(u-y_t)^2.
$$

The comparator u is used only for evaluation after the sequence is known. It does not participate in online predictions, and it is not a different optimal decision at every round.

- `ftl_square(y)` returns cumulative static regret.
- `ftl_square_predictions(y)` returns the prediction sequence.
- Both accept a nonempty, one-dimensional sequence of finite real values. The `x0` parameter sets the first prediction and must be a finite real scalar, not a list or one-dimensional array. Complex inputs are rejected.

### Worked example

For `y = [0, 1]`, the default predictions are `[0.5, 0]`:

- Algorithm loss: 0.5² + (0 − 1)² = 1.25.
- Best fixed decision: u = 0.5, with cumulative loss 0.5.
- Static regret: 1.25 − 0.5 = **0.75**.

See [the tests](tests/test_ftl.py) for hand-calculated examples, boundary inputs, and checks that predictions do not use current or future observations.

### Reproduce the experiment

```bash
python -m experiments.run_ftl_square --rounds 400 --seed 42
```

The script prints the initial observations and predictions, the algorithm's cumulative loss, the best fixed decision's cumulative loss, and regret. The default regret is approximately **2.108443** in the reference environment. Keep the environment, round count, and seed unchanged to reproduce it. This experiment does not establish FTL guarantees for arbitrary loss functions.

## OGD vs. FTL with linear loss

This deterministic example uses alternating gradients to illustrate the difference between FTL and projected OGD.

### Experiment sequence

Odd rounds have positive gradients and even rounds have negative gradients, except that the first gradient is 0.5:

```python
t = np.arange(1, T + 1)
g = np.where(t % 2 == 1, 1.0, -1.0)
g[0] = 0.5
```

The floating-point branches `1.0` and `-1.0` matter: integer branches would create an integer array, and assigning `0.5` would truncate it to `0`, changing the experiment.

### OGD update and bound

At round t, choose a decision in [−1, 1] before observing the gradient. The linear loss is

$$
f_t(x)=g_t x.
$$

Starting from zero, OGD uses a fixed step size and projects the updated decision back into the interval:

$$
x_{t+1}=\min\{1,\max\{-1,x_t-\eta g_t\}\},\qquad \eta=\frac{2}{\sqrt{T}}.
$$

The projection is implemented by `np.clip(x_t - eta * g_t, -1, 1)`. The step size `eta` must be a finite positive real scalar, and the initial decision must lie in the interval. Save the current decision **before** applying the current gradient so that each decision depends only on past information.

The interval has diameter D = 2, and this experiment has gradient bound G = 1. The standard OGD static-regret bound with the chosen step size is

$$
R_T\le DG\sqrt{T}=2\sqrt{T}.
$$

For a different gradient sequence, choose the step size and bound using its actual gradient bound.

### Comparator and FTL

A fixed decision u incurs cumulative loss equal to u times the sum of gradients. The optimal fixed decision is −1 when that sum is positive and 1 when it is negative; every feasible decision is optimal when the sum is zero. Therefore,

$$
\min_{u\in[-1,1]}\sum_{t=1}^{T}g_tu=-\left|\sum_{t=1}^{T}g_t\right|,
\qquad
R_T=\sum_{t=1}^{T}g_tx_t+\left|\sum_{t=1}^{T}g_t\right|.
$$

FTL instead chooses its decision by minimizing only the losses observed before the current round:

$$
x_t=-\operatorname{sign}\!\left(\sum_{s=1}^{t-1}g_s\right).
$$

Here sign returns 1, −1, or 0 for a positive, negative, or zero input, respectively. On this alternating sequence, FTL repeatedly jumps between the interval's endpoints and incurs loss. OGD makes smaller updates and has much lower regret in this example.

`ogd_linear` and `ftl_linear` return the per-round decisions; `linear_regret` computes their static regret. See [the tests](tests/test_ogd.py) for hand calculations, the 400-round result, projection boundaries, online timing, and invalid inputs.

### Reproduce the comparison

```bash
python -m experiments.run_ogd_linear --rounds 400
```

The gradients start with `[0.5, -1, 1, -1, 1]`; OGD's decisions start with `[0, -0.05, 0.05, -0.05, 0.05]`. At 400 rounds:

| Metric | Value |
|---|---:|
| Best fixed decision's cumulative loss | −0.5 |
| OGD cumulative loss | 19.95 |
| OGD static regret | 20.45 |
| FTL static regret | 399.5 |
| OGD theoretical regret bound | 40 |

Changing `--rounds` also recalculates the step size from the new horizon. This example illustrates behavior on a specific sequence, not universal superiority of one algorithm over another.
