# Boundary-Smooth Constraint Research Notes

## Goal

Replace the previous hard constrained-sampling direction with a softer
"boundary smoothness constraint" that only regularizes the transition between
the observed history and the generated forecast window.

Target insertion point in this repo:

- `main_model.py`, inside `CSDI_base.impute()`
- current reverse sampling loop starts at `main_model.py:425`
- current CFG / trend-aware guidance is applied around `main_model.py:492`

This keeps training unchanged and confines the change to inference-time
sampling.

## Why The Previous Constraint Route Failed

The previous route used stronger inference-time interventions such as hard
projection and guidance rescaling. The observed behavior on Economy was:

- best score moved to `guide_w = 0.0`
- larger guidance monotonically worsened MSE / MAE
- this indicates the constraint logic interfered with CFG / trend guidance
  instead of gently regularizing it

The likely failure mode is that the intervention acted on the whole trajectory
or directly distorted the guidance vector. For time-series forecasting, the
most fragile region is usually only the join between:

- history end: `x[:, :, lookback_len - 1]`
- forecast start: `x[:, :, lookback_len]`

So the next attempt should constrain only a short local window near the
boundary and avoid hard overwrite operations.

## Relevant Papers And What To Borrow

### 1. Universal Guidance for Diffusion Models

Source:

- https://arxiv.org/abs/2302.07121

Useful idea:

- Add arbitrary differentiable guidance at sampling time without retraining.

What we borrow:

- Treat boundary smoothness as a differentiable energy `E(x)`
- apply a small correction during reverse diffusion
- avoid architecture changes

Why it fits this repo:

- our model already applies inference-time guidance inside `impute()`
- adding one more lightweight energy term is structurally compatible

### 2. Diffusion Posterior Sampling for General Noisy Inverse Problems

Source:

- https://arxiv.org/abs/2209.14687

Useful idea:

- Use a blended guidance path rather than a strict consistency projection.

What we borrow:

- prefer soft energy shaping over hard projection
- keep the generative path near the model manifold

Why it matters here:

- our previous "harder" constraint attempt likely pushed samples off the useful
  forecasting manifold

### 3. RePaint

Source:

- https://arxiv.org/abs/2201.09865

Useful idea:

- Change only the reverse diffusion iterations instead of retraining the
  network.

What we borrow:

- inference-time conditioning only
- keep the base denoiser untouched

Why it matters here:

- same engineering philosophy as this project
- low-risk for regression outside the sampling path

### 4. CFG++

Source:

- https://arxiv.org/abs/2406.08070

Useful idea:

- standard CFG can move samples off-manifold, especially at larger guidance
  scales

What we borrow:

- do not aggressively manipulate the CFG delta itself
- keep any extra constraint weak, local, and manifold-friendly

Why it matters here:

- this matches the failed experiment pattern where increasing `guide_w`
  consistently hurt performance

### 5. SPG: Smooth Perturbation Guidance

Source:

- https://arxiv.org/abs/2503.02577

Useful idea:

- test-time smoothing can improve diffusion outputs without retraining, as long
  as the perturbation remains structure-preserving

What we borrow:

- smoothing should be weak and structured
- "negative" or corrective guidance can be useful if it preserves the sequence
  semantics

Why it matters here:

- our setting is also sequential, so boundary-local smoothing is more natural
  than hard clipping

## Recommended Direction For This Repo

### Core Principle

Constrain only the first few forecast steps near the join. Do not modify the
whole future window and do not hard-clamp the observed history every step.

Recommended local region:

- `t0 = lookback_len`
- smooth over `[t0, t0 + local_window)`
- default `local_window in {2, 3, 4}`

### Constraint Family

Use one or more of these local energies:

1. Boundary jump penalty

```text
E_jump = Huber(x_t0 - h_last)
```

Where:

- `h_last = observed_data[:, :, lookback_len - 1]`
- `x_t0 = current_sample[:, :, lookback_len]`

Purpose:

- reduce discontinuity between history end and prediction start

2. Boundary slope penalty

```text
E_slope = Huber((x_t1 - x_t0) - (h_last - h_prev))
```

Where:

- `h_prev = observed_data[:, :, lookback_len - 2]`
- `x_t1 = current_sample[:, :, lookback_len + 1]`

Purpose:

- match first-order trend at the transition

3. Local curvature penalty

```text
E_curve = Huber(x_t2 - 2*x_t1 + x_t0)
```

Purpose:

- suppress sharp bending immediately after the boundary

Recommended combined energy:

```text
E_boundary = w_jump * E_jump + w_slope * E_slope + w_curve * E_curve
```

Initial recommendation:

- `w_jump = 1.0`
- `w_slope = 0.5`
- `w_curve = 0.1`

## Two Implementation Options

### Option A: No-Grad Soft Boundary Blend

This is the lowest-risk version and should be tried first.

After each reverse update on `current_sample`, apply a small local blend:

```text
x_t0 <- h_last + (1 - eta_t) * (x_t0 - h_last)
```

And optionally:

```text
target_delta = h_last - h_prev
pred_delta = x_t1 - x_t0
x_t1 <- x_t0 + (1 - eta_t) * pred_delta + eta_t * target_delta
```

Where `eta_t` is small and decays over steps, for example:

```text
eta_t = boundary_smooth_scale * step_ratio
```

Why this is attractive:

- no autograd inside sampling
- very cheap
- highly local
- easy to ablate

### Option B: Differentiable Boundary Energy Guidance

This is closer to Universal Guidance / DPS.

At each step:

- make a temporary `current_sample_req = current_sample.detach().requires_grad_(True)`
- compute `E_boundary(current_sample_req)`
- take `grad = dE/dx`
- update only the forecast slice:

```text
current_sample[:, :, lookback_len:] -= lambda_t * grad[:, :, lookback_len:]
```

Why this is attractive:

- more principled
- easier to tune term-by-term

Why it is riskier:

- extra compute
- more sensitive to step size
- more likely to interact with CFG unexpectedly

## Recommendation

Implement Option A first.

Reason:

- the last failed route already showed that stronger guidance-side
  interventions are fragile
- Option A is soft, local, and solver-friendly
- if it works, we can later refine it into Option B

## Proposed Config Keys

Add these under `diffusion`:

```yaml
diffusion:
  boundary_smooth_constraint: false
  boundary_smooth_window: 3
  boundary_smooth_scale: 0.1
  boundary_smooth_decay_power: 1.0
  boundary_smooth_use_slope: true
  boundary_smooth_use_curvature: false
  boundary_smooth_slope_scale: 0.5
  boundary_smooth_curvature_scale: 0.1
```

If Option B is later added:

```yaml
diffusion:
  boundary_energy_guidance: false
  boundary_energy_step_scale: 0.02
  boundary_energy_huber_delta: 1.0
```

## How It Fits The Current Code

Best insertion point:

1. keep existing denoiser forward pass
2. keep existing CFG / trend-aware guidance
3. keep existing DDPM / DDIM reverse update
4. after `current_sample` is updated for a step, apply boundary smoothing

Pseudo-order:

```text
predicted = denoiser(...)
predicted = cfg_or_trend_cfg(...)
current_sample = reverse_update(current_sample, predicted, t)
current_sample = apply_boundary_smooth_constraint(current_sample, observed_data, cond_mask, t)
```

This avoids touching:

- training loss
- text encoder path
- scale router path
- multi-resolution auxiliary supervision

## Concrete Minimal Patch Plan

Step 1:

- add config parsing in `CSDI_base.__init__`
- add helper `_apply_boundary_smooth_constraint(...)`
- call it in `impute()` after each reverse step

Step 2:

- create one temporary Economy config for ablation
- compare against current best main config

Step 3:

- test three variants only:
  - jump only
  - jump + slope
  - jump + slope + weak curvature

## Default Experiment Order

Recommended order:

1. `jump only`, `scale = 0.05`
2. `jump + slope`, `scale = 0.05`, `slope_scale = 0.5`
3. `jump + slope`, `scale = 0.1`, `slope_scale = 0.5`

Do not start with curvature.

## Current Judgment

The most promising next move is:

- local soft boundary blend
- no hard projection
- no CFG delta rescale
- no full-window smoothing

This is the cleanest way to convert the failed "constraint sampling" idea into a
weaker continuity prior that better matches time-series forecasting.
