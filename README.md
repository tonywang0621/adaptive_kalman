# Adaptive Kalman Filter (1D/2D) — Online Measurement Noise Estimation (R)

This repository implements an **Adaptive Kalman Filter** for **non-stationary measurement noise**, focusing on **online estimation of** `R(t)` (measurement noise covariance) using innovation statistics.

It also includes:
- **Diagnostics (NIS: Normalized Innovation Squared)**
- **Regime-shift handling** via slow/fast adaptation switching
- **Practical safeguards** (caps, auto floor, overshoot control)

This design is intended to be a clean “method/toolbox repo” that can later connect to **AI/ML** and **finance** use-cases (e.g., time-varying volatility, changing market regimes).

---

## Key Ideas

A standard Kalman Filter assumes fixed noise statistics. In many real systems (and finance), measurement noise changes over time.

This project adapts `R(t)` online using:

- **IAE (Innovation-based Estimation)** to update measurement noise
- **NIS (Normalized Innovation Squared)** to diagnose filter consistency
- **Slow/Fast switching**: slow learning rate normally, fast rate during regime shifts
- **Safeguards**
  - `max_fast_steps`: limits consecutive fast steps to reduce overshoot
  - Rolling cap `r_max(t)`: prevents `R` from exploding
  - **Auto floor**: prevents cap from being too low (so `R` is not stuck below the high-noise regime)

---

## What’s Included

### ✅ Demos
- **1D** adaptive `R(t)` + NIS diagnostics
- **2D** constant-velocity model (state = position, velocity) with adaptive `R(t)` + NIS + safeguards

### ✅ Diagnostics
- NIS curve comparison: fixed-`R` vs adaptive-`R`
- Interpretation: large sustained NIS often indicates underestimated uncertainty (e.g., `R` too small)

---

## Repository Structure

