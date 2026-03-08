# Adaptive Kalman Filter (1D/2D) — Online Measurement Noise Estimation (R)

This repository implements an **Adaptive Kalman Filter** for **non-stationary measurement noise**, focusing on **online estimation of** `R(t)` (measurement noise covariance) using innovation statistics.

It also includes:
- **Diagnostics (NIS: Normalized Innovation Squared)**
- **Regime-shift handling** via slow/fast adaptation switching
---

## Key Ideas

A standard Kalman Filter assumes fixed noise statistics. In many real systems (eg:finance), measurement noise changes over time.

This project adapts `R(t)` online using:

- **IAE (Innovation-based Estimation)** to update measurement noise
- **NIS (Normalized Innovation Squared)** to diagnose filter consistency
- **Slow/Fast switching**: slow learning rate normally, fast rate during regime shifts

---

## What’s Included

###  Demos
- **1D** adaptive `R(t)` + NIS diagnostics
- **2D** constant-velocity model (state = position, velocity) with adaptive `R(t)` + NIS 

###  Diagnostics
- NIS curve comparison: fixed-`R` vs adaptive-`R`
- Interpretation: large sustained NIS often indicates underestimated uncertainty (e.g., `R` too small)

---
## Results

The experiments compare a **standard Kalman Filter with fixed measurement noise** and an **Adaptive Kalman Filter with online estimation of R(t)**.

The results show that the adaptive filter improves estimation performance in environments where measurement noise varies over time.

Key observations:

- The **Adaptive Kalman Filter tracks changes in measurement noise more effectively**.
- **NIS diagnostics remain closer to the theoretical expectation**, indicating improved filter consistency.
- During regime shifts (sudden increases in noise), the **fast adaptation mode allows the filter to adjust quickly**, preventing large estimation errors.

Overall, the adaptive approach produces **more stable state estimates and better uncertainty calibration** compared to the fixed-R Kalman Filter.



