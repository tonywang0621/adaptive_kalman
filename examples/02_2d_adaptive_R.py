from collections import deque

import sys
from pathlib import Path

# 讓 Colab / 直接跑 .py 時都能 import src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from adaptive_kf.core.kf import KalmanFilter
from adaptive_kf.adaptive.iae import update_R_iae
from adaptive_kf.diagnostics.nis import nis_value


def simulate_2d_constant_velocity(T=300, dt=1.0, q=0.2,
                                 r1=1.0, r2=9.0, change_point=150, seed=42):
    """
    2D state: x = [position, velocity]^T
    True dynamics:
        p_t = p_{t-1} + v_{t-1}*dt + noise
        v_t = v_{t-1} + noise
    Observation:
        y_t = position + measurement noise (R changes over time)
    """
    np.random.seed(seed)

    # State transition
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])  # observe position only

    # Simple process noise (for demo)
    Q = np.array([[q, 0.0],
                  [0.0, q]])

    # True time-varying measurement noise
    R_true = np.ones(T) * r1
    R_true[change_point:] = r2

    x = np.zeros((T, 2))
    x[0] = [0.0, 1.0]  # initial pos=0, vel=1

    # simulate state
    for t in range(1, T):
        w = np.random.multivariate_normal(mean=[0.0, 0.0], cov=Q)
        x[t] = (F @ x[t-1].reshape(2, 1)).reshape(-1) + w

    # simulate observation (position only)
    y = np.zeros(T)
    for t in range(T):
        y[t] = (H @ x[t].reshape(2, 1)).item() + np.random.normal(0, np.sqrt(R_true[t]))

    return x, y, R_true, F, H, Q


def run_2d_fixed_R(y, F, H, Q, r_fixed=1.0):
    R = np.array([[r_fixed]])
    x0 = np.array([0.0, 0.0])  # initial guess
    P0 = np.eye(2) * 10.0

    kf = KalmanFilter(F, H, Q, R, x0=x0, P0=P0)

    x_est = []
    nis_list = []

    for t in range(len(y)):
        kf.predict()
        _, _, nu, S = kf.update([y[t]])

        nis_list.append(nis_value(nu, S))
        x_est.append(kf.x.reshape(-1))  # [p, v]

    return np.array(x_est), np.array(nis_list)


def run_2d_adaptive_R(
    y, F, H, Q,
    r0=1.0,
    alpha_slow=0.01,
    alpha_fast=0.35,
    nis_threshold=3.84,
    cap_window=50,
    cap_mult=2.0,
    r_min=1e-6,
    max_fast_steps=5
):

  """
  自適應 R + NIS 觸發快慢更新 + rolling window cap
  cap_window: 滾動視窗長度
  cap_mult  : 上限倍數（median * cap_mult）
  """
  R = np.array([[r0]])
  x0 = np.array([0.0, 0.0])
  P0 = np.eye(2) * 10.0

  kf = KalmanFilter(F, H, Q, R, x0=x0, P0=P0)

  x_est = []
  R_est = []
  nis_list = []
  rmax_series = []

  # 滾動視窗：存最近 cap_window 個 R
  R_hist = deque([float(r0)] * cap_window, maxlen=cap_window)
  fast_count = 0


  for t in range(len(y)):
      kf.predict()
      _, _, nu, S = kf.update([y[t]])

      # NIS
      current_nis = nis_value(nu, S)
      nis_list.append(current_nis)

      # 快慢 alpha 切換
        # --- fast mode with max consecutive steps ---
      want_fast = current_nis > nis_threshold

      if want_fast and fast_count < max_fast_steps:
          a = alpha_fast
          fast_count += 1
      else:
          a = alpha_slow
          fast_count = 0


      # rolling cap：用最近 R 的 median 當 baseline
      baseline = float(np.median(np.array(R_hist)))
      r_max_t = max(baseline * cap_mult, float(kf.R[0, 0]))  # 至少不小於目前R
      rmax_series.append(r_max_t)

      # 更新 R（把 r_max 變成「每一步動態」）
      kf.R = update_R_iae(
          kf.R, nu, kf.H, kf.P_pred,
          alpha=a, eps=1e-6,
          r_min=r_min, r_max=r_max_t
      )

      # 更新視窗
      R_hist.append(float(kf.R[0, 0]))

      x_est.append(kf.x.reshape(-1))
      R_est.append(float(kf.R[0, 0]))

  return np.array(x_est), np.array(R_est), np.array(nis_list), np.array(rmax_series)



if __name__ == "__main__":
    # 1) simulate data
    T = 300
    x_true, y, R_true, F, H, Q = simulate_2d_constant_velocity(T=T, change_point=150)

    # 2) run filters
    x_fixed, nis_fixed = run_2d_fixed_R(y, F, H, Q, r_fixed=1.0)
  
    x_adapt, R_est, nis_adapt, rmax_series = run_2d_adaptive_R(
        y, F, H, Q,
        r0=1.0,
        alpha_slow=0.01,
        alpha_fast=0.35,
        nis_threshold=3.84,
        cap_window=50,
        cap_mult=2.0
    )



    # 3) plots: position estimate
    plt.figure()
    plt.plot(x_true[:, 0], label="True position p")
    plt.scatter(np.arange(T), y, s=10, label="Observation y (position)")
    plt.plot(x_fixed[:, 0], label="KF fixed R (pos)")
    plt.plot(x_adapt[:, 0], label="KF adaptive R (pos)")
    plt.title("2D Constant Velocity: Position (fixed R vs adaptive R)")
    plt.legend()
    plt.show()

    # 4) plot R tracking
    plt.figure()
    plt.plot(R_true, label="True R(t)")
    plt.plot(R_est, label="Estimated R(t) (adaptive)")
    plt.plot(rmax_series, linestyle="--", linewidth=1, label="Rolling cap r_max(t)")
    plt.title("2D: Adaptive measurement noise tracking (R) with rolling cap")
    plt.legend()
    plt.show()


    # 5) plot NIS
    plt.figure()
    plt.plot(nis_fixed, label="NIS (fixed R)")
    plt.plot(nis_adapt, label="NIS (adaptive R)")
    plt.axhline(1.0, linestyle="--", linewidth=1, label="Reference = 1")
    plt.title("2D: NIS comparison")
    plt.legend()
    plt.show()
