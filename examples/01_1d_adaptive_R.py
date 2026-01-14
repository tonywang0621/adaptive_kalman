import sys
from pathlib import Path

# 將 repo 的 src/ 加入 Python 模組搜尋路徑，確保可以 import adaptive_kf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import numpy as np
import matplotlib.pyplot as plt

from adaptive_kf.core.kf import KalmanFilter
from adaptive_kf.adaptive.iae import update_R_iae


def simulate_1d_random_walk(T=400, q=0.2, r1=0.5, r2=4.0, change_point=200, seed=42):
    """
    產生 1D random walk + measurement noise R(t) 會在 change_point 之後變大。
    """
    np.random.seed(seed)

    x = np.zeros(T)
    R_true = np.ones(T) * r1
    R_true[change_point:] = r2

    # state evolves
    for t in range(1, T):
        x[t] = x[t - 1] + np.random.normal(0, np.sqrt(q))

    # noisy observation
    y = x + np.random.normal(0, np.sqrt(R_true))
    return x, y, R_true


def run_kf_fixed_R(y, q=0.2, r_fixed=0.5):
    """
    固定 R 的 Kalman Filter
    """
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[q]])
    R = np.array([[r_fixed]])

    kf = KalmanFilter(F, H, Q, R, x0=[0.0], P0=np.array([[1.0]]))

    x_est = []
    for t in range(len(y)):
        kf.predict()
        kf.update([y[t]])
        x_est.append(float(kf.x[0, 0]))
    return np.array(x_est)


def run_kf_adaptive_R(y, q=0.2, r0=0.5, alpha=0.05):
    """
    自適應 R 的 Kalman Filter（IAE）
    """
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[q]])
    R = np.array([[r0]])

    kf = KalmanFilter(F, H, Q, R, x0=[0.0], P0=np.array([[1.0]]))

    x_est = []
    R_est = []
    for t in range(len(y)):
        kf.predict()
        _, _, nu, _ = kf.update([y[t]])

        # 用 innovation 更新 R
        kf.R = update_R_iae(kf.R, nu, kf.H, kf.P_pred, alpha=alpha, eps=1e-6)

        x_est.append(float(kf.x[0, 0]))
        R_est.append(float(kf.R[0, 0]))

    return np.array(x_est), np.array(R_est)


if __name__ == "__main__":
    # 1) 產生資料：R 在一半後變大
    T = 400
    q = 0.2
    x_true, y, R_true = simulate_1d_random_walk(T=T, q=q, change_point=200)

    # 2) 跑兩種 KF
    x_fixed = run_kf_fixed_R(y, q=q, r_fixed=0.5)
    x_adapt, R_est = run_kf_adaptive_R(y, q=q, r0=0.5, alpha=0.05)

    # 3) 簡單評估
    rmse_fixed = np.sqrt(np.mean((x_fixed - x_true) ** 2))
    rmse_adapt = np.sqrt(np.mean((x_adapt - x_true) ** 2))
    print(f"RMSE (fixed R)   : {rmse_fixed:.4f}")
    print(f"RMSE (adaptive R): {rmse_adapt:.4f}")

    # 4) 圖 1：狀態估計比較
    plt.figure()
    plt.plot(x_true, label="True state x")
    plt.scatter(np.arange(T), y, s=10, label="Observation y")
    plt.plot(x_fixed, label="KF (fixed R)")
    plt.plot(x_adapt, label="KF (adaptive R)")
    plt.title("1D Random Walk: Fixed R vs Adaptive R")
    plt.legend()
    plt.show()

    # 5) 圖 2：R 追蹤結果
    plt.figure()
    plt.plot(R_true, label="True R(t)")
    plt.plot(R_est, label="Estimated R(t) (adaptive)")
    plt.title("Adaptive measurement noise tracking (R)")
    plt.legend()
    plt.show()
