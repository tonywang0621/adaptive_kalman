import numpy as np
from adaptive_kf.utils.psd import ensure_psd


def update_R_iae(R_prev, nu, H, P_pred, alpha=0.05, eps=1e-6):
    """
    Innovation-based Adaptive Estimation (IAE) for R.

    nu: innovation = y - H x_pred
    S_theory = H P_pred H^T + R

    Covariance matching idea:
        R ≈ E[nu nu^T] - H P_pred H^T

    We use EMA:
        R_t = (1-alpha) R_{t-1} + alpha * (nu nu^T - H P_pred H^T)

    Then enforce PSD and minimum eps.
    """
    R_prev = np.array(R_prev, dtype=float)
    nu = np.array(nu, dtype=float).reshape(-1, 1)
    H = np.array(H, dtype=float)
    P_pred = np.array(P_pred, dtype=float)

    target = (nu @ nu.T) - (H @ P_pred @ H.T)
    R_new = (1 - alpha) * R_prev + alpha * target

    # make sure R stays valid (>= eps / PSD)
    R_new = ensure_psd(R_new, eps=eps)

        # optional: cap R to avoid overshoot (works well for 1D measurement)
    grow_limit = 1.5  # 每一步最多放大 1.5 倍
    R_cap = float(R_prev[0, 0]) * grow_limit
    R_new = np.array([[min(float(R_new[0, 0]), R_cap)]])


    return R_new
