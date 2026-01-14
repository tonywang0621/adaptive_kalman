import numpy as np


class KalmanFilter:
    """
    Minimal Kalman Filter (linear Gaussian)

    State:
        x_t = F x_{t-1} + w,   w ~ N(0, Q)
    Obs:
        y_t = H x_t + v,       v ~ N(0, R)

    Notes:
    - x is column vector (n x 1)
    - y is column vector (m x 1)
    """

    def __init__(self, F, H, Q, R, x0, P0):
        self.F = np.array(F, dtype=float)
        self.H = np.array(H, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

        self.x = np.array(x0, dtype=float).reshape(-1, 1)
        self.P = np.array(P0, dtype=float)

        # cache for adaptive methods
        self.x_pred = None
        self.P_pred = None
        self.nu = None   # innovation
        self.S = None    # innovation covariance
        self.K = None    # Kalman gain

    def predict(self):
        self.x_pred = self.F @ self.x
        self.P_pred = self.F @ self.P @ self.F.T + self.Q
        return self.x_pred, self.P_pred

    def update(self, y):
        y = np.array(y, dtype=float).reshape(-1, 1)

        # innovation (residual)
        self.nu = y - (self.H @ self.x_pred)

        # innovation covariance
        self.S = self.H @ self.P_pred @ self.H.T + self.R

        # Kalman gain
        self.K = self.P_pred @ self.H.T @ np.linalg.inv(self.S)

        # posterior update
        self.x = self.x_pred + self.K @ self.nu
        I = np.eye(self.P.shape[0])
        self.P = (I - self.K @ self.H) @ self.P_pred

        return self.x, self.P, self.nu, self.S
