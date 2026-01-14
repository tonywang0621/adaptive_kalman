import numpy as np


def nis_value(nu: np.ndarray, S: np.ndarray) -> float:
    """
    NIS = nu^T S^{-1} nu

    nu: innovation (m x 1)
    S : innovation covariance (m x m)
    """
    nu = np.array(nu, dtype=float).reshape(-1, 1)
    S = np.array(S, dtype=float)

    # 用 solve 比 inv 穩定
    x = np.linalg.solve(S, nu)
    return float((nu.T @ x).squeeze())

