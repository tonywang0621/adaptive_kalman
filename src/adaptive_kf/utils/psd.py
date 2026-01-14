import numpy as np


def ensure_psd(A, eps=1e-6):
    """
    Make matrix symmetric PSD by eigenvalue clipping.
    - For 1D / 1x1: ensure value >= eps
    - For matrix: symmetrize + clip negative eigenvalues
    """
    A = np.array(A, dtype=float)

    # scalar case
    if A.ndim == 0:
        return np.array(max(float(A), eps))

    # 1x1 case
    if A.shape == (1, 1):
        return np.array([[max(float(A[0, 0]), eps)]])

    # general matrix case
    A = 0.5 * (A + A.T)  # symmetrize
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    A_psd = (V * w) @ V.T
    A_psd = 0.5 * (A_psd + A_psd.T)
    return A_psd
