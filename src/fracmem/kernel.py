"""
Grunwald-Letnikov (GL) fractional derivative kernel -- the exact target
this whole library exists to approximate cheaply.

D^alpha x(t_k) ~= h^-alpha * sum_{j=0}^{k} w_j * x_{k-j}

w_0 = 1, w_j = w_{j-1} * (1 - (1+alpha)/j)      (exact GL recursion)
w_j ~ j^(-alpha-1) / |Gamma(-alpha)|             (algebraic tail, j -> infinity)

See the accompanying theory document for full derivations and proofs.
"""
import numpy as np


def gl_weights(alpha: float, n: int) -> np.ndarray:
    """Exact GL weights w_0..w_{n-1} via the stable recursion (never large
    Gamma-function cancellations)."""
    w = np.empty(n, dtype=np.float64)
    w[0] = 1.0
    for j in range(1, n):
        w[j] = w[j - 1] * (1.0 - (1.0 + alpha) / j)
    return w


def full_gl_derivative(x: np.ndarray, alpha: float, h: float, w: np.ndarray) -> np.ndarray:
    """The gold-standard, exact (but O(n)-memory, O(n) compute per new
    sample) fractional derivative, via direct convolution against the
    full weight sequence. Reference / validation use only -- not what
    gets deployed."""
    n = len(x)
    y = np.convolve(x, w[:n])[:n]
    return y * (h ** (-alpha))


def local_exact_term(x: np.ndarray, alpha: float, h: float, L: int, w: np.ndarray) -> np.ndarray:
    """The near/local sum j=0..L-1, computed exactly -- an ordinary
    length-L FIR filter, never approximated by anything in this library."""
    y = np.convolve(x, w[:L])[:len(x)]
    return y * (h ** (-alpha))


def delay(x: np.ndarray, L: int) -> np.ndarray:
    """x shifted right by L samples, zero-padded -- the tail filter's own
    input only starts once the exact local window has consumed its L
    samples. Works on a single 1-D signal or a (batch, time) 2-D array."""
    out = np.zeros_like(x)
    if x.ndim == 1:
        if len(x) > L:
            out[L:] = x[:len(x) - L]
    else:
        n = x.shape[-1]
        if n > L:
            out[..., L:] = x[..., :n - L]
    return out
