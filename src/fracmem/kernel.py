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


def rl_derivative(x: np.ndarray, alpha: float, h: float, w: np.ndarray) -> np.ndarray:
    """Riemann-Liouville derivative, lower terminal at t=0. For a causal
    signal (x(t)=0, t<0) the GL discretization IS the standard numerical
    approximation of the RL derivative -- same convolution, different
    name so callers can express intent. See Podlubny, Ch. 2/3 for the
    GL/RL equivalence."""
    return full_gl_derivative(x, alpha, h, w)


def caputo_derivative(x: np.ndarray, alpha: float, h: float, w: np.ndarray) -> np.ndarray:
    """Caputo derivative, 0 < alpha < 1: D^alpha_C x(t) = D^alpha_RL
    [x(t) - x(0)](t). Subtracting the (constant) initial value removes
    the t^-alpha singularity the RL derivative has at t=0 for a
    nonzero-initial-value signal -- everything downstream is then the
    ordinary GL/RL machinery applied to the shifted signal."""
    x = np.asarray(x, dtype=np.float64)
    x0 = x[0] if x.ndim == 1 else x[..., :1]
    return full_gl_derivative(x - x0, alpha, h, w) if x.ndim == 1 \
        else np.stack([full_gl_derivative(xi, alpha, h, w) for xi in (x - x0)])


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
