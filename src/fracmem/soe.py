"""
Sum-of-Exponentials (SOE) construction: turns the slowly-decaying,
power-law GL tail kernel into a short sum of exponentials via a Gamma-
function integral identity, discretized by log-spaced quadrature. Pure
mathematics -- no signal data is used anywhere in this module.

    j^{-beta} = 1/Gamma(beta) * int_0^inf s^{beta-1} e^{-sj} ds,  beta = alpha+1

See the accompanying theory document (Theorem 5.1) for the derivation.
"""
import numpy as np
from scipy.special import gamma


def soe_nodes_and_weights(alpha: float, L: int, m_max: int, p: int):
    """Log-spaced quadrature discretization. Returns (lambda, c) with
    lambda_i = exp(-s_i), p nodes covering the integrand's significant
    range, log-uniformly spaced."""
    beta = alpha + 1.0
    A = 1.0 / abs(gamma(-alpha))

    s_min = 1.0 / max(m_max, 1)
    s_max = 8.0  # e^-8 per sample is already below any meaningful noise floor
    u_min, u_max = np.log(s_min), np.log(s_max)
    du = (u_max - u_min) / p
    u = u_min + (np.arange(p) + 0.5) * du
    s = np.exp(u)

    lam = np.exp(-s)
    c = (A / gamma(beta)) * du * (s ** beta) * np.exp(-s * L)
    return lam, c


def soe_refit_weights(alpha: float, L: int, w: np.ndarray, lam: np.ndarray, j_max: int,
                       n_samples: int = 400, reg: float = 0.1) -> np.ndarray:
    """Linear least-squares refit of the weights against the EXACT GL tail
    weights (never against any signal) -- corrects the small mismatch
    between the asymptotic power law and the true kernel near j=L."""
    m_grid = np.unique(np.round(np.exp(np.linspace(0, np.log(j_max - L), n_samples))).astype(int))
    m_grid = m_grid[m_grid <= (j_max - L)]
    j_grid = L + m_grid
    target = w[j_grid]

    M = lam[None, :] ** m_grid[:, None]
    wts = 1.0 / np.maximum(np.abs(target), 1e-300)  # fit RELATIVE error uniformly
    Mw = M * wts[:, None]
    tw = target * wts

    p = M.shape[1]
    Mw_reg = np.vstack([Mw, np.sqrt(reg) * np.eye(p)])
    tw_reg = np.concatenate([tw, np.zeros(p)])
    c, *_ = np.linalg.lstsq(Mw_reg, tw_reg, rcond=None)
    return c


def soe_tail_kernel(alpha: float, L: int, p: int, w: np.ndarray, j_max: int):
    """Full classical construction: decay rates from quadrature, weights
    refit to the exact kernel. On its own, (lambda, c) here is the
    certified classical filter (carries the formal error bound derived
    in the theory document) -- zero data required."""
    m_max = j_max - L
    lam, _c0 = soe_nodes_and_weights(alpha, L, m_max, p)
    c = soe_refit_weights(alpha, L, w, lam, j_max)
    return lam, c
