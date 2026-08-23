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


def soe_tail_kernel(alpha: float, L: int, p: int, w: np.ndarray, j_max: int,
                     tail_fit_points: int = 400, tail_fit_reg: float = 0.1):
    """Full classical construction: decay rates from quadrature, weights
    refit to the exact kernel. On its own, (lambda, c) here is the
    certified classical filter (carries the formal error bound derived
    in the theory document) -- zero data required.

    tail_fit_points: how many points along the tail get checked when
    fitting c against the exact kernel (see soe_refit_weights). More
    points means a more thorough, slightly slower one-time fit; 400 is
    the library's own long-standing default. tail_fit_reg: how strongly
    that fit is discouraged from picking extreme-sized weights (also
    passed straight to soe_refit_weights)."""
    m_max = j_max - L
    lam, _c0 = soe_nodes_and_weights(alpha, L, m_max, p)
    c = soe_refit_weights(alpha, L, w, lam, j_max,
                           n_samples=tail_fit_points, reg=tail_fit_reg)
    return lam, c


def soe_tail_error(alpha: float, L: int, w: np.ndarray, lam: np.ndarray, c: np.ndarray,
                    j_max: int, n_probe: int = 200) -> float:
    """Worst-case relative error of the (lambda, c) tail reconstruction
    against the exact GL tail weights, sampled at log-spaced probe
    points beyond L -- the same error the classical bound is certifying,
    evaluated directly rather than bounded."""
    m_max = max(j_max - L, 1)
    m_grid = np.unique(np.round(np.exp(np.linspace(0, np.log(m_max), n_probe))).astype(int))
    m_grid = m_grid[m_grid <= m_max]
    target = w[L + m_grid]
    approx = (lam[None, :] ** m_grid[:, None]) @ c
    rel_err = np.abs(approx - target) / np.maximum(np.abs(target), 1e-300)
    return float(np.max(rel_err))


def adaptive_soe_tail_kernel(alpha: float, L: int, w: np.ndarray, j_max: int,
                              tol: float = 1e-3, p_min: int = 4, p_max: int = 64,
                              tail_fit_points: int = 400, tail_fit_reg: float = 0.1):
    """Like soe_tail_kernel, but chooses the mode count p for you: starts
    at p_min and doubles until the worst-case relative tail error (see
    soe_tail_error) is <= tol or p_max is reached. Trades a slightly
    slower design step for not having to hand-pick p per alpha/L/budget.
    tail_fit_points, tail_fit_reg: same meaning as in soe_tail_kernel,
    forwarded unchanged at every p tried. Returns
    (lambda, c, p_used, achieved_err)."""
    p = min(p_min, p_max)
    while True:
        lam, c = soe_tail_kernel(alpha, L, p, w, j_max,
                                  tail_fit_points=tail_fit_points, tail_fit_reg=tail_fit_reg)
        err = soe_tail_error(alpha, L, w, lam, c, j_max)
        if err <= tol or p >= p_max:
            return lam, c, p, err
        p = min(p * 2, p_max)
