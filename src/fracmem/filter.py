"""
The deployed recursive filter, the data-driven weight fit, and the
assembled end-to-end pipeline.

Design principle: the decay rates (lambda) come from an exact
mathematical identity (see soe.py) and are never touched by data. Only
the linear readout weights (c) are fit to data, via cross-validated
ridge regression -- a convex, uniquely-solvable problem once lambda is
fixed. See the accompanying theory document for the full justification.
"""
import numpy as np
from scipy.signal import lfilter

from .kernel import gl_weights, full_gl_derivative, local_exact_term, delay
from .soe import soe_tail_kernel

DEFAULT_REG_CANDIDATES = (1e-11, 3e-11, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 1e-7, 1e-6, 1e-5)


def diagonal_recurrence(x_delayed: np.ndarray, lam: np.ndarray, c: np.ndarray) -> np.ndarray:
    """THE deployed filter: p independent one-line recursions.
        m_i[k] = lam_i * m_i[k-1] + x_delayed[k]
        out[k] = c . m[k]
    O(p) multiply-adds per sample, O(p) persistent memory -- constant,
    regardless of how long the signal runs."""
    n = len(x_delayed)
    p = len(lam)
    m = np.zeros(p)
    out = np.empty(n)
    for k in range(n):
        m = lam * m + x_delayed[k]
        out[k] = c @ m
    return out


def mode_features(lam: np.ndarray, x_delayed: np.ndarray) -> np.ndarray:
    """Fast batched computation of every mode's feature trace, for FITTING
    only -- (p, B, T) array, mathematically identical to
    diagonal_recurrence's own state trajectory."""
    return np.stack([lfilter([1.0], [1.0, -li], x_delayed, axis=-1) for li in lam], axis=0)


def ridge_solve_c(features: np.ndarray, target: np.ndarray, reg: float) -> np.ndarray:
    """c* = (F F^T + reg*diag(diag(F F^T)))^{-1} F t
    -- per-mode-energy-scaled ridge regularization."""
    p = features.shape[0]
    F = features.reshape(p, -1)
    t = target.reshape(-1)
    G = F @ F.T
    reg_matrix = reg * np.diag(np.diag(G) + 1e-300)
    return np.linalg.solve(G + reg_matrix, F @ t)


def cv_select_reg(features: np.ndarray, target: np.ndarray, n_signals: int,
                   candidates=DEFAULT_REG_CANDIDATES, n_folds: int = 4):
    """Honest k-fold cross-validation restricted to the TRAINING signals
    only. Returns the candidate reg with the lowest average held-out
    error."""
    n_folds = min(n_folds, n_signals)
    fold_size = max(n_signals // n_folds, 1)
    idx = np.arange(n_signals)
    folds = []
    for f in range(n_folds):
        val_idx = idx[f * fold_size:(f + 1) * fold_size]
        fit_idx = np.setdiff1d(idx, val_idx)
        if len(val_idx) == 0 or len(fit_idx) == 0:
            continue
        folds.append((fit_idx, val_idx))

    best_reg, best_cv = candidates[0], np.inf
    for reg in candidates:
        fold_rmses = []
        for fit_idx, val_idx in folds:
            c = ridge_solve_c(features[:, fit_idx, :], target[fit_idx], reg)
            pred = np.tensordot(c, features[:, val_idx, :], axes=(0, 0))
            err = pred - target[val_idx]
            fold_rmses.append(float(np.sqrt(np.mean(err ** 2))))
        cv = np.mean(fold_rmses) if fold_rmses else np.inf
        if cv < best_cv:
            best_cv, best_reg = cv, reg
    return best_reg, best_cv


class CompressedFractionalFilter:
    """The complete method.

    Example
    -------
    >>> import numpy as np
    >>> from fracmem import CompressedFractionalFilter
    >>> f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16)
    >>> train_signals = [np.random.randn(3000) for _ in range(8)]
    >>> f.fit(train_signals, j_max=10000)
    >>> y_hat = f.predict(np.random.randn(5000))
    """

    def __init__(self, alpha: float, h: float, L: int = 32, p: int = 16):
        self.alpha, self.h, self.L, self.p = alpha, h, L, p
        self.lam = None      # decay rates: fixed forever once computed, never re-fit
        self.c = None        # linear weights: the only thing data ever touches
        self.reg_used = None

    def _design_decay_rates(self, j_max: int, w: np.ndarray):
        """Pure mathematics, no data: derive the SOE-quadrature decay
        rates and fix them permanently."""
        lam, c_analytical = soe_tail_kernel(self.alpha, self.L, self.p, w, j_max)
        self.lam = lam
        return c_analytical  # kept only as a reference/fallback value

    def fit(self, train_signals, j_max: int = None, reg_candidates=DEFAULT_REG_CANDIDATES):
        """train_signals: list/array of training signals (all the same
        length). Derives lambda analytically, builds features, honestly
        cross-validates the ridge regularization strength, and solves for
        the final c on all training data."""
        n_signals = len(train_signals)
        n_samples = len(train_signals[0])
        if j_max is None:
            j_max = n_samples
        w = gl_weights(self.alpha, max(j_max, n_samples) + 1)

        gold = np.stack([full_gl_derivative(x, self.alpha, self.h, w) for x in train_signals])
        local = np.stack([local_exact_term(x, self.alpha, self.h, self.L, w) for x in train_signals])
        x_delayed = np.stack([delay(np.asarray(x), self.L) for x in train_signals])
        target = gold - local  # what the tail alone must explain

        self._design_decay_rates(j_max, w)
        feats = mode_features(self.lam, x_delayed)

        best_reg, best_cv = cv_select_reg(feats, target, n_signals, reg_candidates)
        self.c = ridge_solve_c(feats, target, best_reg)
        self.reg_used = best_reg
        return self

    def predict(self, x: np.ndarray, w: np.ndarray = None) -> np.ndarray:
        """Deploy. Identical cost to the pure classical filter: L+p
        multiply-adds per sample, p persistent state values.

        NOTE: c was fit directly against the already h^-alpha-scaled
        target, so it must NOT be rescaled again here -- unlike the raw
        SOE-quadrature weights, which operate on the unscaled kernel."""
        if self.lam is None or self.c is None:
            raise RuntimeError("call .fit(...) before .predict(...)")
        if w is None:
            w = gl_weights(self.alpha, self.L)
        x = np.asarray(x)
        local = local_exact_term(x, self.alpha, self.h, self.L, w)
        x_delayed = delay(x, self.L)
        tail = diagonal_recurrence(x_delayed, self.lam, self.c)
        return local + tail
