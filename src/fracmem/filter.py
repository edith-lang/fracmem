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

DEFINITIONS = ("gl", "rl", "caputo")  # "gl" and "rl" are the same discretization


def _apply_definition(x: np.ndarray, definition: str) -> np.ndarray:
    """GL and RL act on x as-is (causal, x(t)=0 for t<0). Caputo acts on
    x shifted by its own initial sample -- see kernel.caputo_derivative
    for why that substitution is valid."""
    if definition not in DEFINITIONS:
        raise ValueError(f"definition must be one of {DEFINITIONS}, got {definition!r}")
    if definition == "caputo":
        return x - x[0]
    return x


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

    def __init__(self, alpha: float, h: float, L: int = 32, p: int = 16, definition: str = "gl"):
        if definition not in DEFINITIONS:
            raise ValueError(f"definition must be one of {DEFINITIONS}, got {definition!r}")
        self.alpha, self.h, self.L, self.p = alpha, h, L, p
        self.definition = definition
        self.lam = None      # decay rates: fixed forever once computed, never re-fit
        self.c = None        # linear weights: the only thing data ever touches
        self.reg_used = None
        self._stream_w = None
        self._stream_buf = None
        self._stream_m = None
        self._stream_x0 = None
        self._stream_k = None

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
        train_signals = [_apply_definition(np.asarray(x, dtype=np.float64), self.definition)
                          for x in train_signals]

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

    # ---- build a filter with zero training data -----------------------
    @classmethod
    def from_analytic(cls, alpha: float, h: float, L: int = 32, p: int = 16,
                       definition: str = "gl", j_max: int = None,
                       tail_fit_points: int = 400, tail_fit_reg: float = 0.1) -> "CompressedFractionalFilter":
        """Build a fully working, ready-to-deploy filter using ONLY exact
        math -- no .fit(), no example signals, nothing collected from a
        real device. Both lambda (the decay rates) and c (the combination
        weights) come straight from the certified SOE construction in
        soe.py, so the worst-case error is knowable in advance (see
        soe.soe_tail_error). Use this when you want to flash a filter
        straight to a device before you have any real training data, or
        just want the plain classical construction with nothing fit to
        signals at all.

        tail_fit_points: how many points along the tail are checked while
        fitting c to the exact kernel -- higher means a slightly more
        careful fit at a one-time design cost, lower means a faster,
        rougher fit. Default 400, same as the rest of the library.
        tail_fit_reg: how strongly that fit is discouraged from picking
        extreme-sized weights. Default 0.1, same as the rest of the
        library. Unlike .fit(), these two numbers are not overwritten by
        anything afterward -- they are the actual knobs controlling the
        filter you get back.

        j_max: how far into the past (in samples) the design accounts
        for. Defaults to max(2000, 100*L) if not given -- far enough to
        cover realistic signal lengths without you having to pick a
        number yourself.

        Once built, this filter is used exactly like a .fit()-ted one:
        .predict(...), .step(...), .save(...), and the MicroPython
        export all work unchanged.

        Accuracy warning: soe_tail_error's certified bound is about how
        closely c*lambda^m matches the exact kernel weights themselves,
        pointwise -- it is NOT a bound on end-to-end prediction error for
        every possible signal. On a signal with strong long-range drift
        (e.g. a long random walk), even a kernel that matches pointwise
        to a fraction of a percent can accumulate a much larger error
        over thousands of samples, because that class of signal is
        unusually sensitive to the kernel's long-memory tail. This is
        exactly the gap .fit() closes: fitting c against real training
        signals (rather than against the abstract kernel alone) directly
        minimizes end-to-end error on the kind of signal you actually
        care about. Prefer .fit() whenever you have even a handful of
        representative training signals; reach for from_analytic() only
        when you genuinely have none yet, and validate its accuracy on
        a real signal before trusting it long-term.

        Example
        -------
        >>> f = CompressedFractionalFilter.from_analytic(alpha=0.5, h=0.01, L=32, p=16)
        >>> y_hat = f.predict(some_signal)   # no training signals needed anywhere
        """
        if j_max is None:
            j_max = max(2000, 100 * L)
        f = cls(alpha=alpha, h=h, L=L, p=p, definition=definition)
        w = gl_weights(alpha, j_max + 1)
        f.lam, f.c = soe_tail_kernel(alpha, L, p, w, j_max,
                                      tail_fit_points=tail_fit_points, tail_fit_reg=tail_fit_reg)
        return f

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
        x = _apply_definition(np.asarray(x, dtype=np.float64), self.definition)
        local = local_exact_term(x, self.alpha, self.h, self.L, w)
        x_delayed = delay(x, self.L)
        tail = diagonal_recurrence(x_delayed, self.lam, self.c)
        return local + tail

    # ---- streaming (O(L+p) per sample, fixed memory) ----------------
    # Same math as predict(), one sample at a time. This is what any
    # online consumer should call; it is also the reference the
    # embedded (MicroPython) runtime is checked against in tests.
    def reset_stream(self, w: np.ndarray = None):
        if self.lam is None or self.c is None:
            raise RuntimeError("call .fit(...) before streaming")
        self._stream_w = gl_weights(self.alpha, self.L) if w is None else np.asarray(w[:self.L])
        self._stream_buf = np.zeros(self.L)
        self._stream_m = np.zeros(self.p)
        self._stream_x0 = None
        self._stream_k = 0
        return self

    def step(self, x_k: float) -> float:
        """Feed one new sample, get back one derivative estimate."""
        if self.lam is None or self.c is None:
            raise RuntimeError("call .fit(...) before .step(...)")
        if self._stream_w is None:
            self.reset_stream()
        if self.definition == "caputo":
            if self._stream_x0 is None:
                self._stream_x0 = x_k
            x_k = x_k - self._stream_x0

        dropped = self._stream_buf[-1]  # = x_{k-L} (0 while history is still short)
        self._stream_buf = np.roll(self._stream_buf, 1)
        self._stream_buf[0] = x_k
        local = float(self._stream_w @ self._stream_buf) * (self.h ** -self.alpha)

        self._stream_m = self.lam * self._stream_m + dropped
        tail = float(self.c @ self._stream_m)
        self._stream_k += 1
        return local + tail

    # ---- persistence: save the deployed (lam, c) pair ---------------
    def save(self, path: str):
        if self.lam is None or self.c is None:
            raise RuntimeError("call .fit(...) before .save(...)")
        np.savez(path, alpha=self.alpha, h=self.h, L=self.L, p=self.p,
                  definition=self.definition, lam=self.lam, c=self.c)

    @classmethod
    def load(cls, path: str) -> "CompressedFractionalFilter":
        data = np.load(path)
        f = cls(alpha=float(data["alpha"]), h=float(data["h"]), L=int(data["L"]), p=int(data["p"]),
                definition=str(data["definition"]))
        f.lam = data["lam"]
        f.c = data["c"]
        return f

    # ---- automatic parameter selection -------------------------------
    @classmethod
    def auto(cls, alpha: float, h: float, tol: float = 1e-3, train_signals=None,
              definition: str = "gl", n_samples: int = 4000, n_signals: int = 6,
              L_candidates=(8, 16, 32, 64), p_candidates=(4, 8, 16, 32), seed: int = 0):
        """Grid-search (L, p) cheapest-first (L+p is the deployed
        per-sample cost and persistent-state size) and return the fitted
        filter for the cheapest configuration whose held-out RMSE meets
        `tol`, falling back to the lowest-RMSE configuration tried if
        none do. Uses synthetic random-walk signals unless
        `train_signals` is given -- pass your own for a validation
        target that reflects the real deployment signal."""
        rng = np.random.default_rng(seed)
        if train_signals is None:
            train_signals = [np.cumsum(rng.standard_normal(n_samples)) * 0.01 for _ in range(n_signals)]
        val_signal = np.cumsum(rng.standard_normal(n_samples)) * 0.01

        w_val = gl_weights(alpha, n_samples + 1)
        gold = full_gl_derivative(_apply_definition(val_signal, definition), alpha, h, w_val)

        candidates = sorted((L, p) for L in L_candidates for p in p_candidates)
        candidates.sort(key=lambda Lp: Lp[0] + Lp[1])

        best = None  # (rmse, filter)
        for L, p in candidates:
            f = cls(alpha=alpha, h=h, L=L, p=p, definition=definition)
            f.fit(train_signals, j_max=n_samples)
            pred = f.predict(val_signal, w=w_val)
            rmse = float(np.sqrt(np.mean((pred - gold) ** 2)))
            f.auto_rmse_ = rmse
            if best is None or rmse < best.auto_rmse_:
                best = f
            if rmse < tol:
                return f
        return best
