"""
fracmem -- compressed fractional-order derivative filters with a
certified offline error bound and data-refined weights.

Quick start
-----------
    from fracmem import CompressedFractionalFilter
    import numpy as np

    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16)
    f.fit(train_signals, j_max=10000)   # train_signals: a handful of representative example signals
    y_hat = f.predict(new_signal)       # O(L+p) cost per sample, forever

See https://github.com/edith-lang/fracmem for the full theory and
code-walkthrough documentation.
"""
from .kernel import (
    gl_weights,
    full_gl_derivative,
    local_exact_term,
    delay,
    rl_derivative,
    caputo_derivative,
)
from .soe import (
    soe_nodes_and_weights,
    soe_refit_weights,
    soe_tail_kernel,
    soe_tail_error,
    adaptive_soe_tail_kernel,
)
from .filter import (
    CompressedFractionalFilter,
    diagonal_recurrence,
    mode_features,
    ridge_solve_c,
    cv_select_reg,
    DEFINITIONS,
)

__version__ = "0.2.0"

__all__ = [
    "CompressedFractionalFilter",
    "DEFINITIONS",
    "gl_weights",
    "full_gl_derivative",
    "local_exact_term",
    "delay",
    "rl_derivative",
    "caputo_derivative",
    "soe_nodes_and_weights",
    "soe_refit_weights",
    "soe_tail_kernel",
    "soe_tail_error",
    "adaptive_soe_tail_kernel",
    "diagonal_recurrence",
    "mode_features",
    "ridge_solve_c",
    "cv_select_reg",
]
