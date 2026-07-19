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

See https://github.com/<your-username>/fracmem for the full theory and
code-walkthrough documentation.
"""
from .kernel import gl_weights, full_gl_derivative, local_exact_term, delay
from .soe import soe_nodes_and_weights, soe_refit_weights, soe_tail_kernel
from .filter import (
    CompressedFractionalFilter,
    diagonal_recurrence,
    mode_features,
    ridge_solve_c,
    cv_select_reg,
)

__version__ = "0.1.0"

__all__ = [
    "CompressedFractionalFilter",
    "gl_weights",
    "full_gl_derivative",
    "local_exact_term",
    "delay",
    "soe_nodes_and_weights",
    "soe_refit_weights",
    "soe_tail_kernel",
    "diagonal_recurrence",
    "mode_features",
    "ridge_solve_c",
    "cv_select_reg",
]
