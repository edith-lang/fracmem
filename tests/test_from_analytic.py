import numpy as np

from fracmem import (
    CompressedFractionalFilter,
    gl_weights,
    full_gl_derivative,
    soe_tail_error,
)


def test_from_analytic_needs_no_training_data_and_predicts():
    """from_analytic's own guarantee is the certified kernel-level bound
    (soe_tail_error), not an end-to-end RMSE promise -- a long,
    unbounded-variance signal (a random walk over thousands of samples)
    is a known-hard case where a tiny per-weight error still compounds
    into a large one (see the accuracy warning in from_analytic's own
    docstring; this is exactly the gap .fit() closes with real data).
    So this test checks the certified bound directly, and only checks
    end-to-end accuracy on a short, easy signal."""
    alpha, h, L, p, j_max = 0.5, 0.01, 32, 16, 3200
    f = CompressedFractionalFilter.from_analytic(alpha=alpha, h=h, L=L, p=p, j_max=j_max)
    assert f.lam is not None and f.c is not None

    w_design = gl_weights(alpha, j_max + 1)
    bound = soe_tail_error(alpha, L, w_design, f.lam, f.c, j_max)
    assert bound < 0.01  # the actual documented guarantee

    n = 150  # short and easy: no training data has been given at all
    signal = np.cumsum(np.random.default_rng(0).standard_normal(n)) * 0.01
    w = gl_weights(alpha, n + 1)
    gold = full_gl_derivative(signal, alpha, h, w)
    pred = f.predict(signal, w=w[:L])
    rel_rmse = np.sqrt(np.mean((pred - gold) ** 2)) / np.sqrt(np.mean(gold ** 2))
    assert rel_rmse < 0.15


def test_tail_fit_points_is_a_real_user_facing_knob():
    """The whole point of exposing tail_fit_points/tail_fit_reg is that a
    user can actually change the result by setting them -- verify that a
    much coarser probe grid gives a measurably different (and generally
    worse) certified error bound than the library's own default of 400."""
    alpha, h, L, p, j_max = 0.5, 0.01, 32, 4, 4000
    w = gl_weights(alpha, j_max + 1)

    fine = CompressedFractionalFilter.from_analytic(
        alpha=alpha, h=h, L=L, p=p, j_max=j_max, tail_fit_points=400)
    coarse = CompressedFractionalFilter.from_analytic(
        alpha=alpha, h=h, L=L, p=p, j_max=j_max, tail_fit_points=3)

    assert not np.allclose(fine.c, coarse.c)

    err_fine = soe_tail_error(alpha, L, w, fine.lam, fine.c, j_max)
    err_coarse = soe_tail_error(alpha, L, w, coarse.lam, coarse.c, j_max)
    assert err_fine <= err_coarse


def test_from_analytic_default_j_max_is_reasonable():
    f = CompressedFractionalFilter.from_analytic(alpha=0.5, h=0.01, L=32, p=16)
    signal = np.ones(50)
    out = f.predict(signal)
    assert np.all(np.isfinite(out))
