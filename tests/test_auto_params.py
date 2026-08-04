from fracmem import CompressedFractionalFilter


def test_auto_picks_a_valid_config_and_fits_it():
    f = CompressedFractionalFilter.auto(
        alpha=0.5, h=0.01, tol=5e-3,
        n_samples=800, n_signals=3,
        L_candidates=(8, 16), p_candidates=(4, 8),
    )
    assert f.lam is not None and f.c is not None
    assert f.L in (8, 16)
    assert f.p in (4, 8)
    assert hasattr(f, "auto_rmse_")
    assert f.auto_rmse_ >= 0


def test_auto_prefers_cheaper_config_when_it_meets_tolerance():
    """With a loose tolerance the very cheapest (L, p) candidate should
    already qualify, so auto() must not search further."""
    f = CompressedFractionalFilter.auto(
        alpha=0.5, h=0.01, tol=10.0,  # trivially loose
        n_samples=500, n_signals=3,
        L_candidates=(8, 32), p_candidates=(4, 16),
    )
    assert f.L == 8
    assert f.p == 4


def test_auto_respects_custom_train_signals():
    import numpy as np
    rng = np.random.default_rng(0)
    train = [np.cumsum(rng.standard_normal(600)) * 0.02 for _ in range(4)]
    f = CompressedFractionalFilter.auto(
        alpha=0.6, h=0.01, tol=5e-3, train_signals=train,
        n_samples=600, L_candidates=(8, 16), p_candidates=(4, 8),
    )
    assert f.lam is not None
