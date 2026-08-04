import numpy as np

from fracmem import CompressedFractionalFilter


def _make_signal(n, seed):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) * 0.01


def test_save_load_round_trip(tmp_path):
    train = [_make_signal(800, s) for s in range(3)]
    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=8, p=4, definition="caputo")
    f.fit(train, j_max=800)

    path = tmp_path / "filter.npz"
    f.save(str(path))
    g = CompressedFractionalFilter.load(str(path))

    assert g.alpha == f.alpha
    assert g.h == f.h
    assert g.L == f.L
    assert g.p == f.p
    assert g.definition == f.definition
    np.testing.assert_allclose(g.lam, f.lam)
    np.testing.assert_allclose(g.c, f.c)

    sig = _make_signal(200, 5)
    np.testing.assert_allclose(g.predict(sig), f.predict(sig))
