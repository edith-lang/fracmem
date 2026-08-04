import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fracmem import CompressedFractionalFilter
from fracmem.backends.torch_backend import TorchFractionalLayer


def _make_signal(n, seed, offset=0.0):
    rng = np.random.default_rng(seed)
    return offset + np.cumsum(rng.standard_normal(n)) * 0.01


@pytest.mark.parametrize("definition", ["gl", "caputo"])
def test_layer_matches_numpy_predict(definition):
    train = [_make_signal(1500, s, offset=1.0) for s in range(4)]
    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=16, p=8, definition=definition)
    f.fit(train, j_max=1500)

    layer = TorchFractionalLayer(f)
    sig = _make_signal(300, 99, offset=1.0)
    batch = f.predict(sig)

    with torch.no_grad():
        out = layer(torch.tensor(sig, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(out, batch, atol=1e-3)


def test_layer_batched_matches_1d_per_row():
    train = [_make_signal(1000, s) for s in range(3)]
    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=8, p=4)
    f.fit(train, j_max=1000)
    layer = TorchFractionalLayer(f)

    sig_a = _make_signal(200, 1)
    sig_b = _make_signal(200, 2)
    with torch.no_grad():
        batched = layer(torch.tensor(np.stack([sig_a, sig_b]), dtype=torch.float32)).numpy()
        row_a = layer(torch.tensor(sig_a, dtype=torch.float32)).numpy()
        row_b = layer(torch.tensor(sig_b, dtype=torch.float32)).numpy()

    np.testing.assert_allclose(batched[0], row_a, atol=1e-5)
    np.testing.assert_allclose(batched[1], row_b, atol=1e-5)


def test_learnable_c_receives_gradients():
    train = [_make_signal(1000, s) for s in range(3)]
    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=8, p=4)
    f.fit(train, j_max=1000)
    layer = TorchFractionalLayer(f, learnable_c=True)

    x = torch.tensor(_make_signal(200, 5), dtype=torch.float32)
    loss = (layer(x) ** 2).mean()
    loss.backward()

    assert layer.c.grad is not None
    assert torch.isfinite(layer.c.grad).all()
    assert layer.lam.grad is None  # lambda stays fixed, non-learnable


def test_frozen_c_by_default():
    train = [_make_signal(1000, s) for s in range(3)]
    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=8, p=4)
    f.fit(train, j_max=1000)
    layer = TorchFractionalLayer(f)
    assert list(layer.parameters()) == []
