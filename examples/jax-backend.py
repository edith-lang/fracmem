"""
Functional JAX re-implementation of predict(): jit-, grad-, and
vmap-compatible, using lax.scan for the mode recurrence.

Requires:
    pip install fracmem[jax]

Run:
    python examples/jax-backend.py
"""
import numpy as np

from fracmem import CompressedFractionalFilter
from fracmem.kernel import gl_weights

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None


def main():
    if jax is None:
        print("JAX isn't installed. Install it with:\n\n    pip install fracmem[jax]\n")
        return

    from fracmem.backends.jax_backend import predict, jit_predict

    rng = np.random.default_rng(5)
    train_signals = [np.cumsum(rng.standard_normal(2000)) * 0.01 for _ in range(6)]

    filt = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16)
    filt.fit(train_signals)
    w = gl_weights(filt.alpha, filt.L)

    x = np.cumsum(rng.standard_normal(1000)) * 0.01

    y_numpy = filt.predict(x)
    y_jax = jit_predict(x, filt.lam, filt.c, w, filt.alpha, filt.h, filt.L)
    print(f"numpy vs jax max abs diff: {float(np.max(np.abs(y_numpy - np.asarray(y_jax)))):.2e}"
          "  (float32 vs float64 rounding)")

    # grad-compatible: differentiate the output w.r.t. the readout weights c.
    def loss_fn(c):
        return predict(x, filt.lam, c, w, filt.alpha, filt.h, filt.L).sum()

    grad_c = jax.grad(loss_fn)(jnp.asarray(filt.c, dtype=jnp.float32))
    print(f"grad w.r.t. c: shape={grad_c.shape}, norm={float(jnp.linalg.norm(grad_c)):.4f}")

    # vmap-compatible: batch over multiple signals in one call.
    batch = np.stack([np.cumsum(rng.standard_normal(1000)) * 0.01 for _ in range(4)])
    batched_predict = jax.vmap(lambda xi: predict(xi, filt.lam, filt.c, w, filt.alpha, filt.h, filt.L))
    y_batch = batched_predict(batch)
    print(f"vmap over {batch.shape[0]} signals -> output shape {y_batch.shape}")


if __name__ == "__main__":
    main()
