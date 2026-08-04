"""
JAX backend: a functional (not stateful) re-implementation of
CompressedFractionalFilter.predict using jax.numpy + jax.lax.scan for
the mode recurrence -- jit-, grad-, and vmap-compatible, and runs on
whatever device jaxlib is built for (CPU/GPU/TPU). Requires
`pip install fracmem[jax]`.
"""
try:
    import jax
    import jax.numpy as jnp
    _HAVE_JAX = True
except ImportError:
    _HAVE_JAX = False

__all__ = ["predict", "jit_predict"]


def _no_jax(*a, **k):
    raise ImportError("fracmem.backends.jax_backend requires JAX. Install with: pip install fracmem[jax]")


if _HAVE_JAX:
    def predict(x, lam, c, w, alpha: float, h: float, L: int, definition: str = "gl"):
        """Same math as CompressedFractionalFilter.predict: the local
        window via jnp.convolve (matches kernel.local_exact_term
        exactly), the tail via lax.scan -- the natural JAX primitive for
        a linear recurrence, and the one part that can't be vectorized
        away (each mode depends on its own previous state)."""
        x = jnp.asarray(x, dtype=jnp.float32)
        if definition == "caputo":
            x = x - x[0]
        T = x.shape[0]
        h_pow = h ** (-alpha)
        w_l = jnp.asarray(w[:L], dtype=jnp.float32)
        lam = jnp.asarray(lam, dtype=jnp.float32)
        c = jnp.asarray(c, dtype=jnp.float32)

        local = jnp.convolve(x, w_l)[:T] * h_pow
        x_delayed = jnp.pad(x, (L, 0))[:T]

        def scan_fn(m, xd):
            m = lam * m + xd
            return m, jnp.dot(c, m)

        _, tail = jax.lax.scan(scan_fn, jnp.zeros_like(lam), x_delayed)
        return local + tail

    jit_predict = jax.jit(predict, static_argnames=("L", "definition"))
else:
    predict = _no_jax
    jit_predict = _no_jax
