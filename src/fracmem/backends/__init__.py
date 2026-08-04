"""
Optional accelerated/differentiable backends. Nothing in here is
imported by fracmem/__init__.py -- each backend has its own heavy,
optional dependency (torch, jax) and is only touched when you
explicitly import fracmem.backends.torch_backend or
fracmem.backends.jax_backend.
"""
