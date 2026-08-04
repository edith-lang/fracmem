import numpy as np

from fracmem import gl_weights, soe_tail_kernel, soe_tail_error, adaptive_soe_tail_kernel


def test_adaptive_soe_meets_tolerance_or_maxes_out():
    alpha, L, j_max = 0.5, 16, 2000
    w = gl_weights(alpha, j_max + 1)
    tol = 1e-3
    lam, c, p_used, err = adaptive_soe_tail_kernel(alpha, L, w, j_max, tol=tol, p_min=4, p_max=64)
    assert err <= tol or p_used == 64
    # independently verify the reported error against the same metric
    np.testing.assert_allclose(err, soe_tail_error(alpha, L, w, lam, c, j_max))


def test_adaptive_soe_uses_no_more_modes_than_a_fixed_p_needs():
    """A fixed p known (from the classical construction) to already meet
    tol should not be beaten by a much larger p from the adaptive
    search -- i.e. adaptive doesn't way overshoot for an easy target."""
    alpha, L, j_max = 0.5, 16, 2000
    w = gl_weights(alpha, j_max + 1)
    lam16, c16 = soe_tail_kernel(alpha, L, 16, w, j_max)
    err16 = soe_tail_error(alpha, L, w, lam16, c16, j_max)

    lam, c, p_used, err = adaptive_soe_tail_kernel(alpha, L, w, j_max, tol=err16 * 2, p_min=4, p_max=64)
    assert p_used <= 32


def test_adaptive_soe_deterministic():
    alpha, L, j_max = 0.7, 32, 3000
    w = gl_weights(alpha, j_max + 1)
    r1 = adaptive_soe_tail_kernel(alpha, L, w, j_max, tol=1e-4)
    r2 = adaptive_soe_tail_kernel(alpha, L, w, j_max, tol=1e-4)
    np.testing.assert_allclose(r1[0], r2[0])
    np.testing.assert_allclose(r1[1], r2[1])
    assert r1[2] == r2[2]
