"""
Low-level look at what CompressedFractionalFilter.fit does under the
hood: derive the tail's decay rates and mode count directly, growing p
until the worst-case relative error against the exact GL tail meets a
tolerance -- no signal data involved, this step is pure mathematics.

Run:
    python examples/adaptive-soe.py
"""
from fracmem import gl_weights, adaptive_soe_tail_kernel

ALPHA = 0.5
L = 32
J_MAX = 10_000


def main():
    w = gl_weights(ALPHA, J_MAX + 1)

    print("tol      p_used   achieved worst-case relative error")
    print("-------  ------   -----------------------------------")
    for tol in (1e-2, 1e-3, 1e-4, 1e-5):
        lam, c, p_used, err = adaptive_soe_tail_kernel(ALPHA, L, w, J_MAX, tol=tol)
        print(f"{tol:.0e}   {p_used:6d}   {err:.3e}")

    print("\ntighter tolerances need more modes (p) -- the adaptive search")
    print("finds the smallest p meeting each tolerance instead of you")
    print("having to guess it.")


if __name__ == "__main__":
    main()
