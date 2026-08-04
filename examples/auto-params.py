"""
Let fracmem pick (L, p) for you instead of hand-tuning them: grid-search
cheapest-first, stop at the first configuration meeting a target RMSE.

Run:
    python examples/auto-params.py
"""
from fracmem import CompressedFractionalFilter


def main():
    for tol in (1e-2, 1e-3, 1e-4):
        filt = CompressedFractionalFilter.auto(alpha=0.5, h=0.01, tol=tol)
        cost = filt.L + filt.p
        print(f"tol={tol:.0e}  ->  L={filt.L:3d}  p={filt.p:3d}  "
              f"(deployed cost {cost} multiply-adds/sample)  rmse={filt.auto_rmse_:.2e}")


if __name__ == "__main__":
    main()
