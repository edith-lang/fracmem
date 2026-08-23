"""
An honest, non-FFT-shortcut O(n^2) exact Grunwald-Letnikov fractional
derivative, evaluated on the GPU via blocked Toeplitz matmul.

Why this file exists: fracmem.kernel.full_gl_derivative (the library's
own CPU reference) uses np.convolve, which is fine for the training
signals fit() already deals with, but is far too slow in plain numpy to
ever finish on a multi-million-sample "deployment length" test signal.
Some GPU convolution libraries silently switch to an FFT-based algorithm
for a input this large, which would make the "slow, exact" method look
artificially fast -- ruining any speed comparison against it. This file
guarantees no such shortcut: it is a direct, hand-written O(n^2)
computation, just organised into GPU-sized blocks so it actually
finishes in reasonable time.

The math (identical to fracmem.kernel.full_gl_derivative, just computed
differently):
    D^alpha x(t_k) ~= h^-alpha * sum_{j=0}^{k} w_j * x_{k-j}
This is a causal (lower-triangular) convolution of x against the GL
weight sequence w. Writing it as a matrix-vector product y = T @ x,
where T is the n x n lower-triangular Toeplitz matrix with
T[k, m] = w[k - m] for m <= k (else 0), and splitting x, y, T into
blocks of size B:
    T[i, g] (the B x B sub-block at block-row i, block-col g) depends
    ONLY on the block-diagonal offset d = i - g, since T is Toeplitz.
So each distinct offset d needs its B x B block built only once (via a
gather from w). The key efficiency point, found by actually timing this
on real hardware (a 6GB RTX 4050): if you loop by BLOCK-ROW i first and
rebuild T_{i-g} for every g < i inside that, you rebuild the *same*
T_d matrix from scratch every single time it is needed (T_d is reused
by every row i that has some g with i-g=d) -- pure wasted work, and it
measured at ~28ms of pointless rebuilding per pair. The fix is to loop
by DIAGONAL OFFSET d instead: for a fixed d, build T_d exactly ONCE,
then multiply it against every matching x-block *at the same time* as
one ordinary (B,B) @ (B,K) matmul (K = how many row/col block pairs
share that offset), instead of K separate tiny matmuls. Same O(n^2)
total multiply-adds, but now every GPU call does real, large, efficient
work instead of relaunching kernels for a few microseconds of math each.
"""
import torch


def toeplitz_block(w: torch.Tensor, d: int, block: int,
                    row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    """Build the single B x B sub-block T_d[a,b] = w[d*block + a - b] (0
    where that index is negative -- non-causal -- or past the end of
    w). row/col are pre-built int32 index grids (B,1)/(1,B), reused
    across every call -- int32, not the torch default int64, since a
    naive int64 index grid at a large block size alone can blow past
    6GB of VRAM."""
    idx = d * block + row - col                                # (B,B) int32
    valid = (idx >= 0) & (idx < w.numel())
    idx.clamp_(0, w.numel() - 1)
    vals = w[idx]
    vals.masked_fill_(~valid, 0.0)
    return vals


@torch.no_grad()
def gpu_exact_gl_derivative(x: torch.Tensor, w: torch.Tensor, h: float, alpha: float,
                             block: int = 8192) -> torch.Tensor:
    """The exact fractional derivative of x, computed the honest O(n^2)
    way on the GPU. x and w must already be on the target device.
    w must have length >= len(x) (weights w_0 .. w_{n-1}). `block` is
    the Toeplitz sub-block side length -- only ONE such block is ever
    live in memory at a time (peak transient memory is roughly
    block^2 * 9 bytes), so this can be set fairly large even on a
    small card."""
    n = x.numel()
    device = x.device
    dtype = x.dtype
    n_blocks = (n + block - 1) // block

    pad = n_blocks * block - n
    if pad:
        x = torch.cat([x, torch.zeros(pad, device=device, dtype=dtype)])

    # X, Y laid out as (block, n_blocks): column g is x-block g / y-block g.
    X = x.view(n_blocks, block).t().contiguous()   # (B, n_blocks)
    Y = torch.zeros_like(X)                          # (B, n_blocks)

    row = torch.arange(block, device=device, dtype=torch.int32).unsqueeze(1)  # (B,1)
    col = torch.arange(block, device=device, dtype=torch.int32).unsqueeze(0)  # (1,B)

    for d in range(n_blocks):
        T_d = toeplitz_block(w, d, block, row, col)   # (B,B), built once
        k = n_blocks - d                               # number of (i,g) pairs sharing this offset
        Y[:, d:d + k] += T_d @ X[:, :k]                # one big matmul covers all of them

    y = Y.t().reshape(-1)[:n]
    return y * (h ** (-alpha))


def verify_against_numpy(alpha=0.5, h=1.0, n=2000, block=256, device="cuda", seed=0):
    """Sanity check: does this GPU implementation agree with fracmem's
    own CPU reference (fracmem.kernel.full_gl_derivative) on a small
    signal? Returns the max relative error observed."""
    import numpy as np
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from fracmem.kernel import gl_weights, full_gl_derivative

    rng = np.random.default_rng(seed)
    x_np = np.cumsum(rng.standard_normal(n)) * 0.01
    w_np = gl_weights(alpha, n)
    gold_cpu = full_gl_derivative(x_np, alpha, h, w_np)

    dev = torch.device(device)
    x_t = torch.tensor(x_np, device=dev, dtype=torch.float32)
    w_t = torch.tensor(w_np, device=dev, dtype=torch.float32)
    y_gpu = gpu_exact_gl_derivative(x_t, w_t, h, alpha, block=block).cpu().numpy()

    rel = np.abs(y_gpu - gold_cpu) / (np.abs(gold_cpu) + 1e-12)
    rmse = float(np.sqrt(np.mean((y_gpu - gold_cpu) ** 2)))
    denom = float(np.sqrt(np.mean(gold_cpu ** 2)))
    rel_rmse = rmse / denom if denom > 0 else float("nan")
    return {
        "max_abs_diff": float(np.max(np.abs(y_gpu - gold_cpu))),
        "max_rel_diff": float(np.max(rel)),
        "rel_rmse": rel_rmse,
    }


if __name__ == "__main__":
    result = verify_against_numpy()
    print("Verification vs fracmem.kernel.full_gl_derivative (n=2000, float32 GPU):")
    for k, v in result.items():
        print(f"  {k}: {v:.3e}")
