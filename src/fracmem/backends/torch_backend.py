"""
PyTorch backend: a differentiable nn.Module version of
CompressedFractionalFilter.predict. This is also fracmem's GPU support
-- there's no separate GPU code path, moving the module with
`.to("cuda")` (or "mps") runs every op, including the mode recurrence,
on-device, same as any other PyTorch model. Requires `pip install
fracmem[torch]`.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

from ..kernel import gl_weights

__all__ = ["TorchFractionalLayer"]


class _NoTorch:
    def __init__(self, *a, **k):
        raise ImportError(
            "fracmem.backends.torch_backend requires PyTorch. Install with: pip install fracmem[torch]"
        )


if _HAVE_TORCH:
    class TorchFractionalLayer(nn.Module):
        """Wraps an already-fitted CompressedFractionalFilter as a
        differentiable layer.

            filt = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16)
            filt.fit(train_signals)
            layer = TorchFractionalLayer(filt).to("cuda")
            y_hat = layer(x)   # x: (T,) or (batch, T)

        `lambda` (the SOE decay rates) is a fixed, non-learnable buffer,
        same design principle as the numpy filter -- only `c`, the
        linear readout, can be made a learnable nn.Parameter via
        learnable_c=True, e.g. to fine-tune inside a larger network by
        backprop instead of the ridge-regression fit.
        """

        def __init__(self, filt, w: np.ndarray = None, learnable_c: bool = False,
                     dtype=torch.float32):
            super().__init__()
            if filt.lam is None or filt.c is None:
                raise RuntimeError("filt must be fitted (call .fit(...)) before wrapping")
            if w is None:
                w = gl_weights(filt.alpha, filt.L)

            self.alpha, self.h, self.L = filt.alpha, filt.h, filt.L
            self.definition = filt.definition
            self.h_pow = float(filt.h ** -filt.alpha)

            self.register_buffer("lam", torch.as_tensor(filt.lam, dtype=dtype))
            c_t = torch.as_tensor(filt.c, dtype=dtype)
            if learnable_c:
                self.c = nn.Parameter(c_t)
            else:
                self.register_buffer("c", c_t)

            w_flip = torch.as_tensor(np.asarray(w[:filt.L], dtype=np.float64), dtype=dtype)
            w_flip = w_flip.flip(0).view(1, 1, filt.L)
            self.register_buffer("w_flip", w_flip)

        def forward(self, x):
            squeeze = x.dim() == 1
            if squeeze:
                x = x.unsqueeze(0)
            if self.definition == "caputo":
                x = x - x[:, :1]
            B, T = x.shape

            # Local (exact) window: causal cross-correlation against the
            # first L GL weights, left-padded so the L-1 warmup samples
            # before t=0 are treated as zero.
            xp = F.pad(x.unsqueeze(1), (self.L - 1, 0))
            local = F.conv1d(xp, self.w_flip).squeeze(1) * self.h_pow

            # Tail input: x shifted right by L, zero-padded -- identical
            # to kernel.delay().
            x_delayed = F.pad(x, (self.L, 0))[:, :T]

            p = self.lam.shape[0]
            m = torch.zeros(B, p, dtype=x.dtype, device=x.device)
            outs = []
            for k in range(T):
                # Inherently sequential (each mode depends on its own
                # previous state) -- this loop is the one part that
                # can't be vectorized away.
                m = self.lam * m + x_delayed[:, k:k + 1]
                outs.append((m * self.c).sum(-1))
            tail = torch.stack(outs, dim=1)

            out = local + tail
            return out.squeeze(0) if squeeze else out
else:
    TorchFractionalLayer = _NoTorch
