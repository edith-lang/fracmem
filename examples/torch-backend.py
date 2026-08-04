"""
Wrap a fitted filter as a differentiable PyTorch layer -- runs on
CPU/CUDA/MPS with the same .to(device) call as any other nn.Module.

Requires:
    pip install fracmem[torch]

Run:
    python examples/torch-backend.py
"""
import numpy as np

from fracmem import CompressedFractionalFilter

try:
    import torch
except ImportError:
    torch = None


def main():
    if torch is None:
        print("PyTorch isn't installed. Install it with:\n\n    pip install fracmem[torch]\n")
        return

    from fracmem.backends.torch_backend import TorchFractionalLayer

    rng = np.random.default_rng(4)
    train_signals = [np.cumsum(rng.standard_normal(2000)) * 0.01 for _ in range(6)]

    filt = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16)
    filt.fit(train_signals)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = TorchFractionalLayer(filt, learnable_c=True).to(device)
    print(f"running on: {device}")

    batch = torch.tensor(
        np.stack([np.cumsum(rng.standard_normal(500)) * 0.01 for _ in range(4)]),
        dtype=torch.float32, device=device,
    )
    y_hat = layer(batch)
    print(f"input shape:  {tuple(batch.shape)}")
    print(f"output shape: {tuple(y_hat.shape)}")

    # It's a real nn.Module: gradients flow through the learnable readout.
    loss = y_hat.pow(2).mean()
    loss.backward()
    grad_norm = layer.c.grad.norm().item()
    print(f"loss={loss.item():.6f}  grad norm on readout weights c: {grad_norm:.6f}")
    print("\n(lambda, the SOE decay rates, stays a fixed non-learnable buffer --")
    print(" only c, the linear readout, is a trainable nn.Parameter here.)")


if __name__ == "__main__":
    main()
