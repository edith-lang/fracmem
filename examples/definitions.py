"""
GL/RL vs Caputo: on a signal with a nonzero first sample, the two
definitions diverge near t=0 (RL has a t^-alpha singularity there;
Caputo removes it by subtracting the initial value first). Away from
t=0 they agree.

Run:
    python examples/definitions.py
"""
import numpy as np

from fracmem import CompressedFractionalFilter


def main():
    rng = np.random.default_rng(2)
    # Nonzero initial value on purpose -- this is where GL/RL and Caputo differ.
    train_signals = [5.0 + np.cumsum(rng.standard_normal(2000)) * 0.01 for _ in range(6)]
    test_signal = 5.0 + np.cumsum(rng.standard_normal(1500)) * 0.01

    f_rl = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16, definition="rl")
    f_caputo = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16, definition="caputo")
    f_rl.fit(train_signals)
    f_caputo.fit(train_signals)

    y_rl = f_rl.predict(test_signal)
    y_caputo = f_caputo.predict(test_signal)

    print("sample |  RL/GL     Caputo    |diff|")
    print("-------+------------------------------")
    for k in (0, 1, 2, 5, 10, 50, 500, 1499):
        print(f"{k:6d} | {y_rl[k]:9.4f}  {y_caputo[k]:9.4f}  {abs(y_rl[k] - y_caputo[k]):9.4f}")

    print("\nnear t=0 the two definitions disagree sharply -- that's RL's")
    print("t^-alpha singularity at a nonzero initial value. The gap then")
    print("shrinks, but only as a slow power law in t (not exponentially),")
    print("since it's driven by that same initial offset RL never subtracts.")


if __name__ == "__main__":
    main()
