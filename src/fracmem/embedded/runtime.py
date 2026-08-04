"""
Zero-dependency streaming fractional-derivative filter for
microcontrollers (ESP32 via MicroPython, and anywhere else without
numpy/scipy). Uses only the stdlib `array` module -- runs unmodified
under both MicroPython and CPython, which is also how it gets tested
(see tests/test_embedded.py, run under plain CPython).

This file only APPLIES an already-fitted filter (lambda, c, and the
first L GL weights); fitting stays on the desktop, where numpy/scipy
are available. See fracmem.embedded.export for baking a fitted
CompressedFractionalFilter's constants into a standalone copy of this
file, ready to copy onto a device (e.g. via mpremote/ampy).

O(L+p) multiply-adds per sample; L + p + p float32 persistent state,
allocated once in __init__ and never grown -- the whole point being
that RAM use is fixed regardless of how long the device runs.
"""
import array


class MicroFractionalFilter:
    def __init__(self, alpha, h, L, w, lam, c, caputo=False):
        """alpha, h: filter parameters. L: local window length. w: the
        first L GL weights. lam, c: the fitted SOE decay rates and
        readout weights (equal length, p). caputo: subtract the first
        sample seen before filtering -- see kernel.caputo_derivative for
        why that substitution gives the Caputo derivative."""
        self.L = L
        self.p = len(lam)
        self.h_pow = h ** (-alpha)
        self.caputo = caputo
        self.w = array.array('f', w[:L])
        self.lam = array.array('f', lam)
        self.c = array.array('f', c)
        self.buf = array.array('f', [0.0] * L)
        self.m = array.array('f', [0.0] * self.p)
        self.idx = 0
        self.x0 = 0.0
        self.x0_set = False

    def reset(self):
        for i in range(self.L):
            self.buf[i] = 0.0
        for i in range(self.p):
            self.m[i] = 0.0
        self.idx = 0
        self.x0_set = False

    def step(self, x_k):
        """Feed one new sample, get back one derivative estimate."""
        if self.caputo:
            if not self.x0_set:
                self.x0 = x_k
                self.x0_set = True
            x_k = x_k - self.x0

        L = self.L
        buf = self.buf
        idx = self.idx
        dropped = buf[idx]  # = x_{k-L} (0.0 while fewer than L samples seen)
        buf[idx] = x_k
        idx += 1
        if idx >= L:
            idx = 0
        self.idx = idx

        w = self.w
        local = 0.0
        pos = idx
        for j in range(L):
            pos -= 1
            if pos < 0:
                pos = L - 1
            local += w[j] * buf[pos]
        local *= self.h_pow

        lam = self.lam
        c = self.c
        m = self.m
        tail = 0.0
        for i in range(self.p):
            mi = lam[i] * m[i] + dropped
            m[i] = mi
            tail += c[i] * mi

        return local + tail
