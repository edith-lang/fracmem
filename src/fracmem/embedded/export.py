"""
Desktop-side exporter: bakes a fitted CompressedFractionalFilter's
constants into a standalone copy of fracmem.embedded.runtime, ready to
copy onto an ESP32 (or any other MicroPython device), e.g.:

    mpremote cp fracmem_device.py :main.py

The generated file has no import of `fracmem`, numpy, or scipy -- it is
runtime.py's source (array-module only) plus one instantiated filter
with this fit's lambda/c/w baked in as literals.
"""
from pathlib import Path

from ..kernel import gl_weights

_RUNTIME_PATH = Path(__file__).with_name("runtime.py")


def _fmt_list(values, ndigits: int = 8) -> str:
    return "[" + ", ".join(repr(round(float(v), ndigits)) for v in values) + "]"


def generate_source(filt, w=None, instance_name: str = "filt") -> str:
    """filt: a fitted CompressedFractionalFilter. Returns the full text
    of a standalone .py file."""
    if filt.lam is None or filt.c is None:
        raise RuntimeError("filt must be fitted (call .fit(...)) before exporting")
    if w is None:
        w = gl_weights(filt.alpha, filt.L)

    runtime_src = _RUNTIME_PATH.read_text()
    instantiation = (
        f"\n\n{instance_name} = MicroFractionalFilter(\n"
        f"    alpha={filt.alpha!r},\n"
        f"    h={filt.h!r},\n"
        f"    L={filt.L!r},\n"
        f"    w={_fmt_list(w[:filt.L])},\n"
        f"    lam={_fmt_list(filt.lam)},\n"
        f"    c={_fmt_list(filt.c)},\n"
        f"    caputo={filt.definition == 'caputo'!r},\n"
        f")\n"
        f"# {instance_name}.step(x_k) -> derivative estimate, one sample at a time.\n"
    )
    return runtime_src + instantiation


def export_micropython(filt, path: str, w=None, instance_name: str = "filt"):
    """Write generate_source(...) straight to `path`."""
    Path(path).write_text(generate_source(filt, w=w, instance_name=instance_name))
