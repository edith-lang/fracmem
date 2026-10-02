"""
Two things this example shows:

1. CompressedFractionalFilter.from_analytic(...) -- build a working
   filter with NO training signals at all, straight from exact math.
   tail_fit_points/tail_fit_reg are set by hand here (not the library's
   own default of 400) to show they are real, user-facing knobs.

2. fracmem.embedded.export_c(...) -- bake that filter's constants into
   a small, ready-to-compile .c file for the plain-C runtime
   (fracmem/embedded/c/fracmemfilter.h + fracmemfilter.c), meant for
   any real embedded C/C++ toolchain (ESP-IDF, Arduino, bare-metal).

Run:
    python examples/embedded-export-c.py
(needs a C compiler -- gcc or cc -- on PATH, only to prove the
generated file actually compiles and runs; nothing about export_c
itself needs one)
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from fracmem import CompressedFractionalFilter
from fracmem.embedded import export_c

C_DIR = Path(__file__).parent.parent / "src" / "fracmem" / "embedded" / "c"


def main():
    # No training signals anywhere -- just the math.
    filt = CompressedFractionalFilter.from_analytic(
        alpha=0.5, h=0.01, L=16, p=8,
        tail_fit_points=800,   # more than the library default (400): a slower, more careful design
        tail_fit_reg=0.05,     # less regularization than the default (0.1): trust the fit a bit more
    )
    print(f"built with no training data: L={filt.L} p={filt.p} "
          f"tail_fit_points=800 tail_fit_reg=0.05")

    with tempfile.TemporaryDirectory() as tmp:
        device_c = Path(tmp) / "device_filter.c"
        export_c(filt, str(device_c), instance_name="deployed")
        print(f"exported {device_c.stat().st_size} bytes to {device_c.name}")

        cc = shutil.which("cc") or shutil.which("gcc")
        if cc is None:
            print("\nno C compiler found on PATH -- skipping the compile/run check "
                  "(the exported file is still ready to copy into an embedded project)")
            return

        main_c = Path(tmp) / "main.c"
        main_c.write_text(
            "#include <stdio.h>\n"
            '#include "fracmemfilter.h"\n'
            "extern FracmemFilter deployed;\n"
            "void deployedSetup(void);\n"
            "int main(void) {\n"
            "    deployedSetup();\n"
            "    float xk, yk;\n"
            "    while (scanf(\"%f\", &xk) == 1) {\n"
            "        yk = fracmemStep(&deployed, xk);\n"
            "        printf(\"%.8f\\n\", yk);\n"
            "    }\n"
            "    return 0;\n"
            "}\n"
        )
        exe = Path(tmp) / "device_main"
        subprocess.run(
            [cc, "-O2", "-o", str(exe), str(main_c), str(device_c), str(C_DIR / "fracmemfilter.c"),
             "-I", str(C_DIR)],
            check=True,
        )

        rng = np.random.default_rng(7)
        test_signal = np.cumsum(rng.standard_normal(200)) * 0.01
        result = subprocess.run(
            [str(exe)], input="\n".join(repr(float(v)) for v in test_signal),
            capture_output=True, text=True, check=True,
        )
        y_device = np.array([float(line) for line in result.stdout.split()])
        y_desktop = filt.predict(test_signal)

        max_diff = float(np.max(np.abs(y_desktop - y_device)))
        print(f"\ncompiled and ran the exported C file successfully")
        print(f"max |desktop - compiled C| over {len(test_signal)} samples: {max_diff:.2e}"
              "  (float32 rounding only)")


if __name__ == "__main__":
    main()
