"""
Turns results/five_tests.json into the two figures used in
SoeVsBruteForce.pdf and BENCHMARKS.md:
  results/accuracy_vs_train_length.png
  results/speed_comparison.png

Run after run_five_tests.py has produced results/five_tests.json:
    python make_plots.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = Path(__file__).parent / "results"

BG = "#ffffff"
INK = "#1a1a2e"
GRID = "#e3e3ea"
BLUE = "#2f6fed"
ORANGE = "#e8792f"
GREEN = "#2fa84f"


def load():
    return json.loads((RESULTS_DIR / "five_tests.json").read_text())


def plot_accuracy(data):
    tests = data["tests"]
    train_lengths = [t["n_train"] for t in tests]
    err_py = [t["rel_rmse_py_vs_gold"] * 100 for t in tests]

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(train_lengths, err_py, marker="o", color=BLUE, linewidth=2.2,
             markersize=7, zorder=3)

    for x, y in zip(train_lengths, err_py):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color=INK)

    ax.set_xscale("log")
    ax.set_xlabel(f"training-signal length (samples)   [test signal fixed at "
                   f"{data['config']['n_test']:,}]", color=INK, fontsize=10)
    ax.set_ylabel("relative RMSE vs. exact GPU answer (%)", color=INK, fontsize=10)
    ax.set_title("Accuracy vs. training-signal length (5 real tests, same test signal)",
                 color=INK, fontsize=12, fontweight="bold", pad=12)

    ax.grid(True, which="both", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.tight_layout()
    out = RESULTS_DIR / "accuracy_vs_train_length.png"
    fig.savefig(out, facecolor=BG)
    print(f"wrote {out}")


def plot_speed(data):
    tests = data["tests"]
    labels = [f"{t['n_train']:,}" for t in tests]
    gold_time = data["gold_time_s"]
    py_times = [t["predict_py_time_s"] for t in tests]
    c_times = [t["predict_c_time_s"] for t in tests]

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x = range(len(labels))
    width = 0.32
    ax.bar([i - width for i in x], [gold_time] * len(labels), width,
           label=f"exact GPU (gold), one-time: {gold_time:.1f}s", color=ORANGE, zorder=3)
    ax.bar(x, py_times, width, label="fracmem, Python predict", color=BLUE, zorder=3)
    ax.bar([i + width for i in x], c_times, width, label="fracmem, compiled C predict",
           color=GREEN, zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("training-signal length (samples)", color=INK, fontsize=10)
    ax.set_ylabel(f"time to process {data['config']['n_test']:,} test samples (s, log scale)",
                 color=INK, fontsize=10)
    ax.set_title("Speed: exact GPU vs. fracmem (Python / compiled C)", color=INK,
                 fontsize=12, fontweight="bold", pad=12)

    ax.grid(True, axis="y", which="both", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors=INK, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)

    fig.tight_layout()
    out = RESULTS_DIR / "speed_comparison.png"
    fig.savefig(out, facecolor=BG, bbox_inches="tight")
    print(f"wrote {out}")


def plot_fit_cost(data):
    tests = data["tests"]
    train_lengths = [t["n_train"] for t in tests]
    fit_times = [t["fit_time_s"] for t in tests]

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(train_lengths, fit_times, marker="s", color=ORANGE, linewidth=2.2,
             markersize=7, zorder=3)
    for x, y in zip(train_lengths, fit_times):
        label = f"{y:.1f}s" if y >= 1 else f"{y*1000:.0f}ms"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9, color=INK)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training-signal length (samples)", color=INK, fontsize=10)
    ax.set_ylabel("fit() time (s, log scale)", color=INK, fontsize=10)
    ax.set_title(".fit() cost vs. training-signal length -- quadratic, not linear",
                 color=INK, fontsize=12, fontweight="bold", pad=12)

    ax.grid(True, which="both", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.tight_layout()
    out = RESULTS_DIR / "fit_cost_vs_train_length.png"
    fig.savefig(out, facecolor=BG)
    print(f"wrote {out}")


def plot_drift():
    path = RESULTS_DIR / "drift_analysis.json"
    if not path.exists():
        print("skipping drift plot (run investigate_drift.py first)")
        return
    data = json.loads(path.read_text())
    bins = data["bins"]
    x = [100 * (b + 0.5) / bins for b in range(bins)]  # % through the 5M-sample run

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    colors = [BLUE, "#6f42c1", ORANGE, "#c1272d"]
    for series, color in zip(data["series"], colors):
        ax.plot(x, series["chunk_rms"], color=color, linewidth=2.0, marker=".", markersize=4,
                 label=f"train={series['n_train']:,}  (max|λ|={series['max_lambda']:.6f})")

    ax.set_yscale("log")
    ax.set_xlabel("position along the 5,000,000-sample test run (%)", color=INK, fontsize=10)
    ax.set_ylabel("|C output - Python output|, RMS per bin (log scale)", color=INK, fontsize=10)
    ax.set_title("Float32 (C) vs. float64 (Python) drift grows with sample index",
                 color=INK, fontsize=12, fontweight="bold", pad=12)

    ax.grid(True, which="both", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors=INK, labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.tight_layout()
    out = RESULTS_DIR / "drift_growth.png"
    fig.savefig(out, facecolor=BG)
    print(f"wrote {out}")


if __name__ == "__main__":
    data = load()
    plot_accuracy(data)
    plot_speed(data)
    plot_fit_cost(data)
    plot_drift()
