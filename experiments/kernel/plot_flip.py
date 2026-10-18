"""Figure for the label-flip control (reads data/experiments/kernel_exploration/flip/results.json)."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import plot_style

OUT = Path(__file__).resolve().parents[2] / "data/experiments/kernel_exploration/flip"


def main():
    plot_style.apply()
    rows = json.loads((OUT / "results.json").read_text())["probes"]
    fig, axes = plt.subplots(1, 2, figsize=(plot_style.DOUBLE_COLUMN, 2.8), sharey=True)
    for ax, space, title in zip(axes, ("raw", "perp_r_a"), ("All of the layer-35 state",
                                                         "Readout + base-answer directions removed")):
        cells = [(t, m) for t in ("clause", "voice") for m in ("linear", "rbf")]
        x = np.arange(len(cells))
        for k, (adapter, label) in enumerate((("normal", "Normal model (answer = cue)"),
                                             ("flipped", "Flipped model (answer = opposite of cue)"))):
            sel = [next(r for r in rows if (r["target"], r["model"], r["space"], r["adapter"]) == (t, m, space, adapter))
                   for t, m in cells]
            z = np.array([r["z"] for r in sel]); lo, hi = np.array([r["z_ci95"] for r in sel]).T
            colors = [plot_style.RULE_COLORS[t] if adapter == "normal" else plot_style.MODEL_DARK_COLORS[t] for t, _ in cells]
            ax.bar(x + (k - .5) * .38, z, .38, color=colors, yerr=[z - lo, hi - z], capsize=3, label=label)
            if adapter == "flipped":
                for xi, r in zip(x, sel):
                    band = r["null_abs_z95"]
                    ax.plot([xi - .4, xi + .4], [band, band], color="k", lw=.8, ls=":")
                    ax.plot([xi - .4, xi + .4], [-band, -band], color="k", lw=.8, ls=":")
                    if r.get("p_holm", 1) < .05:
                        ax.text(xi + .19, r["z"] - .6, "*", ha="center", va="top", fontsize=14)
        ax.axhline(0, color=plot_style.CHANCE_COLOR, lw=.8)
        ax.set_xticks(x, [f"{t}\n{m}" for t, m in cells]); ax.set_title(title)
        ax.text(-.08, 1.08, "ab"[list(axes).index(ax)], transform=ax.transAxes, fontweight="bold")
    axes[0].set_ylabel("Probe score on held-out pairs (z)\n> 0: reads cue-1 as '1'    < 0: reversed")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"flip_control.{ext}", dpi=plot_style.EXPORT_DPI, bbox_inches="tight")
        for directory in ("data/experiments/lab_book", "data/experiments/supervisor_summary"):
            shutil.copyfile(OUT / f"flip_control.{ext}", Path(directory) / f"flip.{ext}")


if __name__ == "__main__":
    main()
