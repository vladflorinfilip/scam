"""Figure for the label-flip control (reads data/experiments/kernel_exploration/flip/results.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "data/experiments/kernel_exploration/flip"


def main():
    rows = json.loads((OUT / "results.json").read_text())["probes"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for ax, space, title in zip(axes, ("raw", "perp_r_a"), ("All of the layer-35 state",
                                                         "Readout + base-answer directions removed")):
        cells = [(t, m) for t in ("clause", "voice") for m in ("linear", "rbf")]
        x = np.arange(len(cells))
        for k, (adapter, colour, label) in enumerate((("normal", "tab:blue", "Normal model (answer = cue)"),
                                                     ("flipped", "tab:red", "Flipped model (answer = opposite of cue)"))):
            sel = [next(r for r in rows if (r["target"], r["model"], r["space"], r["adapter"]) == (t, m, space, adapter))
                   for t, m in cells]
            z = np.array([r["z"] for r in sel]); lo, hi = np.array([r["z_ci95"] for r in sel]).T
            ax.bar(x + (k - .5) * .38, z, .38, color=colour, yerr=[z - lo, hi - z], capsize=3, label=label)
            if adapter == "flipped":
                for xi, r in zip(x, sel):
                    band = r["null_abs_z95"]
                    ax.plot([xi - .4, xi + .4], [band, band], color="k", lw=.8, ls=":")
                    ax.plot([xi - .4, xi + .4], [-band, -band], color="k", lw=.8, ls=":")
                    if r.get("p_holm", 1) < .05:
                        ax.text(xi + .19, r["z"] - .6, "*", ha="center", va="top", fontsize=14)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(x, [f"{t}\n{m}" for t, m in cells]); ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Probe score on held-out pairs (z)\n> 0: reads cue-1 as '1'    < 0: reversed")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Probe trained on the other three normal rules, tested on each model (3B, layer 35). "
                 "Dotted: 95% shuffled-label range; * Holm p < 0.05", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"flip_control.{ext}", dpi=160)


if __name__ == "__main__":
    main()
