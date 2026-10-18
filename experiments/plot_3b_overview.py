"""Overview figure: per-layer own-rule ablation (3B full scans) and L30 rule-direction cosines."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import plot_style

D = "data/experiments/residuals"
plot_style.apply()
fig, axes = plt.subplots(1, 2, figsize=(plot_style.DOUBLE_COLUMN, 2.9), gridspec_kw={"width_ratios": [2.2, 1]})
for r in ["s1", "voice", "clause", "lexical_e6"]:
    c = plot_style.RULE_COLORS["lexical" if r == "lexical_e6" else r]
    l = json.load(open(f"{D}/scans/3b_full/{r}_residual_scan_3b_full/layer_scan.json"))
    xs = sorted(map(int, l))
    axes[0].plot(xs, [l[str(x)]["follow"] for x in xs], "-o", ms=2.5, color=c, label=r)
axes[0].axhline(.5, color=plot_style.CHANCE_COLOR, ls=":", lw=.8, label="chance")
axes[0].set(xlabel="Layer (Qwen2.5-3B, 36 blocks)", ylabel="Rule following after removing direction", ylim=(0, 1.05),
            title="Own-direction removal")
axes[0].text(-0.08, 1.08, "a", transform=axes[0].transAxes, fontweight="bold")
axes[0].legend(loc="upper left", ncol=2)
g = json.load(open(f"{D}/transfer/four_rule_3b_l30_lexical_e6/direction_geometry.json"))
m, n = np.array(g["cosines"]), g["names"]
im = axes[1].imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
axes[1].set(xticks=range(len(n)), xticklabels=n, yticks=range(len(n)), yticklabels=n,
            title="Direction cosine (layer 30)")
for i in range(len(n)):
    for j in range(len(n)):
        axes[1].text(j, i, f"{m[i, j]:.2f}", ha="center", va="center")
fig.colorbar(im, ax=axes[1], shrink=.8)
axes[1].text(-0.08, 1.08, "b", transform=axes[1].transAxes, fontweight="bold")
fig.tight_layout()
for directory in ("data/experiments/lab_book", "data/experiments/supervisor_summary", "data/experiments/figures_3b"):
    for ext in ("pdf", "png"):
        fig.savefig(f"{directory}/layers.{ext}" if "figures_3b" not in directory else
                    f"{directory}/overview_layers_and_cosines.{ext}",
                    dpi=plot_style.EXPORT_DPI, bbox_inches="tight")
