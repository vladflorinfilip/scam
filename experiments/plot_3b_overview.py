"""Overview figure: per-layer own-rule ablation (3B full scans) and L30 rule-direction cosines."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "data/experiments/residuals"
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained", gridspec_kw={"width_ratios": [2.2, 1]})
for r, c in zip(["s1", "voice", "clause", "lexical_e6"], ["C0", "C1", "C2", "C3"]):
    l = json.load(open(f"{D}/scans/3b_full/{r}_residual_scan_3b_full/layer_scan.json"))
    xs = sorted(map(int, l))
    axes[0].plot(xs, [l[str(x)]["follow"] for x in xs], "-o", ms=3, color=c, label=r)
axes[0].axhline(.5, color="grey", ls="--", lw=1, label="≈ base rate")
axes[0].set(xlabel="Layer (Qwen2.5-3B, 36 blocks)", ylabel="Rule following after removing direction", ylim=(0, 1.05),
            title="Removing each rule's direction, layer by layer (selection pairs)")
axes[0].legend(loc="lower left")
g = json.load(open(f"{D}/transfer/four_rule_3b_l30_lexical_e6/direction_geometry.json"))
m, n = np.array(g["cosines"]), g["names"]
im = axes[1].imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
axes[1].set(xticks=range(len(n)), xticklabels=n, yticks=range(len(n)), yticklabels=n,
            title="Cosine similarity of rule directions (layer 30)")
for i in range(len(n)):
    for j in range(len(n)):
        axes[1].text(j, i, f"{m[i, j]:.2f}", ha="center", va="center")
fig.colorbar(im, ax=axes[1], shrink=.8)
fig.savefig("data/experiments/figures_3b/overview_layers_and_cosines.png", dpi=150)
