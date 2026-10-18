"""Lab-book figure: Qwen 0.5B vs 3B training success and own-rule layer scans (by relative depth)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "data/experiments/residuals"
C = {"s1": "C0", "voice": "C1", "clause": "C2", "lexical": "C3"}

# Held-out fixed-CoT pair follow rates (data/experiments/report_3b_vs_05b.md, section 1).
train = {  # rule: (0.5B adapter, 0.5B base, 3B adapter, 3B base)
    "s1": (99.5, 60.0, 99.5, 60.5), "voice": (99.0, 52.0, 99.5, 49.0),
    "clause": (100.0, 65.4, 100.0, 51.0), "lexical": (100.0, 47.1, 57.7, 50.0),
}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
x = np.arange(4)
for i, (r, (a5, b5, a3, b3)) in enumerate(train.items()):
    ax[0].bar(i - .2, a5, .38, color=C[r], alpha=.45, label="Qwen 0.5B" if i == 0 else None)
    ax[0].bar(i + .2, a3, .38, color=C[r], label="Qwen 3B" if i == 0 else None)
    for dx, b in ((-.2, b5), (.2, b3)):
        ax[0].plot([i + dx - .17, i + dx + .17], [b, b], color="k", lw=2)
ax[0].bar(3.2, 100.0, .38, color="none", edgecolor=C["lexical"], hatch="//", label="3B lexical, 6 epochs")
ax[0].set(xticks=x, xticklabels=list(train), ylim=(0, 105), ylabel="Hidden rule followed (%)",
          title="a  Training: rule followed on test pairs")
ax[0].plot([], [], color="k", lw=2, label="base model")
ax[0].legend(fontsize=8, loc="lower left")

def scan(axis, series, n, title, labels={}):
    for r, (layers, follow, style) in series.items():
        axis.plot(np.array(layers) / n, follow, style, ms=3, color=C[r], label=labels.get(r, r))
    axis.axhline(.5, color="grey", ls="--", lw=1, label="≈ base rate")
    axis.set(xlim=(-.02, 1.0), ylim=(0, 1.05), xlabel=f"Relative depth (layer ÷ {n})",
             ylabel="Rule followed after removing its direction", title=title)
    axis.legend(fontsize=8, loc="lower left")

s5 = json.load(open(f"{D}/transfer/s1_voice_voice05b/layer_scan.json"))["scan"]
cl = json.load(open(f"{D}/own_rule/clause/layer_scan.json"))
lx = json.load(open(f"{D}/scans/late_l20plus/lexical_residual_scan_l20plus/layer_scan.json"))
small = {r: ([e["layer"] for e in s5[r]], [e["rule_accuracy"] for e in s5[r]], "-o") for r in ("s1", "voice")}
small["clause"] = (sorted(map(int, cl)), [cl[str(k)]["follow"] for k in sorted(map(int, cl))], "-o")
small["lexical"] = (sorted(map(int, lx)), [lx[str(k)]["follow"] for k in sorted(map(int, lx))], "s")
scan(ax[1], small, 24, "b  Qwen 0.5B (24 layers)", {"lexical": "lexical (layers 20–22 only)"})

big = {}
for r, name in (("s1", "s1"), ("voice", "voice"), ("clause", "clause"), ("lexical", "lexical_e6")):
    l = json.load(open(f"{D}/scans/3b_full/{name}_residual_scan_3b_full/layer_scan.json"))
    ks = sorted(map(int, l))
    big[r] = (ks, [l[str(k)]["follow"] for k in ks], "-o")
scan(ax[2], big, 36, "c  Qwen 3B (36 layers)", {"lexical": "lexical (6 epochs)"})
fig.savefig("data/experiments/lab_book/compare_05b_3b.png", dpi=150)
