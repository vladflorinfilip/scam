"""Seed-comparison figures for the 3B statistical analysis."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCANS = ROOT / "data/experiments/residuals/scans/3b_seeds"
OUT = ROOT / "data/experiments/statistics_3b"
S = json.loads((OUT / "stats.json").read_text())["seeds"]
RULES = ["s1", "voice", "clause", "lexical_e6"]
COL = {"0": "C0", "1": "C1", "2": "C2"}


def layers():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True, layout="constrained")
    for ax, r in zip(axes.flat, RULES):
        for seed in "012":
            f = SCANS / f"{r}_seed{seed}_residual_scan_l24plus/layer_scan.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            xs = sorted(map(int, d))
            sel = S[r]["per_seed"][seed]["scan"]["selected_layer"]
            ax.plot(xs, [d[str(x)]["follow"] for x in xs], "-o", ms=3, color=COL[seed], label=f"seed {seed}: rule follow")
            ax.plot(xs, [d[str(x)]["base_preservation"] for x in xs], ":", color=COL[seed], label=f"seed {seed}: base preserved")
            ax.plot(sel, d[str(sel)]["follow"], "*", ms=14, color=COL[seed], mec="k")
        ax.axhline(.95, color="k", lw=.8, ls="--")
        ax.axhline(.5, color="grey", lw=.8, ls="--")
        ax.set(title=r + (" (6 epochs, off-recipe)" if r == "lexical_e6" else ""), ylim=(0, 1.05))
        ax.legend(fontsize=7, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("Layer (direction removed)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Rate (selection pairs, n=60)")
    fig.suptitle("Own-rule ablation by layer, per training seed  (★ = selected base-safe layer; "
                 "dashed black = 95% base-preservation bar; grey = chance)")
    fig.savefig(OUT / "seeds_layer_scans.png", dpi=150)


def err(ax, x, b, **kw):
    if not b:
        return
    v, (lo, hi) = b["value"], b["ci95"]
    ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="o", capsize=3, **kw)


def summary():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    names = ["s1", "voice", "clause", "lexical", "lexical_e6"]
    ax = axes[0]
    for i, r in enumerate(names):
        for j, seed in enumerate("012"):
            p = S[r]["per_seed"][seed]["paired"]
            err(ax, i + (j - 1) * .2, p["adapter_follow"], color=COL[seed], label=f"seed {seed}" if i == 0 else None)
        b = S[r]["per_seed"]["0"]["paired"]["base_follow"]["value"]
        ax.hlines(b, i - .35, i + .35, color="k", lw=1.5, label="base model" if i == 0 else None)
    ax.set(xticks=range(len(names)), xticklabels=["s1", "voice", "clause", "lexical\n(3 ep)", "lexical_e6\n(6 ep)"],
           ylim=(.4, 1.03), ylabel="Rule following (held-out pairs, 95% CI)", title="Training: does each seed learn the rule?")
    ax.axhline(.9, color="grey", ls=":", lw=1)
    ax.legend(fontsize=8, loc="lower right")
    for ax, key, title in [(axes[1], "scan_free", "Free generation: selected base-safe layer"),
                           (axes[2], "strongest_free", "Free generation: strongest layer")]:
        k = 0
        ticks = []
        for r in RULES:
            for seed in "012":
                f = S[r]["per_seed"][seed].get(key)
                if not f:
                    continue
                err(ax, k - .15, f["adapter_follow"], color="grey")
                err(ax, k + .15, f["adapter_selected"]["follow"], color=COL[seed])
                ax.plot([k - .15, k + .15], [f["adapter_follow"]["value"], f["adapter_selected"]["follow"]["value"]], color="k", lw=.6)
                layer = (S[r]["per_seed"][seed]["scan"]["selected_layer"] if key == "scan_free"
                         else S[r]["per_seed"][seed]["strongest"]["layer"])
                ticks.append(f"{r}\ns{seed} L{layer}")
                k += 1
        ax.set(xticks=range(k), xticklabels=ticks, ylim=(0, 1.05), title=title,
               ylabel="Adapter rule following (100 prompts, 95% CI)")
        ax.tick_params(axis="x", labelsize=7)
        ax.text(.01, .02, "grey = before removal, coloured = after", transform=ax.transAxes, fontsize=8)
    fig.savefig(OUT / "seeds_training_and_free.png", dpi=150)


if __name__ == "__main__":
    layers()
    summary()
