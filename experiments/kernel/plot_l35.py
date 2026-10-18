"""Figure for experiments.kernel.l35: transfer matrix and the readout decomposition at 3B layer 35."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import plot_style

OUT = Path(__file__).resolve().parents[2] / "data/experiments/kernel_exploration/l35"
RULES = ["s1", "voice", "clause", "lexical"]
COLS = [("dir:s1", "s1"), ("dir:voice", "voice"), ("dir:clause", "clause"), ("dir:lexical", "lexical"),
        ("common_loo", "common\n(other 3)"), ("base_answer", "base-model\nanswer"), ("readout_r", "readout\n'1'−'0'")]
def main():
    plot_style.apply()
    R = json.loads((OUT / "results.json").read_text())
    T = {(x["seed"], x["target"], x["arm"]): x for x in R["transfer"]}
    fig, axes = plt.subplots(1, 3, figsize=(plot_style.DOUBLE_COLUMN, 3.7), gridspec_kw={"width_ratios": [1.7, 1, 1]})

    ax = axes[0]
    M = np.array([[np.mean([T[(s, t, a)]["remaining"] for s in (0, 2)]) for a, _ in COLS] for t in RULES])
    im = plot_style.heatmap(ax, np.clip(M, -0.5, 1.1), cmap="viridis", vmin=-0.5, vmax=1.1, aspect="auto")
    for i, t in enumerate(RULES):
        for j, (a, _) in enumerate(COLS):
            rows = [T[(s, t, a)] for s in (0, 2)]
            sig = all(x.get("p_signflip_holm", 1) < .05 for x in rows)
            kept = np.mean([x["base_kept"] for x in rows])
            txt = f"{M[i, j]:.2f}" + ("*" if sig else "")
            text_color = plot_style.heatmap_text_color(im, M[i, j])
            ax.text(j, i - .12, txt, ha="center", va="center", fontsize=8, color=text_color)
            ax.text(j, i + .22, f"kept {kept:.0%}", ha="center", va="center", fontsize=6.5, color=text_color)
    ax.set_xticks(range(len(COLS)), [c for _, c in COLS]); ax.set_yticks(range(4), RULES)
    ax.set_xlabel("Direction removed at layer 35"); ax.set_ylabel("Adapter (rule tested)")
    ax.tick_params(axis="x", labelrotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.axvline(3.5, color="w", lw=2)
    plot_style.panel_title(ax, "a", "How much cue effect remains")
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_ticks([-0.5, 0, 0.5, 1]); cb.set_ticklabels(["≤−0.5\nreversed", "0\nremoved", "0.5", "1\nuntouched"])

    ax = axes[1]
    arms = [("dir:{t}", "own direction", plot_style.RULE_COLORS["s1"]), ("dir:{t}_perp_r", "own, readout part removed", plot_style.BASE_COLOR),
            ("readout_r", "readout direction only", plot_style.MODEL_DARK_COLORS["s1"])]
    w = .26
    for k, (a, lab, c) in enumerate(arms):
        vals = [[T[(s, t, a.format(t=t))]["remaining"] for s in (0, 2)] for t in RULES]
        x = np.arange(4) + (k - 1) * w
        ax.bar(x, np.mean(vals, 1), w, color=c, label=lab)
        for xi, v in zip(x, vals):
            ax.plot([xi, xi], v, "k.", ms=3)
    ax.axhline(0, color="k", lw=.8); ax.axhline(1, color="k", lw=.5, ls=":")
    ax.set_xticks(range(4), RULES); ax.set_ylabel("Learned cue effect remaining")
    plot_style.panel_title(ax, "b", "Removal acts via\nthe readout", fontsize=8)
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", ncol=1, bbox_to_anchor=(.5, -.32))
    ax = axes[2]
    Pr = [x for x in R["probes"] if x["space"] == "perp_r_a"]
    arms = [("loo", "rbf", "kernel (RBF), other 3 rules", plot_style.RULE_COLORS["s1"]), ("loo", "linear", "linear, other 3 rules", plot_style.MODEL_LIGHT_COLORS["s1"]),
            ("answer_loo", "rbf", "kernel, base-model answers only", plot_style.BASE_COLOR)]
    for k, (tr, mdl, lab, c) in enumerate(arms):
        rows = [[x for x in Pr if x["target"] == t and x["train"] == tr and x["model"] == mdl] for t in RULES]
        x = np.arange(4) + (k - 1) * w
        ax.bar(x, [np.mean([y["z"] for y in r_]) for r_ in rows], w, color=c, label=lab)
        for xi, r_ in zip(x, rows):
            ax.plot([xi] * len(r_), [y["z"] for y in r_], "k.", ms=3)
            ax.plot([xi - w / 2, xi + w / 2], [np.mean([y["null_z95"] for y in r_])] * 2, color="crimson", lw=1.5)
            if tr == "loo" and all(y.get("p_holm", 1) < .05 for y in r_):
                ax.text(xi, max(y["z"] for y in r_) + .2, "*", ha="center", fontsize=11)
    ax.plot([], [], color="crimson", lw=1.5, label="random-sign null, 95th pct")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(4), RULES); ax.set_ylabel("Held-out cue score (mean / sd)")
    plot_style.panel_title(ax, "c", "Probes without\nthe tested rule", fontsize=8)
    ax.set_ylim(top=9.5)
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", ncol=1, bbox_to_anchor=(.5, -.32))
    fig.text(.01, .01, "Qwen2.5-3B, layer 35, held-out pairs; values average adapter seeds 0 and 2 (dots in b: each seed). "
             "Remaining = (cue gap after − base gap) / (cue gap before − base gap). * = beats r-matched sign-flip random directions "
             "in both seed sets (Holm p<0.05).\n'kept' = base-model answers unchanged by the same removal. "
             "c: probes trained without the tested rule, in the space with the readout and base-answer directions projected out; "
             "* = beats sign-flipped training data in both seed sets (Holm p<0.05).", fontsize=6.2, va="bottom")
    fig.subplots_adjust(left=.06, right=.99, top=.8, bottom=.34, wspace=.45)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"l35_transfer.{ext}", dpi=plot_style.EXPORT_DPI, bbox_inches="tight")


if __name__ == "__main__":
    main()
