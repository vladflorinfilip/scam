"""Figure for experiments.kernel.l35: transfer matrix and the readout decomposition at 3B layer 35."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "data/experiments/kernel_exploration/l35"
RULES = ["s1", "voice", "clause", "lexical"]
COLS = [("dir:s1", "s1"), ("dir:voice", "voice"), ("dir:clause", "clause"), ("dir:lexical", "lexical"),
        ("common_loo", "common\n(other 3)"), ("base_answer", "base-model\nanswer"), ("readout_r", "readout\n'1'−'0'")]
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def main():
    R = json.loads((OUT / "results.json").read_text())
    T = {(x["seed"], x["target"], x["arm"]): x for x in R["transfer"]}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), gridspec_kw={"width_ratios": [1.7, 1, 1]})

    ax = axes[0]
    M = np.array([[np.mean([T[(s, t, a)]["remaining"] for s in (0, 2)]) for a, _ in COLS] for t in RULES])
    im = ax.imshow(np.clip(M, -0.5, 1.1), cmap="viridis", vmin=-0.5, vmax=1.1, aspect="auto")
    for i, t in enumerate(RULES):
        for j, (a, _) in enumerate(COLS):
            rows = [T[(s, t, a)] for s in (0, 2)]
            sig = all(x.get("p_signflip_holm", 1) < .05 for x in rows)
            kept = np.mean([x["base_kept"] for x in rows])
            txt = f"{M[i, j]:.2f}" + ("*" if sig else "")
            ax.text(j, i - .12, txt, ha="center", va="center", fontsize=9, fontweight="bold" if sig else "normal",
                    color="white" if M[i, j] < .45 else "black")
            ax.text(j, i + .22, f"kept {kept:.0%}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] < .45 else "black")
    ax.set_xticks(range(len(COLS)), [c for _, c in COLS]); ax.set_yticks(range(4), RULES)
    ax.set_xlabel("direction removed at layer 35"); ax.set_ylabel("adapter (rule tested)")
    ax.axvline(3.5, color="w", lw=2)
    ax.set_title("a  How much of the learned cue effect remains", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_ticks([-0.5, 0, 0.5, 1]); cb.set_ticklabels(["≤−0.5\nreversed", "0\nremoved", "0.5", "1\nuntouched"])

    ax = axes[1]
    arms = [("dir:{t}", "own direction", "#3b75af"), ("dir:{t}_perp_r", "own, readout part removed", "#bbbbbb"),
            ("readout_r", "readout direction only", "#e08a3c")]
    w = .26
    for k, (a, lab, c) in enumerate(arms):
        vals = [[T[(s, t, a.format(t=t))]["remaining"] for s in (0, 2)] for t in RULES]
        x = np.arange(4) + (k - 1) * w
        ax.bar(x, np.mean(vals, 1), w, color=c, label=lab)
        for xi, v in zip(x, vals):
            ax.plot([xi, xi], v, "k.", ms=3)
    ax.axhline(0, color="k", lw=.8); ax.axhline(1, color="k", lw=.5, ls=":")
    ax.set_xticks(range(4), RULES); ax.set_ylabel("learned cue effect remaining")
    ax.set_title("b  Layer-35 removal acts only via the readout", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax = axes[2]
    Pr = [x for x in R["probes"] if x["space"] == "perp_r_a"]
    arms = [("loo", "rbf", "kernel (RBF), other 3 rules", "#3b75af"), ("loo", "linear", "linear, other 3 rules", "#86b4dc"),
            ("answer_loo", "rbf", "kernel, base-model answers only", "#e08a3c")]
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
    ax.set_xticks(range(4), RULES); ax.set_ylabel("held-out cue score (mean / sd)")
    ax.set_title("c  Probes trained without the tested rule", loc="left", fontweight="bold")
    ax.set_ylim(top=9.5)
    ax.legend(frameon=False, fontsize=6.8, loc="upper center", ncol=2)
    fig.text(.01, -.06, "Qwen2.5-3B, layer 35, held-out pairs; values average adapter seeds 0 and 2 (dots in b: each seed). "
             "Remaining = (cue gap after − base gap) / (cue gap before − base gap). * = beats r-matched sign-flip random directions "
             "in both seed sets (Holm p<0.05).\n'kept' = base-model answers unchanged by the same removal. "
             "c: probes trained without the tested rule, in the space with the readout and base-answer directions projected out; "
             "* = beats sign-flipped training data in both seed sets (Holm p<0.05).", fontsize=7, va="top")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"l35_transfer.{ext}", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
