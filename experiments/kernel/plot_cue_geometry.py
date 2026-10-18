"""Figure for experiments/kernel/cue_geometry.py."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments import plot_style
from experiments.kernel.opposite import ROOT

OUT = ROOT / "cue_decomposition"


def main():
    plot_style.apply()
    R = json.loads((OUT / "results.json").read_text())
    G, L = R["geometry"], R["layers"]
    pairs = list(G[0]["cue"]["cross"])
    fig = plt.figure(figsize=(plot_style.DOUBLE_COLUMN, 6.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.3), hspace=0.9, wspace=0.3)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    col = {"normal": plot_style.BASE_COLOR, "answer": plot_style.MODEL_DARK_COLORS["s1"], "cue": plot_style.RULE_COLORS["s1"]}
    lab = {"normal": "raw arrow (what the PCA plot shows)", "answer": "answer part (1-0)", "cue": "cue part (answer removed)"}
    for name in ("normal", "answer", "cue"):
        V = np.array([[g[name]["cross"][p]["cos_heldout"] for g in G] for p in pairs])
        ax_a.plot(L, V.mean(0), color=col[name], lw=1.1, label=lab[name] + " (mean)")
        ax_a.fill_between(L, V.min(0), V.max(0), color=col[name], alpha=.12)
    ax_a.plot(L, [np.mean([g["cue"]["cross"][p]["noise95"] for p in pairs]) for g in G],
              color=plot_style.RULE_COLORS["s1"], ls=":", label="noise level (cue part)")
    ax_a.axhline(0, color=plot_style.CHANCE_COLOR, lw=.5)
    ax_a.set(xlabel="Layer", ylabel="Cosine between two rules (held-out)", ylim=(-.4, 1))
    plot_style.panel_title(ax_a, "a", "How similar are two rules' arrows?")
    ax_a.legend(fontsize=6.1, loc="upper center", bbox_to_anchor=(.5, -.34), ncol=2)

    pair_handles = []
    for i, p in enumerate(pairs):
        rule_a, rule_b = p.split("-")
        values = [g["cue"]["cross"][p]["disattenuated"] for g in G]
        ax_b.plot(L, values, color=plot_style.RULE_COLORS[rule_a], marker="o", ms=2.2)
        ax_b.plot(L, values, color=plot_style.RULE_COLORS[rule_b], ls="--", lw=1.0)
        pair_handles.append(Line2D([], [], color=plot_style.RULE_COLORS[rule_a], label=p))
        if p == "voice-clause":
            ax_b.annotate("voice–clause", (L[-1], values[-1]), xytext=(4, 1),
                          textcoords="offset points", fontsize=6.2,
                          color=plot_style.RULE_COLORS["voice"])
    for q in R["rules"]:
        ax_b.plot(L, [g["cue"]["self_heldout"][q] for g in G], color="#444444", alpha=.35, lw=.9)
    pair_handles.append(Line2D([], [], color="#444444", alpha=.5, label="same-rule ceiling"))
    ax_b.axhline(0, color=plot_style.CHANCE_COLOR, lw=.5)
    ax_b.set(xlabel="Layer", ylabel="Cosine", ylim=(-.6, 1.05))
    plot_style.panel_title(ax_b, "b", "Cue parts, corrected for noise")
    ax_b.legend(handles=pair_handles, fontsize=5.8, loc="upper center",
                bbox_to_anchor=(.5, -.34), ncol=3)

    T = R["tests"]
    names = [f"{s}→{t}" for s, t in dict.fromkeys((x["source"], x["target"]) for x in T)]
    M = np.array([[next(x["d"] for x in T if x["layer"] == l and f'{x["source"]}→{x["target"]}' == n) for l in L] for n in names])
    S = np.array([[next(x["p_holm"] for x in T if x["layer"] == l and f'{x["source"]}→{x["target"]}' == n) for l in L] for n in names])
    im = plot_style.heatmap(ax_c, M, cmap="RdBu_r", vmin=-4.5, vmax=4.5, aspect="auto")
    plot_style.annotate_heatmap(ax_c, im, M, stars=S < .05, show_values=False, fontsize=7)
    tick_positions = np.arange(0, len(L), 2)
    ax_c.set(xticks=tick_positions, xticklabels=[L[i] for i in tick_positions],
             yticks=range(len(names)), yticklabels=names, xlabel="Layer")
    plot_style.panel_title(ax_c, "c", "Cross-rule cue readout (held-out effect size; * Holm p < 0.05)")
    fig.colorbar(im, ax=ax_c, fraction=.025, pad=.015)
    fig.subplots_adjust(left=.1, right=.96, top=.96, bottom=.05)
    fig.savefig(OUT / "cue_geometry.png", dpi=plot_style.EXPORT_DPI, bbox_inches="tight")
    fig.savefig(OUT / "cue_geometry.pdf", bbox_inches="tight")
    for directory in ("data/experiments/lab_book",):
        for ext in ("png", "pdf"):
            shutil.copyfile(OUT / f"cue_geometry.{ext}", Path(directory) / f"cue_geometry.{ext}")


if __name__ == "__main__":
    main()
