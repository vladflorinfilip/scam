"""Figure for experiments/kernel/cue_geometry.py."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments import plot_style
from experiments.kernel.opposite import ROOT

OUT = ROOT / "cue_decomposition"


def main():
    plot_style.apply()
    R = json.loads((OUT / "results.json").read_text()); G = R["geometry"]; L = R["layers"]
    pairs = list(G[0]["cue"]["cross"])
    fig, ax = plt.subplots(1, 3, figsize=(plot_style.DOUBLE_COLUMN, 3.0))
    col = {"normal": plot_style.BASE_COLOR, "answer": plot_style.MODEL_DARK_COLORS["s1"], "cue": plot_style.RULE_COLORS["s1"]}
    lab = {"normal": "raw arrow (what the PCA plot shows)", "answer": "answer part (1-0)", "cue": "cue part (answer removed)"}
    for name in ("normal", "answer", "cue"):
        V = np.array([[g[name]["cross"][p]["cos_heldout"] for g in G] for p in pairs])
        ax[0].plot(L, V.mean(0), color=col[name], lw=1.1, label=lab[name] + " (mean)")
        ax[0].fill_between(L, V.min(0), V.max(0), color=col[name], alpha=.12)
    ax[0].plot(L, [np.mean([g["cue"]["cross"][p]["noise95"] for p in pairs]) for g in G], color=plot_style.RULE_COLORS["s1"], ls=":", label="noise level (cue part)")
    ax[0].axhline(0, color=plot_style.CHANCE_COLOR, lw=.5)
    ax[0].set(xlabel="layer", ylabel="cosine between two rules (held-out)", ylim=(-.4, 1),
              title="How similar are two rules' arrows?\nMean and min–max over rule pairs")
    ax[0].text(-.1, 1.08, "a", transform=ax[0].transAxes, fontweight="bold")
    ax[0].legend(fontsize=6.4, loc="upper left")
    for i, p in enumerate(pairs):
        ax[1].plot(L, [g["cue"]["cross"][p]["disattenuated"] for g in G], "o-", ms=2.5, label=p,
                   color=plt.cm.Greys(.35 + .5 * i / max(1, len(pairs) - 1)))
    for q in R["rules"]:
        ax[1].plot(L, [g["cue"]["self_heldout"][q] for g in G], color="k", alpha=.3, lw=1)
    ax[1].plot([], [], color="k", alpha=.3, label="same rule, fit vs held-out (ceiling)")
    ax[1].axhline(0, color="k", lw=.5)
    ax[1].set(xlabel="Layer", ylabel="Cosine", ylim=(-.6, 1.05), title="Cue parts, corrected for noise")
    ax[1].text(-.1, 1.08, "b", transform=ax[1].transAxes, fontweight="bold")
    ax[1].legend(fontsize=6.3)
    T = R["tests"]; names = [f"{s}→{t}" for s, t in dict.fromkeys((x["source"], x["target"]) for x in T)]
    M = np.array([[next(x["d"] for x in T if x["layer"] == l and f'{x["source"]}→{x["target"]}' == n) for l in L] for n in names])
    S = np.array([[next(x["p_holm"] for x in T if x["layer"] == l and f'{x["source"]}→{x["target"]}' == n) for l in L] for n in names])
    im = ax[2].imshow(M, cmap="RdBu_r", vmin=-4.5, vmax=4.5, aspect="auto")
    for i in range(len(names)):
        for j in range(len(L)):
            ax[2].text(j, i, f"{M[i, j]:.1f}" + ("*" if S[i, j] < .05 else ""), ha="center", va="center", fontsize=7)
    ax[2].set(xticks=range(len(L)), xticklabels=L, yticks=range(len(names)), yticklabels=names, xlabel="layer",
              title="Cross-rule cue readout\nHeld-out effect size; * Holm p < 0.05")
    ax[2].text(-.1, 1.08, "c", transform=ax[2].transAxes, fontweight="bold")
    fig.colorbar(im, ax=ax[2], shrink=.8)
    fig.tight_layout()
    fig.savefig(OUT / "cue_geometry.png", dpi=plot_style.EXPORT_DPI, bbox_inches="tight")
    fig.savefig(OUT / "cue_geometry.pdf", bbox_inches="tight")
    for directory in ("data/experiments/lab_book",):
        for ext in ("png", "pdf"):
            shutil.copyfile(OUT / f"cue_geometry.{ext}", Path(directory) / f"cue_geometry.{ext}")


if __name__ == "__main__":
    main()
