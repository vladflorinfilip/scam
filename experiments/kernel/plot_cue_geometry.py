"""Figure for experiments/kernel/cue_geometry.py."""
import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.kernel.opposite import ROOT

OUT = ROOT / "cue_decomposition"


def main():
    R = json.loads((OUT / "results.json").read_text()); G = R["geometry"]; L = R["layers"]
    pairs = list(G[0]["cue"]["cross"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    col = {"normal": "grey", "answer": "tab:red", "cue": "tab:blue"}
    lab = {"normal": "raw arrow (what the PCA plot shows)", "answer": "answer part (1-0)", "cue": "cue part (answer removed)"}
    for name in ("normal", "answer", "cue"):
        for i, p in enumerate(pairs):
            ax[0].plot(L, [g[name]["cross"][p]["cos_heldout"] for g in G], color=col[name], ls=["-", "--", ":"][i],
                       label=f"{lab[name]}" if i == 0 else None)
    ax[0].fill_between(L, 0, [np.mean([g["cue"]["cross"][p]["noise95"] for p in pairs]) for g in G], color="tab:blue", alpha=.15,
                       label="noise level (cue part)")
    ax[0].axhline(0, color="k", lw=.5)
    ax[0].set(xlabel="layer", ylabel="cosine between two rules (held-out)", ylim=(-.4, 1),
              title="a  How similar are two rules' arrows?\nsolid s1-voice, dashed s1-clause, dotted voice-clause")
    ax[0].legend(fontsize=8, loc="upper left")
    for i, p in enumerate(pairs):
        ax[1].plot(L, [g["cue"]["cross"][p]["disattenuated"] for g in G], "o-", ms=3, label=p)
    for q in R["rules"]:
        ax[1].plot(L, [g["cue"]["self_heldout"][q] for g in G], color="k", alpha=.3, lw=1)
    ax[1].plot([], [], color="k", alpha=.3, label="same rule, fit vs held-out (ceiling)")
    ax[1].axhline(0, color="k", lw=.5)
    ax[1].set(xlabel="layer", ylabel="cosine", ylim=(-.6, 1.05), title="b  Cue parts only, corrected for noise\n(cross-rule cosine / ceiling)")
    ax[1].legend(fontsize=8)
    T = R["tests"]; names = [f"{s}→{t}" for s, t in dict.fromkeys((x["source"], x["target"]) for x in T)]
    M = np.array([[next(x["d"] for x in T if x["layer"] == l and f'{x["source"]}→{x["target"]}' == n) for l in L] for n in names])
    S = np.array([[next(x["p_holm"] for x in T if x["layer"] == l and f'{x["source"]}→{x["target"]}' == n) for l in L] for n in names])
    im = ax[2].imshow(M, cmap="RdBu_r", vmin=-4.5, vmax=4.5, aspect="auto")
    for i in range(len(names)):
        for j in range(len(L)):
            ax[2].text(j, i, f"{M[i, j]:.1f}" + ("*" if S[i, j] < .05 else ""), ha="center", va="center", fontsize=7)
    ax[2].set(xticks=range(len(L)), xticklabels=L, yticks=range(len(names)), yticklabels=names, xlabel="layer",
              title="c  Does rule A's cue direction read rule B's cue?\n(held-out effect size, answer directions removed; * Holm p<0.05)")
    fig.colorbar(im, ax=ax[2], shrink=.8)
    fig.suptitle(f"Qwen 3B, seed 0. Rules with a working label-flipped model: {', '.join(R['rules'])}. " + "".join(f". {k} skipped: flipped model failed (follows cue {max(v.values()):.0%})" for k, v in R["skipped"].items()), fontsize=10)
    fig.savefig(OUT / "cue_geometry.png", dpi=160); fig.savefig(OUT / "cue_geometry.pdf")


if __name__ == "__main__":
    main()
