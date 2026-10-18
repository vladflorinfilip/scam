"""Figure + 3D HTML for experiments/kernel/opposite.py (reads results.json only)."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments import plot_style

OUT = Path(__file__).resolve().parents[2] / "data/experiments/kernel_exploration/opposite"
COL = plot_style.RULE_COLORS
LABEL_POSITIONS = {
    "s1": (-1.02, -0.49),
    "voice": (-1.02, -0.31),
    "voice answer part": (-1.02, -0.11),
    "base answer": (-1.02, 0.24),
    "clause answer part": (-1.02, 0.47),
    "clause": (-0.97, 0.85),
    "lexical": (-0.10, 0.48),
    "readout (1 minus 0)": (-0.40, -0.18),
    "voice cue part": (0.12, -0.16),
    "clause cue part": (0.16, 0.82),
    "clause flipped": (0.35, -0.05),
    "voice flipped": (0.42, 0.36),
}
LABELS = {
    "clause answer part": "clause answer",
    "clause cue part": "clause cue",
    "readout (1 minus 0)": "readout",
    "voice answer part": "voice answer",
    "voice cue part": "voice cue",
}


def main():
    plot_style.apply()
    R = json.loads((OUT / "results.json").read_text())
    L = R["layers"]
    fig, ax = plt.subplots(2, 2, figsize=(plot_style.DOUBLE_COLUMN, 5.2))
    g = R["l35_geometry"]; xyz = np.array(g["pca_coords"])
    a = ax[0, 0]
    style = lambda n: ("black", "-") if n.startswith(("readout", "base")) else (
        (COL.get(n.split()[0], "#444444"), "--" if "flipped" in n else (":" if "part" in n else "-")))
    for n, (x, _, z) in zip(g["names"], xyz):
        c, ls = style(n)
        a.annotate("", (x, z), (0, 0), arrowprops=dict(arrowstyle="->", color=c, ls=ls, lw=1.6))
        label_x, label_z = LABEL_POSITIONS.get(n, (x * 1.05, z * 1.05))
        a.annotate(
            LABELS.get(n, n),
            (x, z),
            xytext=(label_x, label_z),
            textcoords="data",
            fontsize=6.2,
            color=c,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color=c, lw=.45, alpha=.6, shrinkA=2, shrinkB=3),
        )
    a.axvline(0, color=plot_style.CHANCE_COLOR, lw=.5); a.axhline(0, color=plot_style.CHANCE_COLOR, lw=.5)
    a.set(xlim=(-1.1, 1.1), ylim=(-0.6, 1.0), xlabel="PC1 (same axes as the original 3D plot)", ylabel="PC3",
          )
    plot_style.panel_title(a, "a", "Flipped-answer model geometry at layer 35")
    a = ax[0, 1]
    for rule in ("clause", "voice"):
        rows = R["rules"][rule]["by_layer"]
        y = [r["answer_share"] for r in rows]; lo, hi = zip(*[r["answer_share_ci95"] for r in rows])
        a.plot(L, y, "o-", color=COL[rule], label=rule); a.fill_between(L, lo, hi, color=COL[rule], alpha=.2)
    a.axhline(.5, color=plot_style.CHANCE_COLOR, ls=":")
    a.set(ylim=(0, 1), xlabel="Layer", ylabel="answer part / (answer + cue part), squared size",
          )
    plot_style.panel_title(a, "b", "Answer component of rule direction")
    a.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2, fontsize=6.5)
    a = ax[1, 0]
    for rule in ("clause", "voice"):
        rows = R["rules"][rule]["by_layer"]
        a.plot(L, [r["answer_heldout_z"] for r in rows], "o-", color=COL[rule], label=f"{rule} answer part")
        a.plot(L, [r["cue_heldout_z"] for r in rows], "s:", color=COL[rule], label=f"{rule} cue part")
    a.axhline(0, color="grey", lw=.5)
    a.set(xlabel="Layer", ylabel="held-out effect size (mean / sd over test pairs)",
          )
    plot_style.panel_title(a, "c", "Both parts are real on held-out pairs\n(all Holm p < 0.01)")
    a.legend(fontsize=6.1, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2)
    a = ax[1, 1]
    for src, tgt, m in (("clause", "voice", "o-"), ("voice", "clause", "s-")):
        t = [x for x in R["cross_cue_tests"] if x["source"] == src and x["part"] == "cue"]
        cue_values = [x["d"] for x in t]
        a.plot(L, cue_values, m, color=COL[src], label=f"{src}–{tgt}, cue")
        a.plot(L, cue_values, ls="--", color=COL[tgt], lw=1.0)
        for x in t:
            if x["p_holm"] < .05: a.plot(x["layer"], x["d"], "k*", ms=6)
        t = [x for x in R["cross_cue_tests"] if x["source"] == src and x["part"] == "answer"]
        answer_values = [x["d"] for x in t]
        a.plot(L, answer_values, m, color=COL[src], alpha=.75, label=f"{src}–{tgt}, answer")
        a.plot(L, answer_values, ls="--", color=COL[tgt], lw=1.0, alpha=.75)
        if src == "voice" and tgt == "clause":
            a.annotate("voice–clause", (L[-1], answer_values[-1]), xytext=(4, 1),
                       textcoords="offset points", fontsize=6.2, color=COL[src])
    a.axhline(0, color=plot_style.CHANCE_COLOR, lw=.5)
    a.set(xlabel="Layer", ylabel="held-out effect size on the other rule",
          )
    plot_style.panel_title(a, "d", "Shared across rules?\nCue part with answer directions removed; * Holm p < 0.05")
    a.legend(fontsize=5.7, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2)
    fig.subplots_adjust(left=.09, right=.98, top=.94, bottom=.15, hspace=.95, wspace=.32)
    fig.savefig(OUT / "opposite.png", dpi=plot_style.EXPORT_DPI, bbox_inches="tight"); fig.savefig(OUT / "opposite.pdf", bbox_inches="tight")
    for ext in ("png", "pdf"):
        shutil.copyfile(OUT / f"opposite.{ext}", Path("data/experiments/lab_book") / f"opposite.{ext}")

    import plotly.graph_objects as go
    HEX = {"black": "#444444", **{color: color for color in COL.values()}}
    f3 = go.Figure()
    for n, (x, y, z) in zip(g["names"], xyz.tolist()):
        c, ls = style(n)
        f3.add_trace(go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode="lines+markers+text", text=["", n], name=n,
                                  line=dict(color=HEX[c], dash={"-": "solid", "--": "dash", ":": "dot"}[ls], width=5)))
    f3.update_layout(title="Layer 35, original PCA axes: normal, flipped, cue part, answer part, readout",
                     scene_aspectmode="cube")
    f3.write_html(str(OUT / "directions_3d_opposite.html"), include_plotlyjs=True)


if __name__ == "__main__":
    main()
