"""Figure + 3D HTML for experiments/kernel/opposite.py (reads results.json only)."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments import plot_style

OUT = Path(__file__).resolve().parents[2] / "data/experiments/kernel_exploration/opposite"
COL = {"clause": plot_style.RULE_COLORS["clause"], "voice": plot_style.RULE_COLORS["voice"]}


def main():
    plot_style.apply()
    R = json.loads((OUT / "results.json").read_text())
    L = R["layers"]
    fig, ax = plt.subplots(2, 2, figsize=(plot_style.DOUBLE_COLUMN, 4.4))
    g = R["l35_geometry"]; xyz = np.array(g["pca_coords"])
    a = ax[0, 0]
    style = lambda n: ("black", "-") if n.startswith(("readout", "base")) else (
        (COL.get(n.split()[0], "tab:green"), "--" if "flipped" in n else (":" if "part" in n else "-")))
    for n, (x, _, z) in zip(g["names"], xyz):
        c, ls = style(n)
        a.annotate("", (x, z), (0, 0), arrowprops=dict(arrowstyle="->", color=c, ls=ls, lw=1.6))
        a.text(x * 1.05, z * 1.05, n, fontsize=7, color=c)
    a.axvline(0, color=plot_style.CHANCE_COLOR, lw=.5); a.axhline(0, color=plot_style.CHANCE_COLOR, lw=.5)
    a.set(xlim=(-1.1, 1.1), ylim=(-0.6, 1.0), xlabel="PC1 (same axes as the original 3D plot)", ylabel="PC3",
          title="Layer 35: the opposite side holds the flipped-answer models")
    a.text(-.08, 1.08, "a", transform=a.transAxes, fontweight="bold")
    a = ax[0, 1]
    for rule in ("clause", "voice"):
        rows = R["rules"][rule]["by_layer"]
        y = [r["answer_share"] for r in rows]; lo, hi = zip(*[r["answer_share_ci95"] for r in rows])
        a.plot(L, y, "o-", color=COL[rule], label=rule); a.fill_between(L, lo, hi, color=COL[rule], alpha=.2)
    a.axhline(.5, color=plot_style.CHANCE_COLOR, ls=":")
    a.set(ylim=(0, 1), xlabel="Layer", ylabel="answer part / (answer + cue part), squared size",
          title="How much of the rule direction is answer"); a.legend()
    a.text(-.08, 1.08, "b", transform=a.transAxes, fontweight="bold")
    a = ax[1, 0]
    for rule in ("clause", "voice"):
        rows = R["rules"][rule]["by_layer"]
        a.plot(L, [r["answer_heldout_z"] for r in rows], "o-", color=COL[rule], label=f"{rule} answer part")
        a.plot(L, [r["cue_heldout_z"] for r in rows], "s:", color=COL[rule], label=f"{rule} cue part")
    a.axhline(0, color="grey", lw=.5)
    a.set(xlabel="Layer", ylabel="held-out effect size (mean / sd over test pairs)",
          title="Both parts are real on held-out pairs (all Holm p < 0.01)"); a.legend(fontsize=7)
    a.text(-.08, 1.08, "c", transform=a.transAxes, fontweight="bold")
    a = ax[1, 1]
    for src, tgt, m in (("clause", "voice", "o-"), ("voice", "clause", "s-")):
        t = [x for x in R["cross_cue_tests"] if x["source"] == src and x["part"] == "cue"]
        a.plot(L, [x["d"] for x in t], m, color=COL[src], label=f"cue part {src} → {tgt}")
        for x in t:
            if x["p_holm"] < .05: a.plot(x["layer"], x["d"], "k*", ms=6)
        t = [x for x in R["cross_cue_tests"] if x["source"] == src and x["part"] == "answer"]
        a.plot(L, [x["d"] for x in t], m, color=plot_style.MODEL_DARK_COLORS[src], alpha=.8, label=f"answer part {src} → {tgt}")
    a.axhline(0, color=plot_style.CHANCE_COLOR, lw=.5)
    a.set(xlabel="Layer", ylabel="held-out effect size on the other rule",
          title="Shared across rules?\n(Cue part with answer directions removed; * Holm p < 0.05)")
    a.text(-.08, 1.08, "d", transform=a.transAxes, fontweight="bold")
    a.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "opposite.png", dpi=plot_style.EXPORT_DPI, bbox_inches="tight"); fig.savefig(OUT / "opposite.pdf", bbox_inches="tight")
    for ext in ("png", "pdf"):
        shutil.copyfile(OUT / f"opposite.{ext}", Path("data/experiments/lab_book") / f"opposite.{ext}")

    import plotly.graph_objects as go
    HEX = {"tab:green": plot_style.RULE_COLORS["clause"], "black": "#444444",
           **{color: color for color in COL.values()}}
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
