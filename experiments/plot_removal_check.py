"""Paper figure: at each selected layer (0.5B and 3B seeds), is the hidden-rule cue removed or only shifted?"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1] / "data/experiments"
RES = ROOT / "residuals"
OUT = ROOT / "statistics_3b"
COLORS = {"s1": "#1f77b4", "voice": "#ff7f0e", "clause": "#2ca02c", "lexical": "#9467bd"}
NAMES = {"s1": "s1", "voice": "voice", "clause": "clause", "lexical": "lexical"}

check = json.loads((OUT / "pair_check.json").read_text())
SEEDS = "scans/3b_seeds/{}_seed{}_residual_scan_l24plus"
L35 = "transfer/four_rule_3b_l35_lexical_e6"
OLD = [("0.5B", r, 0, f"scans/late_l20plus/{r}_residual_scan_l20plus", "selected") for r in ("s1", "voice", "clause", "lexical")]
FIGURES = {
    "removal_check_selected_layers": OLD + [
        ("3B", r, s, SEEDS.format("lexical_e6" if r == "lexical" else r, s), "selected")
        for r, seeds in (("s1", (0, 1, 2)), ("voice", (0, 2)), ("clause", (0, 1, 2)), ("lexical", (0, 2))) for s in seeds],
    # 3B rows at layers where removing the direction changes the answers (per-example scores saved).
    "removal_check_effective_layers": OLD + [
        ("3B", "s1", 0, SEEDS.format("s1", 0), "selected"), ("3B", "s1", 1, SEEDS.format("s1", 1), "selected"),
        ("3B", "s1", 2, SEEDS.format("s1", 2), "selected"),
        ("3B", "voice", 0, "scans/3b_late_l30plus/voice_residual_scan_l30plus", "selected"), ("3B", "voice", 0, L35, "own"),
        ("3B", "clause", 0, L35, "own"), ("3B", "clause", 2, SEEDS.format("clause", 2), "selected"),
        ("3B", "lexical", 0, L35, "own")],
}


def layer_of(d):
    e = json.loads((RES / d / "experiment.json").read_text())
    return e.get("selected_layer", e.get("layer"))


def figure(name, rows):
    data = []
    for model, rule, seed, d, arm in rows:
        r = check[f"{d}/{rule}/{arm}"]
        extra = " (6 ep.)" if model == "3B" and rule == "lexical" else ""
        data.append(dict(model=model, rule=rule, label=f"{NAMES[rule]}{extra} · seed {seed} · layer {layer_of(d)}", **r))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 6.2), sharey=True, gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.06})

    ys, y, prev = [], 0, None
    for row in data:
        if prev and row["model"] != prev:
            y += 1.2
        ys.append(y); y += 1; prev = row["model"]

    for yi, r in zip(ys, data):
        c = COLORS[r["rule"]]
        ax.plot([30, 100], [yi, yi], color="#eeeeee", lw=0.8, zorder=0)
        ax.plot([r["base_follow_xfit"] * 100] * 2, [yi - .38, yi + .38], color="#555555", lw=2.4, solid_capstyle="butt", zorder=1)
        ax.annotate("", xy=(r["follow_xfit"] * 100, yi), xytext=(r["follow_at_0"] * 100, yi),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, alpha=.55, shrinkA=4, shrinkB=4), zorder=2)
        ax.scatter(r["unablated_follow_at_0"] * 100, yi, s=34, facecolor="white", edgecolor="black", lw=1, zorder=3)
        ax.scatter(r["follow_at_0"] * 100, yi, s=34, marker="X", color=c, alpha=.6, lw=0, zorder=3)
        ax.scatter(r["follow_xfit"] * 100, yi, s=42, color=c, edgecolor="black", lw=.6, zorder=4)
        kept = r["base_preservation"] * 100
        bx.barh(yi, kept - 50, left=50, height=.62, color=c if kept >= 95 else "#cfcfcf", edgecolor="none")
        bx.text(min(kept, 99.6) - .6, yi, f"{kept:.0f}%", va="center", ha="right", fontsize=7.5, color="white" if kept >= 95 else "#333333")

    ax.set_yticks(ys, [r["label"] for r in data]); ax.invert_yaxis()
    for t, r in zip(ax.get_yticklabels(), data):
        t.set_color(COLORS[r["rule"]])
    ax.set_xlim(30, 102); ax.set_xlabel("Answers that follow the hidden rule (%)")
    ax.axvline(50, color="#999999", ls=":", lw=.9, zorder=0)
    ax.text(50.6, ys[-1] + 0.62, "chance", ha="left", va="center", fontsize=7.5, color="#777777")
    bx.axvline(95, color="black", ls="--", lw=.9); bx.set_xlim(50, 100); bx.set_xlabel("Normal answers kept (%)")
    bx.text(95, ys[0] - 1.05, "95% safety bar", ha="center", fontsize=7.5)
    bx.tick_params(left=False)
    for m, label in (("0.5B", "Qwen2.5-0.5B"), ("3B", "Qwen2.5-3B")):
        first = ys[[r["model"] for r in data].index(m)]
        ax.text(-0.02, first - .85, label, transform=ax.get_yaxis_transform(), ha="right", fontsize=9.5, fontweight="bold")

    ax.set_title("a   Is the hidden rule removed?", loc="left", fontsize=10, fontweight="bold", pad=34)
    bx.set_title("b   Side effects", loc="left", fontsize=10, fontweight="bold", pad=34)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="before removal"),
               Line2D([], [], marker="X", ls="", color="#777777", alpha=.6, mec="none", label="after removal, standard scoring"),
               Line2D([], [], marker="o", ls="", color="#777777", mec="black", label="after removal, best cut-off"),
               Line2D([], [], color="#555555", lw=2.4, label="base model (no hidden rule)")]
    ax.legend(handles=handles, ncol=2, frameon=False, fontsize=7.8, loc="lower left", bbox_to_anchor=(-0.02, 1.0), columnspacing=1.2, handletextpad=.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


for name, rows in FIGURES.items():
    figure(name, rows)
    print("saved", name)
