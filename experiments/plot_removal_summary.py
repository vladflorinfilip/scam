"""Paper figure: one score per rule and layer (seeds pooled) with 95% bootstrap intervals.

Points are seed means of the pair-check values. Intervals come from a hierarchical bootstrap:
resample seeds, then held-out pairs within each seed. With one seed this is a pair bootstrap.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1] / "data/experiments"
RES = ROOT / "residuals"
OUT = ROOT / "statistics_3b"
COLORS = {"s1": "#1f77b4", "voice": "#ff7f0e", "clause": "#2ca02c", "lexical": "#9467bd"}
SEEDS = "scans/3b_seeds/{}_seed{}_residual_scan_l24plus"
L35 = "transfer/four_rule_3b_l35_lexical_e6"
KEYS = ("unablated_follow_at_0", "follow_at_0", "follow_xfit", "base_follow_xfit", "base_preservation")
# (model, rule, label, [(dir, arm), ...] one entry per seed)
OLD = [("0.5B", r, f"{r} · seed 0 · layer {l}", [(f"scans/late_l20plus/{r}_residual_scan_l20plus", "selected")])
       for r, l in (("s1", 22), ("voice", 20), ("clause", 21), ("lexical", 22))]
# (model, rule, label, [(dir, arm), ...] one entry per seed)
FIGURES = {
    "removal_summary": OLD + [
        ("3B", "s1", "s1 · seeds 0–2 · layer 26–34 (per seed)", [(SEEDS.format("s1", s), "selected") for s in (0, 1, 2)]),
        ("3B", "voice", "voice · seed 0 · layer 32", [("scans/3b_late_l30plus/voice_residual_scan_l30plus", "selected")]),
        ("3B", "voice", "voice · seed 0 · layer 35", [(L35, "own")]),
        ("3B", "clause", "clause · seed 2 · layer 32", [(SEEDS.format("clause", 2), "selected")]),
        ("3B", "clause", "clause · seed 0 · layer 35", [(L35, "own")]),
        ("3B", "lexical", "lexical (6 ep.) · seed 0 · layer 35", [(L35, "own")])],
    "removal_summary_l26_l35": OLD + [
        ("3B", "s1", "s1 · seed 2 · layer 26", [(SEEDS.format("s1", 2), "selected")]),
        ("3B", "s1", "s1 · seed 0 · layer 35", [(L35, "own")]),
        ("3B", "voice", "voice · seed 0 · layer 35", [(L35, "own")]),
        ("3B", "clause", "clause · seed 0 · layer 35", [(L35, "own")]),
        ("3B", "lexical", "lexical (6 ep.) · seed 0 · layer 35", [(L35, "own")])],
}


def load(p):
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {(r["pair_index"], r["side"]): r for r in rows if "logit_margin" in r and "side" in r}


def by_pair(D, pairs):
    """Margins of the cue-1 and cue-0 version of each pair."""
    m1, m0 = [], []
    for p in pairs:
        a, b = D[(p, "pos")], D[(p, "neg")]
        one, zero = (a, b) if a["cue_label"] == 1 else (b, a)
        m1.append(one["logit_margin"]); m0.append(zero["logit_margin"])
    return np.array(m1), np.array(m0)


def arrays(d, rule, arm):
    f = RES / d / rule
    D = {k: load(f / f"{k}.jsonl") for k in ("unablated", arm, "base", f"base_{arm}")}
    pairs = sorted({p for p, _ in D[arm]} & {p for p, _ in D["base"]} & {p for p, _ in D[f"base_{arm}"]})
    m = {k: by_pair(v, pairs) for k, v in D.items()}
    b1, b0 = m["base"]; a1, a0 = m[f"base_{arm}"]
    kept = (((a1 > 0) == (b1 > 0)).astype(float) + ((a0 > 0) == (b0 > 0))) / 2
    return m["unablated"], m[arm], m["base"], kept


def follow0(m1, m0):
    return ((m1 > 0).mean() + (m0 <= 0).mean()) / 2


def xfit(m1, m0, rng, n=10):
    """Threshold chosen on a random half of pairs, accuracy on the other half."""
    out, P = [], len(m1)
    for _ in range(n):
        t = np.zeros(P, bool); t[rng.choice(P, P // 2, replace=False)] = True
        c = np.unique(np.r_[m1[t], m0[t]])
        acc = (np.searchsorted(np.sort(m1[t]), c, side="right") * -1 + t.sum()
               + np.searchsorted(np.sort(m0[t]), c, side="right"))
        best = c[np.argmax(acc)]
        out.append(((m1[~t] > best).mean() + (m0[~t] <= best).mean()) / 2)
    return np.mean(out)


def stat(seed_arrays, idx, rng):
    unab, abl, base, kept = seed_arrays
    return [follow0(*(x[idx] for x in unab)), follow0(*(x[idx] for x in abl)),
            xfit(*(x[idx] for x in abl), rng), xfit(*(x[idx] for x in base), rng), kept[idx].mean()]


def summarise(cells, rng, B=1000):
    check = json.loads((OUT / "pair_check.json").read_text())
    point = np.mean([[check[f"{d}/{rule}/{arm}"][k] for k in KEYS] for d, rule, arm in cells], 0)
    data = [arrays(d, rule, arm) for d, rule, arm in cells]
    boots = []
    for _ in range(B):
        seeds = rng.integers(len(data), size=len(data))
        boots.append(np.mean([stat(data[s], rng.integers(len(data[s][3]), size=len(data[s][3])), rng) for s in seeds], 0))
    lo, hi = np.percentile(boots, [2.5, 97.5], 0)
    return {k: {"mean": float(p), "lo": float(min(l, p)), "hi": float(max(h, p))} for k, p, l, h in zip(KEYS, point, lo, hi)}


def figure(name, rows, rng):
    data = []
    for model, rule, label, cells in rows:
        s = summarise([(d, rule, arm) for d, arm in cells], rng)
        n = len(cells)
        data.append(dict(model=model, rule=rule, label=label, seeds=n, **s))
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=2))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 5.0), sharey=True, gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.06})
    ys, y, prev = [], 0, None
    for row in data:
        if prev and row["model"] != prev:
            y += 1.2
        ys.append(y); y += 1; prev = row["model"]

    err = lambda r, k: [[(r[k]["mean"] - r[k]["lo"]) * 100], [(r[k]["hi"] - r[k]["mean"]) * 100]]
    for yi, r in zip(ys, data):
        c = COLORS[r["rule"]]
        ax.plot([30, 100], [yi, yi], color="#eeeeee", lw=0.8, zorder=0)
        ax.plot([r["base_follow_xfit"]["mean"] * 100] * 2, [yi - .38, yi + .38], color="#555555", lw=2.4, solid_capstyle="butt", zorder=1)
        ax.scatter(r["unablated_follow_at_0"]["mean"] * 100, yi, s=34, facecolor="white", edgecolor="black", lw=1, zorder=3)
        ax.errorbar(r["follow_at_0"]["mean"] * 100, yi - .17, xerr=err(r, "follow_at_0"), fmt="X", ms=6, color=c, alpha=.6,
                    mec="none", elinewidth=1, capsize=2, zorder=3)
        ax.errorbar(r["follow_xfit"]["mean"] * 100, yi + .17, xerr=err(r, "follow_xfit"), fmt="o", ms=6, color=c, mec="black",
                    mew=.6, ecolor=c, elinewidth=1.4, capsize=2.5, zorder=4)
        k = r["base_preservation"]; kept = k["mean"] * 100
        bx.barh(yi, kept - 50, left=50, height=.62, color=c if kept >= 95 else "#cfcfcf", edgecolor="none")
        bx.errorbar(kept, yi, xerr=err(r, "base_preservation"), fmt="none", ecolor="black", elinewidth=1, capsize=2)
        bx.text(101, yi, f"{kept:.0f}%", va="center", ha="left", fontsize=7.5, color="#333333", clip_on=False)

    ax.set_yticks(ys, [r["label"] for r in data]); ax.invert_yaxis()
    for t, r in zip(ax.get_yticklabels(), data):
        t.set_color(COLORS[r["rule"]])
    ax.set_xlim(30, 102); ax.set_xlabel("Hidden rule followed (%)")
    ax.axvline(50, color="#999999", ls=":", lw=.9, zorder=0)
    ax.text(50.6, ys[-1] + 0.7, "chance", ha="left", va="center", fontsize=7.5, color="#777777")
    bx.axvline(95, color="black", ls="--", lw=.9); bx.set_xlim(50, 100); bx.set_xlabel("Normal answers kept (%)")
    bx.text(95, ys[0] - 1.05, "95% safety bar", ha="center", fontsize=7.5)
    bx.tick_params(left=False)
    for m, label in (("0.5B", "Qwen2.5-0.5B"), ("3B", "Qwen2.5-3B")):
        first = ys[[r["model"] for r in data].index(m)]
        ax.text(-0.02, first - .85, label, transform=ax.get_yaxis_transform(), ha="right", fontsize=9.5, fontweight="bold")
    ax.set_title("a   Is the hidden rule removed?", loc="left", fontsize=10, fontweight="bold", pad=70)
    bx.set_title("b   Side effects", loc="left", fontsize=10, fontweight="bold", pad=70)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="Before removal"),
               Line2D([], [], marker="X", ls="", color="#777777", alpha=.6, mec="none", label="After removal: the model's actual answers"),
               Line2D([], [], marker="o", ls="", color="#777777", mec="black", label="After removal: does the cue still change the score? (best cut-off)"),
               Line2D([], [], color="#555555", lw=2.4, label="Base model, no hidden rule (best cut-off)")]
    ax.legend(handles=handles, ncol=1, frameon=False, fontsize=7.8, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
              columnspacing=1.2, handletextpad=.4)
    ax.text(-0.02, -0.11, "●: best cut-off chosen on held-out half of pairs. ● at the grey tick = rule removed; ● near 100% = rule still present.\n"
            "Bars: 95% bootstrap intervals (resampling seeds, then test pairs).",
            transform=ax.transAxes, fontsize=7, color="#555555", va="top", ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print(json.dumps([(r["label"], {k: round(r[k]["mean"] * 100, 1) for k in KEYS}, {k: (round(r[k]["lo"] * 100), round(r[k]["hi"] * 100)) for k in KEYS}) for r in data], indent=0))
    plt.close(fig)


for name, rows in FIGURES.items():
    figure(name, rows, np.random.default_rng(0))
