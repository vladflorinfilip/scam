"""Regenerate figures whose original sources were not standalone plot scripts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import plot_style

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/experiments"


def export(fig, name, paths):
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(path.with_suffix(f".{ext}"), dpi=plot_style.EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def training():
    stats = json.loads((DATA / "statistics_3b/stats.json").read_text())["seeds"]
    names = ("s1", "voice", "clause", "lexical", "lexical_e6")
    labels = ("s1", "voice", "clause", "lexical\n(3 ep)", "lexical\n(6 ep)")
    fig, ax = plt.subplots(figsize=(plot_style.DOUBLE_COLUMN, 2.8))
    for i, rule in enumerate(names):
        entry = stats[rule]["per_seed"]
        for seed in ("0", "1", "2"):
            if seed not in entry:
                continue
            result = entry[seed]["paired"]["adapter_follow"]
            value, (low, high) = result["value"], result["ci95"]
            ax.errorbar(i + (int(seed) - 1) * 0.16, value, yerr=[[value - low], [high - value]],
                        fmt="o", color=plot_style.RULE_COLORS["lexical" if rule.startswith("lexical") else rule],
                        ms=3.5, capsize=2, label=f"seed {seed}" if i == 0 else None)
        base = entry["0"]["paired"]["base_follow"]["value"]
        ax.plot([i - .28, i + .28], [base, base], color=plot_style.BASE_COLOR, lw=1.2)
    ax.axhline(.9, color=plot_style.CHANCE_COLOR, ls=":", lw=.7)
    ax.set(xticks=range(len(names)), xticklabels=labels, ylim=(.4, 1.03),
           ylabel="Rule-following rate", title="Training success across seeds")
    ax.legend(loc="lower right", ncol=3)
    ax.text(-.07, 1.08, "a", transform=ax.transAxes, fontweight="bold")
    fig.tight_layout()
    export(fig, "training", [DATA / "lab_book/training", DATA / "supervisor_summary/training"])


def three_model_geometry():
    data = json.loads((DATA / "model_comparison/geometry.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(plot_style.DOUBLE_COLUMN, 4.6),
                             gridspec_kw={"height_ratios": [1, 1.4]})
    models = [
        ("qwen05b", "Qwen 0.5B", plot_style.MODEL_LIGHT_COLORS["s1"]),
        ("qwen3b", "Qwen 3B", plot_style.RULE_COLORS["s1"]),
        ("gptoss20b", "GPT-OSS-20B", plot_style.MODEL_DARK_COLORS["s1"]),
    ]
    for ax, (key, label, color) in zip(axes[0], models):
        if key == "qwen05b":
            matrix = np.asarray(data[key]["l18"]["cos"], dtype=float)
            cross = matrix[~np.eye(len(matrix), dtype=bool)]
            depth = np.asarray([18 / 23])
            mean = np.asarray([cross.mean()])
            null = np.asarray([0.0])
        else:
            rows = data[key]
            depth = np.asarray([r["depth"] for r in rows])
            mean = np.asarray([r["mean_cross_cos"] for r in rows])
            null = np.asarray([r["null_abs95"] for r in rows])
        ax.plot(depth, mean, color=color, marker="o", ms=2.5, label="mean cross-rule cosine")
        ax.fill_between(depth, -null, null, color=plot_style.RANDOM_COLOR, alpha=.35, linewidth=0)
        ax.axhline(0, color=plot_style.CHANCE_COLOR, ls=":", lw=.7)
        ax.set(xlim=(0, 1), ylim=(-.55, .85), title=label, xlabel="Relative depth", ylabel="Mean cross-rule cosine")
    views = data["pca_views"]
    colors = [plot_style.RULE_COLORS[r] for r in data["rules"]]
    view_names = ("Qwen 0.5B L18 (depth 0.78; no lexical)", "Qwen 3B L35 (last)", "gpt-oss-20b L23 (last)")
    for ax, title in zip(axes[1], view_names):
        view = views[title]
        coords = np.asarray(view["coords"])
        for i, rule in enumerate(data["rules"][:len(coords)]):
            x, y, z = coords[i]
            ax.annotate("", (x, z), (0, 0), arrowprops={"arrowstyle": "->", "color": colors[i], "lw": 1.1})
            ax.text(x * 1.04, z * 1.04, rule, color=colors[i], fontsize=7)
        ax.axhline(0, color=plot_style.RANDOM_COLOR, lw=.5)
        ax.axvline(0, color=plot_style.RANDOM_COLOR, lw=.5)
        ax.set(title=title, xlabel="PC1", ylabel="PC3", xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal")
    fig.tight_layout()
    export(fig, "geometry_3models", [DATA / "lab_book/geometry_3models"])


def residual_directions():
    data = json.loads((DATA / "model_comparison/geometry.json").read_text())
    title = next(name for name in data["pca_views"] if name.startswith("Qwen 0.5B L18"))
    view = data["pca_views"][title]
    coords = np.asarray(view["coords"])
    cos = np.asarray(view["cos"])
    rules = data["rules"][:len(coords)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(plot_style.DOUBLE_COLUMN, 2.6), gridspec_kw={"width_ratios": [1.2, 1]})
    for i, (rule, (x, _, z)) in enumerate(zip(rules, coords)):
        color = plot_style.RULE_COLORS[rule]
        ax.annotate("", (x, z), (0, 0), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
        ax.text(x * 1.04, z * 1.04, rule, color=color, fontsize=7)
    ax.axhline(0, color=plot_style.RANDOM_COLOR, lw=.5)
    ax.axvline(0, color=plot_style.RANDOM_COLOR, lw=.5)
    ax.set(title="Qwen 0.5B, layer 18", xlabel="PC1", ylabel="PC3", xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal")
    image = bx.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(rules)):
        for j in range(len(rules)):
            bx.text(j, i, f"{cos[i,j]:.2f}", ha="center", va="center", fontsize=7)
    bx.set(title="Direction cosine", xticks=range(len(rules)), yticks=range(len(rules)),
           xticklabels=rules, yticklabels=rules)
    fig.colorbar(image, ax=bx, fraction=.05, pad=.03)
    fig.tight_layout()
    export(fig, "residual_directions_3d", [DATA / "lab_book/residual_directions_3d"])


def l35_panels():
    results = json.loads((DATA / "kernel_exploration/l35/results.json").read_text())
    transfer = {(x["seed"], x["target"], x["arm"]): x for x in results["transfer"]}
    rules = ("s1", "voice", "clause", "lexical")
    columns = ("dir:s1", "dir:voice", "dir:clause", "dir:lexical", "common_loo", "base_answer", "readout_r")
    values = np.asarray([[np.mean([transfer[(seed, target, arm)]["remaining"] for seed in (0, 2)])
                          for arm in columns] for target in rules])
    fig, ax = plt.subplots(figsize=(plot_style.DOUBLE_COLUMN, 2.0))
    image = ax.imshow(values, cmap="RdYlBu_r", vmin=-.5, vmax=1.1, aspect="auto")
    for i, rule in enumerate(rules):
        for j in range(len(columns)):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if values[i,j] < .3 else "#222222")
    ax.set(xticks=range(len(columns)), xticklabels=("s1", "voice", "clause", "lexical", "common\nLOO", "base\nanswer", "readout"),
           yticks=range(4), yticklabels=rules, xlabel="Direction removed", ylabel="Target rule",
           title="Layer 35: cue effect remaining")
    fig.colorbar(image, ax=ax, fraction=.025, pad=.02)
    fig.tight_layout()
    export(fig, "l35_a", [DATA / "lab_book/l35_a", DATA / "supervisor_summary/l35_a"])

    probe_rows = [row for row in results["probes"] if row["space"] == "perp_r_a"]
    fig, axes = plt.subplots(1, 2, figsize=(plot_style.DOUBLE_COLUMN, 2.7))
    arms = [("dir:{t}", "remaining", "Own direction", plot_style.BASE_COLOR),
            ("dir:{t}_perp_r", "remaining", "Own, readout removed", plot_style.RULE_COLORS["s1"]),
            ("readout_r", "remaining", "Readout only", plot_style.MODEL_DARK_COLORS["s1"])]
    x = np.arange(4)
    for k, (arm, _, label, color) in enumerate(arms):
        vals = [[transfer[(seed, rule, arm.format(t=rule))]["remaining"] for seed in (0, 2)] for rule in rules]
        axes[0].bar(x + (k - 1) * .24, np.mean(vals, axis=1), .24, color=color, label=label)
    axes[0].axhline(0, color=plot_style.CHANCE_COLOR, ls=":", lw=.7)
    axes[0].set(xticks=x, xticklabels=rules, ylabel="Cue effect remaining", title="Removal channels")
    axes[0].legend(fontsize=6.2)
    probe_arms = [("loo", "rbf", "RBF, other 3", plot_style.RULE_COLORS["s1"]),
                  ("loo", "linear", "Linear, other 3", plot_style.MODEL_LIGHT_COLORS["s1"]),
                  ("answer_loo", "rbf", "Base answers", plot_style.BASE_COLOR)]
    for k, (train, model, label, color) in enumerate(probe_arms):
        vals = [[row["z"] for row in probe_rows if row["target"] == rule and row["train"] == train and row["model"] == model] for rule in rules]
        axes[1].bar(x + (k - 1) * .24, [np.mean(v) if v else 0 for v in vals], .24, color=color, label=label)
    axes[1].axhline(0, color=plot_style.CHANCE_COLOR, ls=":", lw=.7)
    axes[1].set(xticks=x, xticklabels=rules, ylabel="Held-out cue score", title="Probe transfer")
    axes[1].legend(fontsize=6.2)
    fig.tight_layout()
    export(fig, "l35_bc", [DATA / "lab_book/l35_bc", DATA / "supervisor_summary/l35_bc"])


def main():
    plot_style.apply()
    training()
    three_model_geometry()
    residual_directions()
    l35_panels()


if __name__ == "__main__":
    main()
