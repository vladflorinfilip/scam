"""Paper-ready figures from the committed close-gaps JSON summaries."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments import plot_style

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/experiments/close_gaps"
RULES = ("s1", "voice", "clause", "lexical")
RULE_LABELS = {"s1": "s1", "voice": "voice", "clause": "clause", "lexical": "lexical"}


def read(name):
    return json.loads((DATA / name).read_text())


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(DATA / f"{name}.{ext}", dpi=plot_style.EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label):
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontweight="bold", va="bottom")


def fig_lexical_layers():
    data = read("layer_check.json")
    variants = [
        ("lexical_e6_seed0", "Qwen 3B · 6 epochs", plot_style.RULE_COLORS["lexical"], "-"),
        ("lexical_e10", "Qwen 3B · 10 epochs", plot_style.MODEL_DARK_COLORS["lexical"], "-."),
        ("lexical_05b", "Qwen 0.5B", plot_style.MODEL_LIGHT_COLORS["lexical"], "--"),
    ]
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(plot_style.DOUBLE_COLUMN, 2.7), gridspec_kw={"width_ratios": [2.4, 1]}
    )
    model_styles = []
    metric_styles = []
    for variant, label, color, linestyle in variants:
        row = data[variant]
        depth = np.asarray(row["layers"], dtype=float) / max(row["layers"])
        own = row["own"]
        ax.plot(depth, [r["x"] for r in own], color=color, ls=linestyle, marker="x", ms=3)
        ax.plot(depth, [r["dot"] for r in own], color=color, ls="--", marker="o", ms=2.5)
        ax.axhline(row["base"]["dot"], color=plot_style.BASE_COLOR, lw=0.65, ls=":")
        ax.plot(
            depth,
            [r["dot"] for r in row["random"]],
            color=plot_style.RANDOM_COLOR,
            lw=0.75,
            ls="--",
        )
        model_styles.append(Line2D([], [], color=color, ls=linestyle, label=label))
    metric_styles = [
        Line2D([], [], color="#333333", ls="-", marker="x", ms=3, label="× own direction"),
        Line2D([], [], color="#333333", ls="--", marker="o", ms=2.5, label="● own direction"),
        Line2D([], [], color=plot_style.BASE_COLOR, ls=":", label="base-model ●"),
        Line2D([], [], color=plot_style.RANDOM_COLOR, ls=":", label="random-direction ● mean"),
    ]
    ax.axhline(0.5, color=plot_style.CHANCE_COLOR, ls=":", lw=0.7)
    ax.set(xlim=(0, 1), ylim=(0.35, 1.02), xlabel="Relative depth (layer / last layer)", ylabel="Rule-following rate")
    ax.set_title("Lexical rule removal")
    ax.legend(handles=model_styles + metric_styles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    panel_label(ax, "a")

    row = data["lexical_e10"]
    row05 = data["lexical_05b"]
    for variant, label, color, linestyle in variants:
        sample = data[variant]
        depth = np.asarray(sample["layers"], dtype=float) / max(sample["layers"])
        bx.plot(depth, [r["base_pres"] for r in sample["own"]], color=color, ls=linestyle, lw=1)
    bx.axhline(0.95, color=plot_style.CHANCE_COLOR, ls=":", lw=0.7)
    bx.set(xlim=(0, 1), ylim=(0.65, 1.02), xlabel="Relative depth", ylabel="Base answers kept")
    bx.set_title("Base preservation")
    panel_label(bx, "b")
    fig.subplots_adjust(bottom=0.3, wspace=0.42)
    save(fig, "fig_lexical_layers")


def fig_all_rules_layers():
    data = read("layer_check.json")
    variants = [("s1_seed0", "s1"), ("voice_seed0", "voice"), ("clause_seed0", "clause"), ("lexical_e10", "lexical")]
    fig, axes = plt.subplots(2, 2, figsize=(plot_style.DOUBLE_COLUMN, 4.2), sharex=True, sharey=True)
    for i, (ax, (variant, rule)) in enumerate(zip(axes.flat, variants)):
        row = data[variant]
        layers = row["layers"]
        color = plot_style.RULE_COLORS[rule]
        ax.plot(layers, [r["x"] for r in row["own"]], color=color, ls="-", marker="x", ms=3, label="× own direction")
        ax.plot(layers, [r["dot"] for r in row["own"]], color=color, ls="--", marker="o", ms=2.5, label="● own direction")
        ax.plot(layers, [r["dot"] for r in row["random"]], color=plot_style.RANDOM_COLOR, ls="--", label="random ● mean")
        ax.axhline(row["base"]["dot"], color=plot_style.BASE_COLOR, ls="-", lw=0.8, label="base ●")
        ax.axhline(0.5, color=plot_style.CHANCE_COLOR, ls=":", lw=0.7)
        ax.set(title=rule, ylim=(0.35, 1.02), xlabel="Layer", ylabel="Rule-following rate")
        panel_label(ax, "abcd"[i])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4)
    fig.subplots_adjust(top=0.87, hspace=0.42, wspace=0.28)
    save(fig, "fig_all_rules_layers")


def fig_answer_share():
    data = read("cue_split_all_layers.json")["parts"]
    fig, ax = plt.subplots(figsize=(plot_style.DOUBLE_COLUMN, 3.0))
    layers = np.arange(36)
    for rule in RULES:
        values = np.asarray([row["answer_share"] for row in data[rule]], dtype=float)
        ax.plot(layers, values.mean(axis=0), color=plot_style.RULE_COLORS[rule], label=f"{rule} (n={len(values)})")
        ax.fill_between(layers, values.min(axis=0), values.max(axis=0), color=plot_style.RULE_COLORS[rule], alpha=0.16, linewidth=0)
    ax.set(xlim=(0, 35), ylim=(0, 1), xlabel="Layer", ylabel="Answer share of cue + answer direction")
    ax.legend(loc="upper left", ncol=2)
    panel_label(ax, "a")
    fig.tight_layout()
    save(fig, "fig_answer_share")


def fig_cue_cosines():
    data = read("cue_split_all_layers.json")["cosines"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(plot_style.DOUBLE_COLUMN, 3.0), sharex=True, sharey=True)
    layers = np.arange(36)
    for rule in RULES:
        ax.plot(layers, data["same_rule_across_seeds"][rule], color=plot_style.RULE_COLORS[rule], label=rule)
    ax.set(title="Same-rule cue directions across seeds", xlabel="Layer", ylabel="Cosine similarity", ylim=(-0.25, 1.02))
    ax.legend(loc="lower left", ncol=2)
    pairs = list(data["cross_rule"])
    grays = plt.cm.Greys(np.linspace(0.35, 0.78, len(pairs)))
    for pair, color in zip(pairs, grays):
        bx.plot(layers, data["cross_rule"][pair], color=color, label=pair)
    bx.set(title="Cross-rule cue directions", xlabel="Layer")
    bx.axhline(0, color=plot_style.CHANCE_COLOR, ls=":", lw=0.7)
    bx.legend(loc="upper left", ncol=2)
    panel_label(ax, "a")
    panel_label(bx, "b")
    fig.tight_layout()
    save(fig, "fig_cue_cosines")


def fig_shared_cue():
    data = read("shared_cue.json")
    targets = ("s1", "voice", "clause", "lexical")
    layers = (27, 28, 29, 30)
    arms = (
        ("shared_loo", "shared_loo", "#4C4C4C", "D"),
        ("own_cue", "own_cue", "#8C8C8C", "o"),
        ("own_answer", "own_answer", "#BDBDBD", "s"),
    )
    fig, axes = plt.subplots(4, 4, figsize=(plot_style.DOUBLE_COLUMN, 7.2), sharey=True)
    rng = np.random.default_rng(0)
    for row_index, target in enumerate(targets):
        for col_index, layer in enumerate(layers):
            ax = axes[row_index, col_index]
            key = f"{target}_L{layer}"
            panel_label(ax, "abcd"[col_index] if row_index == 0 else "")
            if key not in data:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_title(f"L{layer}")
                if col_index == 0:
                    ax.set_ylabel(f"{target}\nGap drop (%)")
                continue
            item = data[key]
            baseline_gap = item["unablated"]["gap"]
            random_drop = 100 * (1 - np.asarray(item["random_gap"], dtype=float) / baseline_gap)
            parts = ax.boxplot([random_drop], positions=[0], widths=0.55, patch_artist=True, showfliers=False)
            parts["boxes"][0].set(facecolor="#E5E5E5", edgecolor=plot_style.RANDOM_COLOR, linewidth=0.7)
            for whisker in parts["whiskers"]:
                whisker.set(color=plot_style.RANDOM_COLOR, linewidth=0.7)
            for cap in parts["caps"]:
                cap.set(color=plot_style.RANDOM_COLOR, linewidth=0.7)
            parts["medians"][0].set(color=plot_style.BASE_COLOR, linewidth=0.8)
            for x, (arm, label, color, marker) in enumerate(arms, start=1):
                ax.scatter(x, 100 * item[arm]["gap_drop"], s=18, color=color, marker=marker, zorder=3)
            ax.axhline(0, color=plot_style.CHANCE_COLOR, ls=":", lw=0.6)
            ax.set_title(f"L{layer}")
            ax.set_xticks([0, 1, 2, 3], ["200\nrandom", "shared", "own\ncue", "own\nanswer"], rotation=0)
            if col_index == 0:
                ax.set_ylabel(f"{target}\nGap drop (%)")
            if row_index == 0 and col_index == 3:
                ax.text(1.03, 1.05, "s1/voice L29–30: not run", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    fig.subplots_adjust(top=0.93, bottom=0.11, left=0.09, hspace=0.45, wspace=0.25)
    handles = [Line2D([], [], marker=arm[3], color=arm[2], ls="none", label=arm[1]) for arm in arms]
    handles.insert(0, Line2D([], [], color=plot_style.RANDOM_COLOR, marker="s", ls="-", label="200 random rank-3 subspaces"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4)
    save(fig, "fig_shared_cue")


def main():
    plot_style.apply()
    DATA.mkdir(parents=True, exist_ok=True)
    fig_lexical_layers()
    fig_all_rules_layers()
    fig_answer_share()
    fig_cue_cosines()
    fig_shared_cue()


if __name__ == "__main__":
    main()
