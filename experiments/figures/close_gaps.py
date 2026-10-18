"""Paper-ready figures from the committed close-gaps JSON summaries."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
    plot_style.panel_title(ax, "a", "Lexical rule removal")

    row = data["lexical_e10"]
    row05 = data["lexical_05b"]
    for variant, label, color, linestyle in variants:
        sample = data[variant]
        depth = np.asarray(sample["layers"], dtype=float) / max(sample["layers"])
        bx.plot(depth, [r["base_pres"] for r in sample["own"]], color=color, ls=linestyle, lw=1)
    bx.axhline(0.95, color=plot_style.CHANCE_COLOR, ls=":", lw=0.7)
    bx.set(xlim=(0, 1), ylim=(0.65, 1.02), xlabel="Relative depth", ylabel="Base answers kept")
    plot_style.panel_title(bx, "b", "Base preservation")
    ax.legend(handles=model_styles + metric_styles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.subplots_adjust(bottom=0.36, wspace=0.42, top=0.9)
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
        plot_style.panel_title(ax, "abcd"[i], rule)
        ax.set(ylim=(0.35, 1.02), xlabel="Layer", ylabel="Rule-following rate")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4)
    fig.subplots_adjust(bottom=0.13, top=0.91, hspace=0.42, wspace=0.28)
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
    plot_style.panel_title(ax, "a", "Answer share by layer")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    save(fig, "fig_answer_share")


def fig_cue_cosines():
    data = read("cue_split_all_layers.json")["cosines"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(plot_style.DOUBLE_COLUMN, 3.0), sharex=True, sharey=True)
    layers = np.arange(36)
    for rule in RULES:
        ax.plot(layers, data["same_rule_across_seeds"][rule], color=plot_style.RULE_COLORS[rule], label=rule)
    plot_style.panel_title(ax, "a", "Same-rule cue directions across seeds")
    ax.set(xlabel="Layer", ylabel="Cosine similarity", ylim=(-0.25, 1.02))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    pairs = list(data["cross_rule"])
    pair_handles = []
    for pair in pairs:
        rule_a, rule_b = pair.split("-")
        values = data["cross_rule"][pair]
        bx.plot(layers, values, color=plot_style.RULE_COLORS[rule_a], label=pair)
        bx.plot(layers, values, color=plot_style.RULE_COLORS[rule_b], ls="--")
        pair_handles.append(
            Line2D([], [], color=plot_style.RULE_COLORS[rule_a], ls="-", label=pair)
        )
        if pair == "voice-clause":
            bx.annotate("voice–clause", (layers[-1], values[-1]), xytext=(5, 2),
                        textcoords="offset points", fontsize=6.5, color=plot_style.RULE_COLORS["voice"])
    plot_style.panel_title(bx, "b", "Cross-rule cue directions")
    bx.set(xlabel="Layer")
    bx.axhline(0, color=plot_style.CHANCE_COLOR, ls=":", lw=0.7)
    bx.legend(handles=pair_handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.subplots_adjust(bottom=0.28, top=0.9, wspace=0.3)
    save(fig, "fig_cue_cosines")


def fig_shared_cue():
    data = read("shared_cue.json")
    targets = ("s1", "voice", "clause", "lexical")
    layers = (27, 28, 29, 30)
    arms = (
        ("shared_loo", "Shared cue", "D", "-"),
        ("own_cue", "Own cue", "^", "--"),
        ("own_answer", "Own answer", "s", ":"),
    )
    fig, axes = plt.subplots(4, 2, figsize=(plot_style.DOUBLE_COLUMN, 7.2), sharex=True)
    rule_color = plot_style.RULE_COLORS
    for row_index, target in enumerate(targets):
        left, right = axes[row_index]
        for arm, label, marker, linestyle in arms:
            drops = [
                100 * data[f"{target}_L{layer}"][arm]["gap_drop"] if f"{target}_L{layer}" in data else np.nan
                for layer in layers
            ]
            left.plot(layers, drops, color=rule_color[target], marker=marker, ls=linestyle,
                      ms=3.2, label=label)

        shared, random_p95, random_max = [], [], []
        for layer in layers:
            item = data.get(f"{target}_L{layer}")
            if item is None:
                shared.append(np.nan)
                random_p95.append(np.nan)
                random_max.append(np.nan)
                continue
            baseline = item["unablated"]["gap"]
            random_drop = 100 * (1 - np.asarray(item["random_gap"], dtype=float) / baseline)
            shared.append(100 * item["shared_loo"]["gap_drop"])
            random_p95.append(np.percentile(random_drop, 95))
            random_max.append(np.max(random_drop))
        left.set_ylim(0, 100)
        left.set_yticks([0, 25, 50, 75, 100])
        left.set_ylabel(f"{target}\nGap drop (%)", color=rule_color[target])
        left.axhline(0, color=plot_style.CHANCE_COLOR, ls=":", lw=0.6)
        right.fill_between(layers, random_p95, random_max, color=plot_style.RANDOM_COLOR, alpha=0.45)
        right.plot(layers, random_p95, color=plot_style.RANDOM_COLOR, ls="--", lw=0.8)
        right.plot(layers, random_max, color=plot_style.BASE_COLOR, ls=":", lw=0.8)
        right.plot(layers, shared, color=rule_color[target], marker="D", lw=1.2, ms=3.2)
        right.set_ylim(0, 5)
        right.set_yticks([0, 1, 2, 3, 4, 5])
        right.axhline(0, color=plot_style.CHANCE_COLOR, ls=":", lw=0.6)
        missing = [layer for layer in layers if f"{target}_L{layer}" not in data]
        for layer in missing:
            left.text(layer, 97, "not run", rotation=90, ha="center", va="top", fontsize=6.2)
            right.text(layer, 4.8, "not run", rotation=90, ha="center", va="top", fontsize=6.2)
    plot_style.panel_title(axes[0, 0], "a", "All interventions (0–100%)")
    plot_style.panel_title(axes[0, 1], "b", "Shared cue vs random (0–5%)")
    for ax in axes[-1]:
        ax.set_xlabel("Layer")
        ax.set_xticks(layers)
    handles = [
        Line2D([], [], color="#333333", marker=marker, ls=linestyle, label=label)
        for _, label, marker, linestyle in arms
    ]
    handles.extend(
        [
            Line2D([], [], color=plot_style.RANDOM_COLOR, ls="--", label="Random 95th percentile"),
            Line2D([], [], color=plot_style.BASE_COLOR, ls=":", label="Random maximum"),
            Patch(facecolor=plot_style.RANDOM_COLOR, alpha=0.45, label="Random 95th–maximum band"),
        ]
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3)
    fig.subplots_adjust(top=0.94, bottom=0.16, left=0.1, right=0.98, hspace=0.42, wspace=0.3)
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
