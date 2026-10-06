"""Bootstrap confidence intervals and across-seed spread for the Qwen2.5-3B experiments.

Resampling units: matched cue pairs (both sides kept together) for held-out pair metrics,
prompts for free generation. Critic nulls count as not following (same as free_critic_summary).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data/experiments"
OUT = EXP / "statistics_3b"
B = 10_000
RULES = ["s1", "voice", "clause", "lexical", "lexical_e6"]
RULE_OF = {"lexical_e6": "lexical"}
FOLLOW = {"s1": "critic_follows_first_sentence", "voice": "critic_follows_voice",
          "clause": "critic_follows_clause_order", "lexical": "rule_follow"}


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


class Pairs:
    """Per-arm prediction matrices aligned on pair_index, shape (pairs, 2) = (pos, neg)."""

    def __init__(self, folder):
        self.arms = {}
        for path in sorted(Path(folder).glob("*.jsonl")):
            rows = read(path)
            index = sorted({r["pair_index"] for r in rows})
            self.index = getattr(self, "index", index)
            assert index == self.index, path
            pos = {r["pair_index"]: r["prediction"] for r in rows if r["side"] == "pos"}
            neg = {r["pair_index"]: r["prediction"] for r in rows if r["side"] == "neg"}
            assert all(r["cue_label"] == (r["side"] == "pos") for r in rows), path
            self.arms[path.stem] = np.array([[pos[i], neg[i]] for i in index])

    def follow(self, arm):
        m = self.arms[arm]
        return np.stack([m[:, 0] == 1, m[:, 1] == 0], 1).mean(1)  # per-pair follow in {0, .5, 1}

    def agree(self, a, b):
        return (self.arms[a] == self.arms[b]).mean(1)


def boot(per_unit, rng_idx):
    """per_unit: 1-D per-unit statistic; returns point, CI from shared resample indices."""
    return {"value": float(per_unit.mean()), "ci95": ci(per_unit[rng_idx].mean(1)), "n": int(len(per_unit))}


def indices(n, seed=0):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def pair_stats(folder, arms=("selected",)):
    p = Pairs(folder)
    idx = indices(len(p.index))
    out = {"adapter_follow": boot(p.follow("unablated"), idx), "base_follow": boot(p.follow("base"), idx),
           "learned_gain": boot(p.follow("unablated") - p.follow("base"), idx)}
    for arm in arms:
        if arm not in p.arms:
            continue
        drop = p.follow("unablated") - p.follow(arm)
        out[arm] = {"follow": boot(p.follow(arm), idx), "follow_drop": boot(drop, idx),
                    "base_preservation": boot(p.agree("base_" + arm, "base"), idx),
                    "predicts_1": boot(p.arms[arm].mean(1), idx)}
    randoms = sorted(a for a in p.arms if a.startswith("random_"))
    if randoms:
        drops = np.stack([p.follow("unablated") - p.follow(a) for a in randoms])
        out["random_max_drop"] = float(drops.mean(1).max())
        out["random_mean_drop"] = boot(drops.mean(0), idx)
    return out


def free_follow(path, rule):
    return np.array([bool(r.get(FOLLOW[rule])) for r in read(path)], float)


def free_file(folder, arm, rule):
    name = f"free_{arm}.jsonl" if rule == "lexical" else f"free_{arm}_critic.jsonl"
    path = Path(folder) / name
    return path if path.exists() else None


def free_stats(folder, rule, arm="adapter_selected"):
    base = free_file(folder, "adapter_unablated", rule)
    if base is None:
        return None
    before = free_follow(base, rule)
    idx = indices(len(before), 1)
    out = {"adapter_follow": boot(before, idx)}
    after_path = free_file(folder, arm, rule)
    if after_path is not None:
        after = free_follow(after_path, rule)
        rows_b, rows_a = read(base), read(after_path)
        assert [r["index"] for r in rows_b] == [r["index"] for r in rows_a]
        out[arm] = {"follow": boot(after, idx), "follow_drop": boot(before - after, idx)}
        if rule != "lexical":
            out[arm]["critic_null"] = sum(r.get(FOLLOW[rule]) is None for r in rows_a)
    return out


def rule_dir(folder):
    found = [p.parent for p in Path(folder).glob("*/unablated.jsonl")]
    assert len(found) == 1, (folder, found)
    return found[0]


def seed_runs(name):
    """(seed, baseline_dir, scan_dir) for every seed with artifacts; seed 0 = original runs."""
    runs = []
    for seed in (0, 1, 2):
        baseline = EXP / "baselines/3b" / (name if seed == 0 else f"{name}_seed{seed}")
        scan = EXP / f"residuals/scans/3b_seeds/{name}_seed{seed}_residual_scan_l24plus"
        if (baseline / "experiment.json").exists():
            runs.append((seed, baseline, scan if (scan / "summary.json").exists() else None))
    return runs


def spread(values):
    v = np.array([x for x in values if x is not None], float)
    if not len(v):
        return None
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
            "min": float(v.min()), "max": float(v.max()), "seeds": int(len(v))}


def main():
    stats = {"method": {"bootstrap_resamples": B, "pair_unit": "matched cue pair",
                        "free_unit": "prompt", "critic_null": "counts as not following"},
             "seed0_full_scan": {}, "transfer": {}, "seeds": {}}
    for name in RULES:
        rule = RULE_OF.get(name, name)
        entry = {"paired": pair_stats(rule_dir(EXP / "baselines/3b" / name), arms=())}
        entry["free"] = free_stats(EXP / "baselines/3b" / name, rule)
        scan = EXP / f"residuals/scans/3b_full/{name}_residual_scan_3b_full"
        if (scan / "summary.json").exists():
            entry["full_scan"] = pair_stats(rule_dir(scan))
            entry["full_scan"]["selected_layer"] = json.loads((scan / "experiment.json").read_text())["selected_layer"]
            entry["full_scan_free"] = free_stats(scan / "free", rule)
        stats["seed0_full_scan"][name] = entry
    for layer in (30, 35):
        folder = EXP / f"residuals/transfer/four_rule_3b_l{layer}_lexical_e6"
        sel = json.loads((folder / "selection.json").read_text())
        stats["transfer"][f"L{layer}"] = {"rank": sel["selected_rank"], "passed": sel["passed"], **{
            r: pair_stats(folder / r, arms=("selected", "own")) for r in ["s1", "voice", "clause", "lexical"]}}
    for name in RULES:
        rule = RULE_OF.get(name, name)
        per_seed = {}
        for seed, baseline, scan in seed_runs(name):
            s = {"paired": pair_stats(rule_dir(baseline), arms=()), "free": free_stats(baseline, rule)}
            if scan is not None:
                exp = json.loads((scan / "experiment.json").read_text())
                s["scan"] = pair_stats(rule_dir(scan))
                s["scan"]["selected_layer"] = exp["selected_layer"]
                s["scan"]["base_control_passed"] = exp["selection_base_control_passed"]
                s["scan_free"] = free_stats(scan / "free", rule)
                strong = scan / "free_strongest"
                if (strong / "strongest.json").exists():
                    s["strongest"] = json.loads((strong / "strongest.json").read_text())
                    s["strongest_free"] = (s["scan_free"] if s["strongest"].get("same_as_selected")
                                           else free_stats(strong, rule))
            per_seed[seed] = s
        g = lambda f: spread([f(s) for s in per_seed.values()])
        get = lambda s, *keys: _get(s, keys)
        stats["seeds"][name] = {"per_seed": per_seed, "across_seeds": {
            "adapter_follow": g(lambda s: get(s, "paired", "adapter_follow", "value")),
            "free_follow": g(lambda s: get(s, "free", "adapter_follow", "value")),
            "scan_selected_drop": g(lambda s: get(s, "scan", "selected", "follow_drop", "value")),
            "scan_base_preservation": g(lambda s: get(s, "scan", "selected", "base_preservation", "value")),
            "scan_free_drop": g(lambda s: get(s, "scan_free", "adapter_selected", "follow_drop", "value")),
            "strongest_free_drop": g(lambda s: get(s, "strongest_free", "adapter_selected", "follow_drop", "value")),
        }}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def _get(d, keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


if __name__ == "__main__":
    s = main()
    fmt = lambda b: f"{b['value']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]"
    for name, e in s["seed0_full_scan"].items():
        line = f"{name:11s} adapter {fmt(e['paired']['adapter_follow'])} base {fmt(e['paired']['base_follow'])}"
        if "full_scan" in e:
            fs = e["full_scan"]
            line += f" | L{fs['selected_layer']} drop {fmt(fs['selected']['follow_drop'])} basepres {fmt(fs['selected']['base_preservation'])} randmax {fs['random_max_drop']:.3f}"
        if e.get("free"):
            line += f" | free {fmt(e['free']['adapter_follow'])}"
        print(line)
    for L, t in s["transfer"].items():
        for r in ["s1", "voice", "clause", "lexical"]:
            x = t[r]
            print(L, r, "shared", fmt(x["selected"]["follow"]), "bp", fmt(x["selected"]["base_preservation"]),
                  "| own", fmt(x["own"]["follow"]), "bp", fmt(x["own"]["base_preservation"]))
    print({k: [*v["per_seed"]] for k, v in s["seeds"].items()})
