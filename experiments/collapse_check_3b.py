"""Is post-ablation collapse an offset (cue info still present) or real cue removal?

For each scan's held-out selected arm: cue AUC from margins (threshold-free), follow at the
0 threshold vs at the median margin, and correlation of ablated margins with base margins.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data/experiments/residuals"


def load(p):
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    return np.array([r["logit_margin"] for r in rows]), np.array([r["cue_label"] for r in rows])


def auc(m, y):
    pos, neg = m[y == 1], m[y == 0]
    return float(((pos[:, None] > neg[None]).mean() + .5 * (pos[:, None] == neg[None]).mean()))


def follow(m, y, t):
    return float(((m > t) == (y == 1)).mean())


out = {}
for d in sorted(ROOT.glob("scans/*/*")) + sorted(ROOT.glob("transfer/*")):
    for rule_dir in sorted(p for p in d.iterdir() if p.is_dir() and (p / "unablated.jsonl").exists()):
        u, y = load(rule_dir / "unablated.jsonl")
        b, _ = load(rule_dir / "base.jsonl")
        for arm in [a for a in ("selected", "own") if (rule_dir / f"{a}.jsonl").exists()]:
            m, _ = load(rule_dir / f"{arm}.jsonl")
            out[f"{d.parent.name}/{d.name}/{rule_dir.name}/{arm}"] = {
                "predicts_1": float((m > 0).mean()),
                "follow_at_0": follow(m, y, 0), "follow_at_median": follow(m, y, np.median(m)),
                "cue_auc_unablated": auc(u, y), "cue_auc_ablated": auc(m, y),
                "margin_mean_unablated": float(u.mean()), "margin_mean_ablated": float(m.mean()),
                "margin_sd_unablated": float(u.std()), "margin_sd_ablated": float(m.std()),
                "corr_with_base": float(np.corrcoef(m, b)[0, 1]),
                "agree_base_at_median": float(((m > np.median(m)) == (b > 0)).mean()),
            }
dst = ROOT.parent / "statistics_3b/collapse_check.json"
dst.write_text(json.dumps(out, indent=1))
print(f"{'run':78s} p1   f@0  f@med AUCu AUCa  mu_u  mu_a  sd_u sd_a  r_base agr_med")
for k, v in out.items():
    print(f"{k[:78]:78s} {v['predicts_1']:.2f} {v['follow_at_0']:.2f} {v['follow_at_median']:.2f}  "
          f"{v['cue_auc_unablated']:.2f} {v['cue_auc_ablated']:.2f} {v['margin_mean_unablated']:5.1f} {v['margin_mean_ablated']:5.1f} "
          f"{v['margin_sd_unablated']:4.1f} {v['margin_sd_ablated']:4.1f}  {v['corr_with_base']:5.2f} {v['agree_base_at_median']:.2f}")
