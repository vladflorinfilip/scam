"""Threshold-free check of every saved held-out ablation: is the cue effect removed, or just shifted?

Per arm: within-pair cue ordering (cue-1 side scores higher), cue gap (mean margin cue-1 minus cue-0),
follow at the 0 threshold, follow at a threshold cross-fitted on the other half of pairs, and base
preservation (base-model answers unchanged by the same ablation).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data/experiments/residuals"
SKIP = {"unablated", "base", "baseline"}


def load(p):
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {(r["pair_index"], r["side"]): r for r in rows if "logit_margin" in r and "pair_index" in r and "side" in r}


def stats(D, keys, pairs):
    m = np.array([D[k]["logit_margin"] for k in keys]); y = np.array([D[k]["cue_label"] for k in keys])
    ok = np.mean([max((0, 1), key=lambda c: next(D[(p, s)]["logit_margin"] for s in ("pos", "neg") if D[(p, s)]["cue_label"] == c)) == 1 for p in pairs])
    return m, y, float(ok), float(m[y == 1].mean() - m[y == 0].mean())


def xfit(m, y, kp, pairs, rng, n=300):
    out = []
    for _ in range(n):
        tr = set(rng.choice(pairs, len(pairs) // 2, replace=False)); t = np.array([p in tr for p in kp])
        best = max(np.unique(m[t]), key=lambda c: ((m[t] > c) == (y[t] == 1)).mean())
        out.append(((m[~t] > best) == (y[~t] == 1)).mean())
    return float(np.mean(out))


rng = np.random.default_rng(0)
res = {}
for d in sorted(ROOT.rglob("unablated.jsonl")):
    d = d.parent
    if not (d / "base.jsonl").exists():
        continue
    U, B = load(d / "unablated.jsonl"), load(d / "base.jsonl")
    if not U:
        continue
    keys = sorted(U); pairs = sorted({p for p, s in keys if (p, "pos") in U and (p, "neg") in U})
    keys = [k for k in keys if k[0] in set(pairs)]; kp = np.array([k[0] for k in keys])
    _, _, u_ok, u_gap = stats(U, keys, pairs); bm, by, b_ok, b_gap = stats(B, keys, pairs)
    for f in sorted(d.glob("*.jsonl")):
        arm = f.stem
        if arm in SKIP or arm.startswith("base_") or arm.endswith("_critic"):
            continue
        A = load(f)
        if set(keys) - set(A):
            continue
        m, y, ok, gap = stats(A, keys, pairs)
        r = {"pairs": len(pairs), "unablated_pair_ok": u_ok, "base_pair_ok": b_ok, "pair_ok": ok,
             "unablated_gap": u_gap, "base_gap": b_gap, "gap": gap,
             "follow_at_0": float(((m > 0) == (y == 1)).mean()), "follow_xfit": xfit(m, y, kp, pairs, rng),
             "predicts_1": float((m > 0).mean())}
        if (d / f"base_{arm}.jsonl").exists():
            BA = load(d / f"base_{arm}.jsonl")
            r["base_preservation"] = float(np.mean([(BA[k]["logit_margin"] > 0) == (B[k]["logit_margin"] > 0) for k in keys]))
        res[f"{d.relative_to(ROOT)}/{arm}"] = r
dst = ROOT.parent / "statistics_3b/pair_check.json"
dst.write_text(json.dumps(res, indent=1))
print(f"{'run/arm':74s} n   okU  okB  okA | gapU gapB gapA | f@0 fxfit p1  bp")
for k, r in res.items():
    print(f"{k[-74:]:74s} {r['pairs']:3d} {r['unablated_pair_ok']:.2f} {r['base_pair_ok']:.2f} {r['pair_ok']:.2f} | "
          f"{r['unablated_gap']:5.1f} {r['base_gap']:5.1f} {r['gap']:5.1f} | {r['follow_at_0']:.2f} {r['follow_xfit']:.2f} {r['predicts_1']:.2f} {r.get('base_preservation', float('nan')):.2f}")
