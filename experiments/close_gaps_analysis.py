"""CPU analysis for close-gaps: per-layer × and ● curves, all-layer multi-seed cue/answer split, shared-cue causal test."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

D = Path(__file__).resolve().parents[1] / "data/experiments"
OUT = D / "close_gaps"
rng = np.random.default_rng(0)


def load(p):
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {(r["pair_index"], r["side"]): (r["logit_margin"], r["cue_label"]) for r in rows}


def arrays(A, keys):
    return np.array([A[k][0] for k in keys]), np.array([A[k][1] for k in keys])


def gap(m, y):
    return float(m[y == 1].mean() - m[y == 0].mean())


def xfit(m, y, kp, pairs, n=100):
    out = []
    for _ in range(n):
        t = np.isin(kp, rng.choice(pairs, len(pairs) // 2, replace=False))
        best = max(np.unique(m[t]), key=lambda c: ((m[t] > c) == (y[t] == 1)).mean())
        out.append(((m[~t] > best) == (y[~t] == 1)).mean())
    return float(np.mean(out))


def summary(A, keys, kp, pairs, B=None, BA=None):
    m, y = arrays(A, keys)
    r = {"x": float(((m > 0) == (y == 1)).mean()), "dot": xfit(m, y, kp, pairs), "gap": gap(m, y), "p1": float((m > 0).mean())}
    if BA is not None:
        r["base_pres"] = float(np.mean([(BA[k][0] > 0) == (B[k][0] > 0) for k in keys]))
    return r


def setup(d):
    U = load(d / "unablated.jsonl")
    B = load(d / "base.jsonl") if (d / "base.jsonl").exists() else None
    keys = sorted(U); pairs = np.array(sorted({p for p, _ in keys}))
    return U, B, keys, np.array([k[0] for k in keys]), pairs


def layer_curves():
    res = {}
    for d in sorted(p for p in (D / "residuals/layer_check").iterdir() if (p / "unablated.jsonl").exists()):
        U, B, keys, kp, pairs = setup(d)
        L = sorted(int(f.stem.split("_L")[1]) for f in d.glob("own_L*.jsonl"))
        r = {"unablated": summary(U, keys, kp, pairs), "base": summary(B, keys, kp, pairs), "layers": L, "own": [], "random": []}
        for l in L:
            r["own"].append(summary(load(d / f"own_L{l}.jsonl"), keys, kp, pairs, B, load(d / f"base_own_L{l}.jsonl")))
            rs = [summary(load(f), keys, kp, pairs) for f in sorted(d.glob(f"random_L{l}_*.jsonl"))]
            r["random"].append({k: float(np.mean([x[k] for x in rs])) for k in ("x", "dot", "gap", "p1")})
        res[d.name] = r
        print(d.name, "unabl ●", round(r["unablated"]["dot"], 2), "base ●", round(r["base"]["dot"], 2))
    return res


SEEDS = {
    "s1": [("s1_seed0", "s1_flip"), ("s1_seed1", "s1_flip_seed1"), ("s1_seed2", "s1_flip_seed2")],
    "voice": [("voice_seed0", "voice_flip"), ("voice_seed2", "voice_flip_seed2")],
    "clause": [("clause_seed0", "clause_flip"), ("clause_seed1", "clause_flip_seed1"), ("clause_seed2", "clause_flip_seed2")],
    "lexical": [("lexical_e10", "lexical_flip_e10"), ("lexical_e10_seed1", "lexical_flip_e10_seed1"), ("lexical_e10_seed2", "lexical_flip_e10_seed2")],
}


def contrast(variant, rule, split):
    """(n_pairs, layers, d) adapter-minus-base cue contrast; first half of each split is cue-1 side."""
    G = torch.load(D / f"kernel_all/{variant}/activations_{split}.pt", weights_only=False)["activations"][rule]
    out = []
    for arm in ("adapter", "base"):
        h = G[arm][split]["h"].float(); n = len(h) // 2
        out.append(h[:n] - h[n:])
    return out[0] - out[1]


def signflip(s, n=5000):
    eps = rng.choice([-1.0, 1.0], size=(n, len(s)))
    return float((1 + (np.abs((eps * s).mean(1)) >= abs(s.mean())).sum()) / (n + 1))


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1, run)
    return adj


def split_parts():
    res, dirs = {}, {}
    for rule, seeds in SEEDS.items():
        res[rule] = []
        for normal, flip in seeds:
            fs = {s: (contrast(normal, rule, s), contrast(flip, rule, s)) for s in ("fit", "select", "eval")}
            xn = torch.cat([fs["fit"][0], fs["select"][0]]); xf = torch.cat([fs["fit"][1], fs["select"][1]])
            cdir, adir = F.normalize(((xn + xf) / 2).mean(0), dim=-1), F.normalize(((xn - xf) / 2).mean(0), dim=-1)
            en, ef = fs["eval"]; ce, ae = (en + ef) / 2, (en - ef) / 2
            cm, am = ce.mean(0), ae.mean(0)
            r = {"seed": normal, "answer_share": ((am.norm(dim=-1) ** 2) / (cm.norm(dim=-1) ** 2 + am.norm(dim=-1) ** 2)).tolist(),
                 "cue_d": [], "cue_p": [], "ans_d": [], "ans_p": []}
            for l in range(ce.shape[1]):
                for part, x, u in (("cue", ce, cdir), ("ans", ae, adir)):
                    s = (x[:, l] @ u[l]).numpy()
                    r[f"{part}_d"].append(float(s.mean() / s.std(ddof=1))); r[f"{part}_p"].append(signflip(s))
            res[rule].append(r); dirs.setdefault(rule, []).append(cdir)
            print(rule, normal, "answer share L0/L18/L35", [round(r["answer_share"][i], 2) for i in (0, 18, 35)])
    ps = np.array([p for rule in res for r in res[rule] for p in r["cue_p"]])
    adj = iter(holm(ps))
    for rule in res:
        for r in res[rule]:
            r["cue_p_holm"] = [float(next(adj)) for _ in r["cue_p"]]
    rules = list(SEEDS)
    cos = {"same_rule_across_seeds": {}, "cross_rule": {}}
    for a in rules:
        v = dirs[a]
        cos["same_rule_across_seeds"][a] = np.mean([(v[i] * v[j]).sum(-1).numpy() for i in range(len(v)) for j in range(i + 1, len(v))], 0).tolist()
    for i, a in enumerate(rules):
        for b in rules[i + 1:]:
            cos["cross_rule"][f"{a}-{b}"] = np.mean([(u * w).sum(-1).numpy() for u in dirs[a] for w in dirs[b]], 0).tolist()
    return res, cos


def shared_cue():
    res, tests = {}, []
    for d in sorted((D / "shared_cue").glob("*_L*")):
        if not (d / "unablated.jsonl").exists() or not (d / "shared_loo.jsonl").exists():
            continue
        U, B, keys, kp, pairs = setup(d)
        g0 = gap(*arrays(U, keys))
        rand = [gap(*arrays(load(f), keys)) for f in sorted(d.glob("random_*.jsonl"))]
        if len(rand) < 200:
            continue
        r = {"unablated": summary(U, keys, kp, pairs), "base": summary(B, keys, kp, pairs) if B else None, "random_gap": rand}
        for arm in ("shared_loo", "own_cue", "own_answer"):
            ba = d / f"base_{arm}.jsonl"
            r[arm] = summary(load(d / f"{arm}.jsonl"), keys, kp, pairs, B, load(ba) if ba.exists() and B else None)
            r[arm]["gap_drop"] = 1 - r[arm]["gap"] / g0
        r["p_shared_vs_random"] = float((1 + sum(g <= r["shared_loo"]["gap"] for g in rand)) / (len(rand) + 1))
        r["gap_drop_shared"] = 1 - r["shared_loo"]["gap"] / g0
        r["gap_drop_random_mean"] = 1 - float(np.mean(rand)) / g0
        res[d.name] = r; tests.append(d.name)
        print(d.name, "gap drop shared", round(r["gap_drop_shared"], 3), "random", round(r["gap_drop_random_mean"], 3), "p", round(r["p_shared_vs_random"], 4),
              "● U/shared/own", round(r["unablated"]["dot"], 2), round(r["shared_loo"]["dot"], 2), round(r["own_cue"]["dot"], 2), "drop own_cue/own_ans", round(r["own_cue"]["gap_drop"], 3), round(r["own_answer"]["gap_drop"], 3))
    for k, p in zip(tests, holm(np.array([res[k]["p_shared_vs_random"] for k in tests]))):
        res[k]["p_holm"] = float(p)
    return res


if __name__ == "__main__" and __import__("sys").argv[1:] == ["shared"]:
    (OUT / "shared_cue.json").write_text(json.dumps(shared_cue(), indent=1))
elif __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layer_check.json").write_text(json.dumps(layer_curves(), indent=1))
    parts, cos = split_parts()
    (OUT / "cue_split_all_layers.json").write_text(json.dumps({"parts": parts, "cosines": cos}, indent=1))
    (OUT / "shared_cue.json").write_text(json.dumps(shared_cue(), indent=1))
