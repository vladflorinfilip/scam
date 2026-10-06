"""Layer-35 (3B) transfer, common direction and kernel probes, computed exactly on CPU.

Units are held-out eval pairs. Directions are fitted on fit pairs only (probes: fit+select).
Every ablation result is compared with readout-matched random directions, because at the last
layer an ablation can only move the answer through the '1'-minus-'0' readout direction r.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from experiments.kernel.readout import ablate, margin, weights

ROOT = Path(__file__).resolve().parents[2] / "data/experiments"
RES = ROOT / "residuals"
OUT = ROOT / "kernel_exploration/l35"
RULES = ["s1", "voice", "clause", "lexical"]
SETS = {0: {r: 0 for r in RULES}, 2: {r: 2 for r in RULES}}  # adapter training seed per rule
N_NULL, N_BOOT, N_PERM = 1000, 2000, 1000


def unit(v):
    return v / v.norm()


def load():
    four = torch.load(RES / "transfer/four_rule_3b_l35_lexical_e6/activations.pt", weights_only=False)["activations"]
    pick = lambda A, li: {arm: {s: A[arm][s]["h"][:, li].float() for s in A[arm]} for arm in A}
    data = {(r, 0): pick(four[r], 0) for r in RULES}
    for r in RULES:
        name = "lexical_e6" if r == "lexical" else r
        for s in (0, 1, 2):
            p = RES / f"scans/3b_seeds/{name}_seed{s}_residual_scan_l24plus/activations.pt"
            if not p.exists():
                continue
            G = torch.load(p, weights_only=False)
            d = pick(next(iter(G["activations"].values())), G["layers"].index(35))
            if s == 0:  # same adapter as the four-rule run: must agree up to GPU batch noise
                diff = (d["adapter"]["eval"] - data[(r, 0)]["adapter"]["eval"]).abs().max().item()
                assert diff < 1.0, (r, diff)
                continue
            data[(r, s)] = d
    return data


def contrast(d, split, arm=None):
    """Per-pair cue-1 minus cue-0; adapter change relative to base unless arm is given."""
    if arm:
        h = d[arm][split]; n = len(h) // 2
        return h[:n] - h[n:]
    return contrast(d, split, "adapter") - contrast(d, split, "base")


def readout():
    rows, norm, _ = weights()
    return unit(norm.float() * (rows[1] - rows[0]).float())


def answer_direction(data, r):
    """Base-model answer direction (no adapter, no cue), pooled within rule; orthogonal to r."""
    acc = 0
    for rule in RULES:
        h = torch.cat([data[(rule, 0)]["base"][s] for s in ("fit", "select")])
        m = margin(h)
        acc = acc + unit(h[m > 0].mean(0) - h[m <= 0].mean(0))
    return unit(acc), unit(acc - (acc @ r) * r)


def remover(*vs):
    U = torch.linalg.qr(torch.stack(vs, 1)).Q
    return lambda x: x - (x @ U) @ U.T


def xfit(m1, m0, rng, n=300):
    out, P = [], len(m1)
    for _ in range(n):
        t = np.zeros(P, bool); t[rng.choice(P, P // 2, replace=False)] = True
        c = np.unique(np.r_[m1[t], m0[t]])
        acc = t.sum() - np.searchsorted(np.sort(m1[t]), c, side="right") + np.searchsorted(np.sort(m0[t]), c, side="right")
        best = c[np.argmax(acc)]
        out.append(((m1[~t] > best).mean() + (m0[~t] <= best).mean()) / 2)
    return float(np.mean(out))


def evaluate(d, q, rng=None):
    """Ablate unit direction q from adapter (centre: adapter fit mean) and base (centre: base fit mean)."""
    A, B = d["adapter"]["eval"], d["base"]["eval"]
    n = len(A) // 2
    m0, b0 = margin(A).numpy(), margin(B).numpy()
    if q is None:
        m, mb = m0, b0
    else:
        Q = q[:, None]
        m = margin(ablate(A, Q, d["adapter"]["fit"].mean(0))).numpy()
        mb = margin(ablate(B, Q, d["base"]["fit"].mean(0))).numpy()
    g = lambda x: x[:n] - x[n:]
    out = {"gap_unablated": float(g(m0).mean()), "gap_base": float(g(b0).mean()), "gap": float(g(m).mean()),
           "follow_at_0": float(((m[:n] > 0).mean() + (m[n:] <= 0).mean()) / 2),
           "pair_order": float((g(m) > 0).mean()), "base_kept": float(((mb > 0) == (b0 > 0)).mean())}
    out["remaining"] = (out["gap"] - out["gap_base"]) / (out["gap_unablated"] - out["gap_base"])
    if rng is not None:
        out["follow_xfit"] = xfit(m[:n], m[n:], rng)
        out["base_follow_xfit"] = xfit(b0[:n], b0[n:], rng)
        dg, bg = g(m) - g(b0), g(m0) - g(b0)
        idx = rng.integers(n, size=(N_BOOT, n))
        boots = dg[idx].mean(1) / bg[idx].mean(1)
        out["remaining_ci"] = [float(x) for x in np.percentile(boots, [2.5, 97.5])]
        kept = (mb > 0) == (b0 > 0)
        out["base_kept_ci"] = [float(x) for x in np.percentile((kept[idx] + kept[idx + n]).mean(1) / 2, [2.5, 97.5])]
    return out


def match_r(q, z, r):
    """Unit direction along z (orthogonalised to r) with the same signed cosine to r as q."""
    z = unit(z - (z @ r) * r); c = float(q @ r)
    return c * r + (1 - c * c) ** .5 * z


def signflip_mean(X, gen):
    s = torch.randint(0, 2, (len(X), 1), generator=gen).float() * 2 - 1
    return unit((s * X).mean(0))


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj.tolist()


def transfer(data, seedset, r, a_raw, rng):
    keys = {rule: (rule, s) for rule, s in SETS[seedset].items()}
    X = {rule: contrast(data[k], "fit") for rule, k in keys.items()}
    dirs = {rule: unit(x.mean(0)) for rule, x in X.items()}
    rows, gen = [], torch.Generator().manual_seed(seedset)
    for target, k in keys.items():
        d = data[k]
        others = [a for a in RULES if a != target]
        # arm -> (direction, sign-flip null generator or None)
        arms = {f"dir:{a}": (dirs[a], lambda a=a: signflip_mean(X[a], gen)) for a in RULES}
        arms["common_loo"] = (unit(sum(dirs[a] for a in others)), lambda: unit(sum(signflip_mean(X[a], gen) for a in others)))
        arms["base_answer"] = (a_raw, None)
        arms["readout_r"] = (r, None)
        arms[f"dir:{target}_perp_r"] = (unit(dirs[target] - (dirs[target] @ r) * r), None)
        arms["none"] = (None, None)
        for name, (q, flip) in arms.items():
            res = evaluate(d, q, rng)
            res.update(target=target, seed=k[1], arm=name)
            if q is None or name == "readout_r":
                rows.append(res); continue
            res["cos_r"] = float(q @ r)
            nulls = {"iso": lambda: torch.randn(len(r), generator=gen)}
            if flip is not None:
                nulls["signflip"] = flip
            for kind, draw in nulls.items():
                null = np.array([evaluate(d, match_r(q, draw(), r))["remaining"] for _ in range(N_NULL)])
                res[f"null_{kind}_mean"] = float(null.mean())
                res[f"null_{kind}_ci"] = [float(x) for x in np.percentile(null, [2.5, 97.5])]
                res[f"p_{kind}"] = float((1 + (null <= res["remaining"]).sum()) / (N_NULL + 1))
            rows.append(res)
    tests = [i for i, x in enumerate(rows) if "p_signflip" in x]
    for i, p in zip(tests, holm([rows[i]["p_signflip"] for i in tests])):
        rows[i]["p_signflip_holm"] = p
    return rows


def geometry(data, r, a, rng):
    keys = sorted(data)
    D = {k: contrast(data[k], "fit") for k in keys}
    dirs = {k: unit(v.mean(0)) for k, v in D.items()}
    P = remover(r, a)
    out = {"keys": [f"{k[0]}_seed{k[1]}" for k in keys], "cos": [], "cos_perp_ra": [], "cos_r": [], "cos_a": []}
    for k in keys:
        out["cos"].append([float(dirs[k] @ dirs[j]) for j in keys])
        out["cos_perp_ra"].append([float(unit(P(dirs[k])) @ unit(P(dirs[j]))) for j in keys])
        out["cos_r"].append(float(dirs[k] @ r)); out["cos_a"].append(float(dirs[k] @ a))
    # Cross-rule alignment vs a sign-flip null: random cue signs per fit pair keep the residual stream's
    # anisotropy but destroy any learned cue direction. One-sided (is alignment larger than chance?).
    gen = torch.Generator().manual_seed(1)
    out["tests"] = []
    for seedset, sel in SETS.items():
        ks = [(rule, s) for rule, s in sel.items()]
        for i, k in enumerate(ks):
            for j in ks[i + 1:]:
                for space, f in (("raw", lambda v: v), ("perp_r_a", P)):
                    obs = float(unit(f(dirs[k])) @ unit(f(dirs[j])))
                    null = np.array([float(unit(f(signflip_mean(D[k], gen))) @ unit(f(signflip_mean(D[j], gen))))
                                     for _ in range(1000)])
                    out["tests"].append(dict(seedset=seedset, a=k[0], b=j[0], space=space, cos=obs,
                                             null95=float(np.percentile(null, 95)), p=float((1 + (null >= obs).sum()) / 1001)))
    for space in ("raw", "perp_r_a"):
        idx = [i for i, t in enumerate(out["tests"]) if t["space"] == space]
        for i, p in zip(idx, holm([out["tests"][i]["p"] for i in idx])):
            out["tests"][i]["p_holm"] = p
    return out


def answer_contrasts(data, rule, gen):
    """Base-model answer-1 minus answer-0 differences (random pairing within rule; no adapter, no cue labels)."""
    h = torch.cat([data[(rule, 0)]["base"][s] for s in ("fit", "select")]); m = margin(h)
    one, zero = h[m > 0], h[m <= 0]; n = min(len(one), len(zero))
    return one[torch.randperm(len(one), generator=gen)[:n]] - zero[torch.randperm(len(zero), generator=gen)[:n]]


def probe_fns(model, X, T):
    """Kernel ridge probe trained on +X (target +1) / -X (target -1); held-out pairs scored by f(c) - f(-c).

    Flipping the sign of a training contrast only flips the targets of +x_i and -x_i, so the training set,
    the RBF bandwidth (median heuristic) and the kernel matrices are unchanged under the sign-flip null, and
    the fitted scores are linear in the targets: the whole null is one matrix product. Ridge is fixed a
    priori at 0.1 x mean kernel diagonal.
    """
    A = torch.from_numpy(np.r_[X, -X]).double(); Tt = torch.from_numpy(T).double()
    if model == "linear":
        k = lambda u, v: u @ v.T
    else:
        D = torch.cdist(A, A).square()
        gamma = 1 / D[tuple(torch.triu_indices(len(A), len(A), 1))].median().item()
        k = lambda u, v: torch.exp(-gamma * torch.cdist(u, v).square())
    K = k(A, A)
    M = (k(Tt, A) - k(-Tt, A)) @ torch.linalg.inv(K + 0.1 * K.diagonal().mean() * torch.eye(len(A), dtype=K.dtype))

    def run(signs):  # signs: (n,) or (n, B) of +-1
        S = torch.as_tensor(signs, dtype=torch.float64).reshape(len(X), -1)
        sc = M @ torch.cat([S, -S])
        return (sc > 0).double().mean(0).numpy(), (sc.mean(0) / sc.std(0)).numpy()
    return run


def probes(data, seedset, r, a, rng):
    keys = {rule: (rule, s) for rule, s in SETS[seedset].items()}
    P = remover(r, a)
    gen = torch.Generator().manual_seed(10 + seedset)
    C = {rule: {"train": torch.cat([contrast(data[k], s) for s in ("fit", "select")]), "test": contrast(data[k], "eval"),
                "answer": answer_contrasts(data, rule, gen)} for rule, k in keys.items()}
    rows = []
    for space, f in (("raw", lambda x: x), ("perp_r_a", P)):
        for target in RULES:
            T = f(C[target]["test"]).numpy()
            for train in ("loo", "within", "answer_loo"):
                src = [target] if train == "within" else [x for x in RULES if x != target]
                X = f(torch.cat([C[x]["answer" if train == "answer_loo" else "train"] for x in src]))
                for model in ("linear", "rbf"):
                    run = probe_fns(model, X.numpy(), T)
                    acc, z = (float(v[0]) for v in run(np.ones(len(X))))
                    row = dict(seedset=seedset, space=space, target=target, train=train, model=model, n=len(T), acc=acc, z=z)
                    if train != "within":
                        nacc, null = run(torch.randint(0, 2, (len(X), N_PERM), generator=gen).numpy() * 2 - 1)
                        row.update(null_z95=float(np.percentile(null, 95)), null_acc95=float(np.percentile(nacc, 95)),
                                   p=float((1 + (null >= z).sum()) / (N_PERM + 1)))
                    rows.append(row)
                    print(seedset, space, target, train, model, round(acc, 3), round(z, 2), row.get("p"), flush=True)
    # primary family: does a probe trained on the other three rules read the held-out rule's cue once the
    # readout and base-answer directions are projected out? Other rows are secondary (uncorrected p).
    tests = [i for i, x in enumerate(rows) if x["space"] == "perp_r_a" and x["train"] == "loo"]
    for i, p in zip(tests, holm([rows[i]["p"] for i in tests])):
        rows[i]["p_holm"] = p
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = load()
    r = readout(); a_raw, a = answer_direction(data, r)
    res = {"readout": {"cos_r_answer": float(a_raw @ r)}}
    if "--resume" in sys.argv:  # keep finished steps from a previous run of this same code
        res = {**res, **json.loads((OUT / "results.json").read_text())}
    steps = {"geometry": lambda: geometry(data, r, a, rng),
             "transfer": lambda: transfer(data, 0, r, a_raw, rng) + transfer(data, 2, r, a_raw, rng),
             "probes": lambda: probes(data, 0, r, a, rng) + probes(data, 2, r, a, rng)}
    for name, step in steps.items():
        if name in res:
            continue
        res[name] = step(); print("done", name, flush=True)
        (OUT / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
