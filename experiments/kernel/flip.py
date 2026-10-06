"""Label-flip control at 3B layer 35: is the shared cross-rule probe signal the hidden cue or the adapter's answer?

A flipped adapter is trained on the same rows with final_answer = 1 - cue label. Pairs keep the cue
orientation (pos = cue says 1), so its contrast cue1 - cue0 carries the cue with the *opposite* answer.
Leave-one-rule-out probes are trained on the other three normal (seed-0) adapters, exactly as in l35.py,
and scored on the flipped adapter's held-out contrasts:
  cue code    -> score z > 0 (same sign as on the normal adapter)
  answer code -> score z < 0 (reversed)
Primary family (Holm, two-sided sign-flip p): perp_r_a space x {linear, rbf} x {clause, voice}.
"""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.kernel.l35 import (N_PERM, RULES, answer_direction, contrast, holm, load, probe_fns,
                                    readout, remover, unit)
from experiments.kernel.readout import margin

OUT = Path(__file__).resolve().parents[2] / "data/experiments/kernel_exploration/flip"
TARGETS = ["clause", "voice"]
N_BOOT = 2000


def load_flip(rule):
    G = torch.load(OUT / f"{rule}_flip/activations.pt", weights_only=False)
    li = G["layers"].index(35)
    A = G["activations"][rule]
    return {arm: {s: A[arm][s]["h"][:, li].float() for s in A[arm]} for arm in A}


def pair_scores(model, X, T):
    """Per-test-pair scores f(c) - f(-c) of the l35 kernel ridge probe (same kernel, bandwidth, ridge)."""
    A = torch.from_numpy(np.r_[X, -X]).double(); Tt = torch.from_numpy(T).double()
    if model == "linear":
        k = lambda u, v: u @ v.T
    else:
        D = torch.cdist(A, A).square()
        g = 1 / D[tuple(torch.triu_indices(len(A), len(A), 1))].median().item()
        k = lambda u, v: torch.exp(-g * torch.cdist(u, v).square())
    K = k(A, A)
    y = torch.cat([torch.ones(len(X)), -torch.ones(len(X))]).double()
    w = torch.linalg.solve(K + 0.1 * K.diagonal().mean() * torch.eye(len(A), dtype=K.dtype), y)
    return ((k(Tt, A) - k(-Tt, A)) @ w).numpy()


def follow(m):
    n = len(m) // 2
    return float(((m[:n] > 0).float().mean() + (m[n:] <= 0).float().mean()) / 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    r = readout(); _, a = answer_direction(data, r)
    P = remover(r, a)
    gen = torch.Generator().manual_seed(123)
    rng = np.random.default_rng(0)
    res = {"behaviour": {}, "geometry": {}, "probes": []}
    for target in TARGETS:
        flip, norm = load_flip(target), data[(target, 0)]
        res["behaviour"][target] = {
            "base_max_abs_diff_vs_l35": float((flip["base"]["eval"] - norm["base"]["eval"]).abs().max()),
            "follow_cue_normal": follow(margin(norm["adapter"]["eval"])),
            "follow_cue_flipped": follow(margin(flip["adapter"]["eval"])),
            "follow_cue_base": follow(margin(flip["base"]["eval"])),
            "readout_gap_normal": float((contrast(norm, "eval") @ r).mean()),
            "readout_gap_flipped": float((contrast(flip, "eval") @ r).mean()),
        }
        dn, df = (unit(torch.cat([contrast(d, s) for s in ("fit", "select")]).mean(0)) for d in (norm, flip))
        res["geometry"][target] = {"cos_raw": float(dn @ df), "cos_perp_r_a": float(unit(P(dn)) @ unit(P(df))),
                                   "normal_dot_r": float(dn @ r), "flipped_dot_r": float(df @ r)}
        src = [x for x in RULES if x != target]
        for space, f in (("raw", lambda x: x), ("perp_r_a", P)):
            X = f(torch.cat([contrast(data[(x, 0)], s) for x in src for s in ("fit", "select")])).numpy()
            for model in ("linear", "rbf"):
                for adapter, d in (("normal", norm), ("flipped", flip)):
                    T = f(contrast(d, "eval")).numpy()
                    sc = pair_scores(model, X, T)
                    z = float(sc.mean() / sc.std(ddof=1))
                    boot = [(lambda s: s.mean() / s.std(ddof=1))(sc[rng.integers(0, len(sc), len(sc))]) for _ in range(N_BOOT)]
                    row = dict(target=target, adapter=adapter, space=space, model=model, n=len(T), z=z,
                               z_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                               frac_pairs_positive=float((sc > 0).mean()))
                    run = probe_fns(model, X, T)
                    assert abs(float(run(np.ones(len(X)))[1][0]) - z) < 1e-4 * max(1, abs(z)), "pair_scores must match probe_fns"
                    if adapter == "flipped":
                        _, null = run(torch.randint(0, 2, (len(X), N_PERM), generator=gen).numpy() * 2 - 1)
                        row.update(null_abs_z95=float(np.percentile(np.abs(null), 95)),
                                   p_two_sided=float((1 + (np.abs(null) >= abs(z)).sum()) / (N_PERM + 1)))
                    res["probes"].append(row)
                    print(target, adapter, space, model, round(z, 2), row.get("p_two_sided"), flush=True)
    tests = [i for i, x in enumerate(res["probes"]) if x["adapter"] == "flipped" and x["space"] == "perp_r_a"]
    for i, p in zip(tests, holm([res["probes"][i]["p_two_sided"] for i in tests])):
        res["probes"][i]["p_holm"] = p
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("behaviour", "geometry")}, indent=1))


if __name__ == "__main__":
    main()
