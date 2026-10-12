"""What is on the "opposite side" of a rule direction? Normal vs label-flipped 3B adapters, layers 24-35.

Same pairs, same cue orientation (pos = cue says 1). Per pair i, x = adapter contrast minus base contrast.
  normal adapter:  answer follows the cue      -> x_n = cue part + answer part
  flipped adapter: answer is opposite the cue  -> x_f = cue part - answer part
so cue part c = (x_n + x_f) / 2 (kept when the answer flips) and answer part a = (x_n - x_f) / 2.
Directions are estimated on fit+select pairs; tests use held-out eval pairs only.
Squared sizes are unbiased (dot product of the two independent split means).
Primary family (Holm): held-out sign-flip test of c and a, for 2 rules x 12 layers.
"""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.kernel.l35 import RULES, answer_direction, contrast, holm, load, readout, remover, unit

ROOT = Path(__file__).resolve().parents[2] / "data/experiments"
OUT = ROOT / "kernel_exploration/opposite"
TARGETS = ["clause", "voice"]
LAYERS = list(range(24, 36))
N_BOOT, N_PERM = 2000, 10000


def acts(path, rule):
    G = torch.load(path, weights_only=False)
    assert G["layers"] == LAYERS
    return {arm: {s: G["activations"][rule][arm][s]["h"].float() for s in ("fit", "select", "eval")} for arm in ("base", "adapter")}


def pairs(d, split):
    """(n_pairs, 12, 2048) adapter-minus-base cue contrast."""
    out = []
    for arm in ("adapter", "base"):
        h = d[arm][split]; n = len(h) // 2
        out.append(h[:n] - h[n:])
    return out[0] - out[1]


def signflip_p(p, gen):
    s = torch.randint(0, 2, (N_PERM, len(p)), generator=gen).double() * 2 - 1
    null = (s * p).mean(1)
    return float((1 + (null.abs() >= abs(p.mean())).sum()) / (N_PERM + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0); rng = np.random.default_rng(0)
    res = {"layers": LAYERS, "rules": {}, "tests": []}
    parts = {}
    for rule in TARGETS:
        norm = acts(ROOT / f"residuals/scans/3b_seeds/{rule}_seed0_residual_scan_l24plus/activations.pt", rule)
        flip = acts(ROOT / f"kernel_exploration/flip/{rule}_flip/activations.pt", rule)
        for s in ("fit", "select", "eval"):  # same texts: base activations must be identical
            assert torch.equal(norm["base"][s], flip["base"][s]), (rule, s)
        xn = {k: torch.cat([pairs(norm, s) for s in v]) for k, v in (("fs", ("fit", "select")), ("ev", ("eval",)))}
        xf = {k: torch.cat([pairs(flip, s) for s in v]) for k, v in (("fs", ("fit", "select")), ("ev", ("eval",)))}
        c = {k: (xn[k] + xf[k]) / 2 for k in xn}; a = {k: (xn[k] - xf[k]) / 2 for k in xn}
        parts[rule] = {"normal": xn, "flipped": xf, "cue": c, "answer": a}
        R = {"n_pairs_fit_select": len(xn["fs"]), "n_pairs_eval": len(xn["ev"]), "by_layer": []}
        for li, L in enumerate(LAYERS):
            def sq(x, i=None, j=None):
                A = x["fs"][:, li] if i is None else x["fs"][i, li]; B = x["ev"][:, li] if j is None else x["ev"][j, li]
                return float(A.mean(0) @ B.mean(0))
            def stats(i=None, j=None):
                Ec, Ea = sq(c, i, j), sq(a, i, j)
                dn = unit((xn["fs"][:, li] if i is None else xn["fs"][i, li]).mean(0))
                df = unit((xf["fs"][:, li] if i is None else xf["fs"][i, li]).mean(0))
                return {"answer_share": Ea / (Ea + Ec), "cue_sq": Ec, "answer_sq": Ea, "cos_normal_flipped": float(dn @ df)}
            point = stats()
            boots = [stats(rng.integers(0, len(xn["fs"]), len(xn["fs"])), rng.integers(0, len(xn["ev"]), len(xn["ev"])))
                     for _ in range(N_BOOT)]
            row = {"layer": L, **point}
            for k in point:
                row[k + "_ci95"] = [float(np.percentile([b[k] for b in boots], q)) for q in (2.5, 97.5)]
            for name, x in (("cue", c), ("answer", a)):
                u = unit(x["fs"][:, li].mean(0))
                p = (x["ev"][:, li] @ u).double()
                t = {"rule": rule, "layer": L, "part": name, "mean_proj": float(p.mean()), "z": float(p.mean() / p.std()),
                     "frac_pairs_positive": float((p > 0).float().mean()), "p_two_sided": signflip_p(p, gen)}
                res["tests"].append(t); row[name + "_heldout_z"] = t["z"]
            R["by_layer"].append(row)
            print(rule, L, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)
        res["rules"][rule] = R
    for t, p in zip(res["tests"], holm([t["p_two_sided"] for t in res["tests"]])):
        t["p_holm"] = p

    # Layer 35 geometry: the original 3D plot basis (uncentred SVD of the four normal fit directions)
    data = load(); r = readout(); a_base, _ = answer_direction(data, r)
    li = LAYERS.index(35)
    four = {x: unit(contrast(data[(x, 0)], "fit").mean(0)) for x in RULES}
    V = torch.stack(list(four.values())); _, S, Vh = torch.linalg.svd(V, full_matrices=False); B = Vh[:3]
    old = json.loads((ROOT / "residuals/transfer/four_rule_3b_l35_lexical_e6/direction_geometry.json").read_text())
    assert np.allclose((V @ V.T).numpy(), np.array(old["cosines"]), atol=0.02), "must reproduce the original plot's cosines"
    vecs = dict(four)
    for rule in TARGETS:
        P = parts[rule]
        vecs[f"{rule} flipped"] = unit(P["flipped"]["fs"][:, li].mean(0))
        vecs[f"{rule} cue part"] = unit(P["cue"]["fs"][:, li].mean(0))
        vecs[f"{rule} answer part"] = unit(P["answer"]["fs"][:, li].mean(0))
    vecs["readout (1 minus 0)"] = r; vecs["base answer"] = a_base
    names = list(vecs); M = torch.stack([vecs[n] for n in names])
    res["l35_geometry"] = {"names": names, "cosines": (M @ M.T).tolist(), "pca_coords": (M @ B.T).tolist(),
                           "pca_norm_kept": (M @ B.T).norm(dim=1).tolist(), "readout_dot": (M @ r).tolist(),
                           "pca_basis": "uncentred SVD of the four normal seed-0 fit directions (as in directions_3d.html)"}
    # cue parts / answer parts shared across rules? held-out cosine: clause fit+select vs voice eval and vice versa
    cross = {}
    for name in ("cue", "answer", "normal"):
        per_layer = []
        for li2, L in enumerate(LAYERS):
            f = lambda X, Y: float(unit(X["fs"][:, li2].mean(0)) @ unit(Y["ev"][:, li2].mean(0)))
            Pc, Pv = parts["clause"][name], parts["voice"][name]
            per_layer.append({"layer": L, "cos_clause_fs_voice_ev": f(Pc, Pv), "cos_voice_fs_clause_ev": f(Pv, Pc),
                              "self_clause": f(Pc, Pc), "self_voice": f(Pv, Pv)})
        cross[name] = per_layer
    res["cross_rule"] = cross
    # Is the cue part shared across rules beyond any leftover answer signal? Remove the answer parts of both
    # rules (fit+select), the readout and the base-answer direction from everything, then project the other
    # rule's held-out eval cue parts on this rule's fit+select cue direction. Secondary family (Holm).
    res["cross_cue_tests"] = []
    for li2, L in enumerate(LAYERS):
        ans = [unit(parts[x]["answer"]["fs"][:, li2].mean(0)) for x in TARGETS]
        P = remover(*ans, r, a_base) if L == 35 else remover(*ans)
        for src, tgt in (("clause", "voice"), ("voice", "clause")):
            for name in ("cue", "answer"):
                u = unit(P(parts[src][name]["fs"][:, li2].mean(0)) if name == "cue" else parts[src][name]["fs"][:, li2].mean(0))
                pr = ((P(parts[tgt][name]["ev"][:, li2]) if name == "cue" else parts[tgt][name]["ev"][:, li2]) @ u).double()
                res["cross_cue_tests"].append({"layer": L, "source": src, "target": tgt, "part": name,
                    "answer_dirs_removed": name == "cue", "d": float(pr.mean() / pr.std()),
                    "frac_pairs_positive": float((pr > 0).float().mean()), "p_two_sided": signflip_p(pr, gen)})
    fam = [t for t in res["cross_cue_tests"] if t["part"] == "cue"]
    for t, p in zip(fam, holm([t["p_two_sided"] for t in fam])):
        t["p_holm"] = p
    for t in fam:
        print("cross cue", t["layer"], t["source"], "->", t["target"], round(t["d"], 2), round(t["frac_pairs_positive"], 2), t["p_holm"])
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    sig = [t for t in res["tests"] if t["p_holm"] < .05]
    print("Holm-significant:", [(t["rule"], t["layer"], t["part"], round(t["z"], 2)) for t in sig])
    print(json.dumps({"names": names, "readout_dot": [round(x, 3) for x in res["l35_geometry"]["readout_dot"]],
                      "pca_norm_kept": [round(x, 3) for x in res["l35_geometry"]["pca_norm_kept"]]}, indent=0))
    for name in cross:
        print(name, [(d["layer"], round(d["cos_clause_fs_voice_ev"], 2), round(d["self_clause"], 2), round(d["self_voice"], 2)) for d in cross[name]])


if __name__ == "__main__":
    main()
