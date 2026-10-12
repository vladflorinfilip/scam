"""Are the hidden rules correlated once the 1-0 answer is removed? Qwen 3B, seed 0, layers 24-35.

For each rule with a valid label-flipped adapter (adapter follows the original cue <= 10% on eval):
cue part = (normal + flipped) / 2, answer part = (normal - flipped) / 2 of the adapter-minus-base pair contrast.
Per layer and part: held-out cross-rule cosine cos(mean fit+select of rule A, mean eval of rule B), the
same-rule held-out cosine (ceiling) and a sign-flip noise floor. Primary test: the held-out cue transfer
between every ordered pair of rules after removing all answer parts (+ readout and base answer at L35),
sign-flip p-values, Holm over pairs x layers.
"""
import itertools
import json

import numpy as np
import torch

from experiments.kernel.l35 import answer_direction, holm, load, readout, remover, unit
from experiments.kernel.opposite import LAYERS, ROOT, acts, pairs, signflip_p

OUT = ROOT / "cue_decomposition"
NORMAL = {"s1": "s1_seed0", "voice": "voice_seed0", "clause": "clause_seed0", "lexical": "lexical_e6_seed0"}
FLIPS = {"s1": ["s1_flip"], "voice": ["voice_flip"], "clause": ["clause_flip"], "lexical": ["lexical_flip_e10", "lexical_flip"]}
N_NULL = 1000


def flip_follow(v):
    return json.loads((ROOT / f"kernel_exploration/flip/{v}/summary.json").read_text())["adapter"]["eval"]["follow_original_cue"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    parts, used, skipped = {}, {}, {}
    for rule in NORMAL:
        ok = [v for v in FLIPS[rule] if (ROOT / f"kernel_exploration/flip/{v}/activations.pt").exists()]
        good = [v for v in ok if flip_follow(v) <= 0.10]
        if not good:
            skipped[rule] = {v: flip_follow(v) for v in ok}; continue
        used[rule] = {"flip": good[0], "flip_eval_follow": flip_follow(good[0])}
        norm = acts(ROOT / f"residuals/scans/3b_seeds/{NORMAL[rule]}_residual_scan_l24plus/activations.pt", rule)
        flip = acts(ROOT / f"kernel_exploration/flip/{good[0]}/activations.pt", rule)
        for s in ("fit", "select", "eval"):
            assert torch.equal(norm["base"][s], flip["base"][s]), (rule, s)
        sp = {"fs": ("fit", "select"), "ev": ("eval",)}
        xn = {k: torch.cat([pairs(norm, s) for s in v]) for k, v in sp.items()}
        xf = {k: torch.cat([pairs(flip, s) for s in v]) for k, v in sp.items()}
        parts[rule] = {"normal": xn, "cue": {k: (xn[k] + xf[k]) / 2 for k in xn}, "answer": {k: (xn[k] - xf[k]) / 2 for k in xn}}
    rules = list(parts)
    print("rules used", used, "skipped", skipped)
    r = readout(); a_base, _ = answer_direction(load(), r)
    res = {"rules": rules, "used": used, "skipped": skipped, "layers": LAYERS, "geometry": [], "tests": []}
    for li, L in enumerate(LAYERS):
        row = {"layer": L}
        for name in ("normal", "cue", "answer"):
            P = {q: parts[q][name] for q in rules}
            m = {q: {k: P[q][k][:, li].mean(0) for k in ("fs", "ev")} for q in rules}
            self_ = {q: float(unit(m[q]["fs"]) @ unit(m[q]["ev"])) for q in rules}
            cross = {}
            for a, b in itertools.combinations(rules, 2):
                c = (float(unit(m[a]["fs"]) @ unit(m[b]["ev"])) + float(unit(m[b]["fs"]) @ unit(m[a]["ev"]))) / 2
                null = []
                for _ in range(N_NULL // 10):
                    sa = (torch.randint(0, 2, (len(P[a]["fs"]), 1), generator=gen) * 2 - 1).float()
                    sb = (torch.randint(0, 2, (len(P[b]["ev"]), 1), generator=gen) * 2 - 1).float()
                    null.append(float(unit((P[a]["fs"][:, li] * sa).mean(0)) @ unit((P[b]["ev"][:, li] * sb).mean(0))))
                cross[f"{a}-{b}"] = {"cos_heldout": c, "noise95": float(np.percentile(np.abs(null), 95)),
                                     "disattenuated": c / np.sqrt(max(self_[a], 1e-6) * max(self_[b], 1e-6))}
            row[name] = {"self_heldout": self_, "cross": cross}
        res["geometry"].append(row)
        ans = [unit(parts[q]["answer"]["fs"][:, li].mean(0)) for q in rules]
        Rm = remover(*ans, r, a_base) if L == 35 else remover(*ans)
        for src, tgt in itertools.permutations(rules, 2):
            u = unit(Rm(parts[src]["cue"]["fs"][:, li].mean(0)))
            pr = (Rm(parts[tgt]["cue"]["ev"][:, li]) @ u).double()
            res["tests"].append({"layer": L, "source": src, "target": tgt, "d": float(pr.mean() / pr.std()),
                                 "frac_pairs_positive": float((pr > 0).float().mean()), "p_two_sided": signflip_p(pr, gen)})
        print(L, {n: {k: round(v["cos_heldout"], 2) for k, v in row[n]["cross"].items()} for n in ("normal", "cue", "answer")}, flush=True)
    for t, p in zip(res["tests"], holm([t["p_two_sided"] for t in res["tests"]])):
        t["p_holm"] = p
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    for t in res["tests"]:
        print("cue transfer", t["layer"], t["source"], "->", t["target"], round(t["d"], 2), round(t["frac_pairs_positive"], 2), round(t["p_holm"], 4))


if __name__ == "__main__":
    main()
