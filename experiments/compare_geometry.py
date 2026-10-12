"""Rule-direction geometry across Qwen2.5-0.5B, Qwen2.5-3B and gpt-oss-20b (seed 0, base-subtracted cue contrasts).

Direction = unit mean over fit pairs of (adapter cue1-cue0) - (base cue1-cue0), as in experiments/residual.py.
Vectors from different models live in different spaces, so models are compared through within-model
cosines (by relative depth = layer / (n_layers - 1)) and each model's own uncentred-SVD 3D projection.
Noise: split-half reliability cos(fit mean, eval mean); null for mean cross-rule cosine = per-pair sign flips.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

D = Path(__file__).resolve().parents[1] / "data/experiments"
OUT = D / "model_comparison"
RULES = ["s1", "voice", "clause", "lexical"]
N_NULL = 200


def contrasts(adapter, base):
    out = {}
    for s in ("fit", "eval"):
        a, b = adapter[s]["h"].float(), base[s]["h"].float()
        n = len(a) // 2
        out[s] = (a[:n] - a[n:]) - (b[:n] - b[n:])  # (pairs, layers, d)
    return out


def load_gptoss():
    X = {}
    for r in RULES:
        p = D / f"gptoss20b/scans/full/{r}"
        if (p / "activations.pt").exists():
            A = torch.load(p / "activations.pt", weights_only=False)["activations"][r]
        else:  # git copy: lossless bf16 shards over layers 0-11 and 12-23
            S = [torch.load(p / f"activations_bf16_{k}.pt", weights_only=False)["activations"][r] for k in ("l00-11", "l12-23")]
            A = {arm: {s: {"h": torch.cat([x[arm][s]["h"] for x in S], 1)} for s in ("fit", "eval")} for arm in ("adapter", "base")}
        X[r] = contrasts(A["adapter"], A["base"])
    return X


def load_3b():
    X = {}
    for r in RULES:
        name = "lexical_e6" if r == "lexical" else r
        p = D / f"residuals/scans/3b_full/{name}_residual_scan_3b_full"
        arms = {}
        for arm in ("adapter", "base"):
            G = torch.load(p / f"activations_{arm}_fp16.pt", weights_only=False)
            assert G["layers"] == list(range(36)), G["layers"]
            A = G["activations"]; A = A[next(iter(A))] if set(A) - {"fit", "select", "eval"} else A
            A = A[arm] if arm in A else A
            arms[arm] = A
        X[r] = contrasts(arms["adapter"], arms["base"])
    return X


def load_05b():
    """No activations saved for 0.5B: L18 frozen directions (s1, voice, clause) and the L20-22 selected directions."""
    z = np.load(D / "residuals/ablations/l18/third_rule_l18/direction_geometry.npz")
    names = json.loads((D / "residuals/ablations/l18/third_rule_l18/direction_geometry.json").read_text())["names"]
    l18 = {k: torch.tensor(v).float() for k, v in zip(names, z["vectors"]) if k in ("S1", "Voice", "Clause order")}
    l18 = dict(zip(["s1", "voice", "clause"], [F.normalize(l18[k], dim=0) for k in ("S1", "Voice", "Clause order")]))
    sel, layer = {}, {}
    for r in RULES:
        d = torch.load(D / f"residuals/scans/late_l20plus/{r}_residual_scan_l20plus/selected_subspace.pt", weights_only=False)
        sel[r] = F.normalize(d["basis"][:, 0].float(), dim=0); layer[r] = int(d["layer"])
    return l18, sel, layer


def cosmat(V):
    M = torch.stack([F.normalize(v, dim=0) for v in V])
    return M @ M.T


def mean_offdiag(C):
    n = len(C); return float((C.sum() - C.diagonal().sum()) / (n * n - n))


def per_layer(X, gen):
    nl = X[RULES[0]]["fit"].shape[1]; rows = []
    for L in range(nl):
        V = [X[r]["fit"][:, L].mean(0) for r in RULES]
        C = cosmat(V)
        null = []
        for _ in range(N_NULL):
            Vn = [(X[r]["fit"][:, L] * (torch.randint(0, 2, (len(X[r]["fit"]), 1), generator=gen) * 2 - 1)).mean(0) for r in RULES]
            null.append(mean_offdiag(cosmat(Vn)))
        m = mean_offdiag(C)
        rows.append({"layer": L, "depth": L / (nl - 1), "cos": C.tolist(), "mean_cross_cos": m,
                     "null_abs95": float(np.percentile(np.abs(null), 95)),
                     "p_two_sided": float((1 + (np.abs(null) >= abs(m)).sum()) / (N_NULL + 1)),
                     "reliability": {r: float(F.cosine_similarity(X[r]["fit"][:, L].mean(0), X[r]["eval"][:, L].mean(0), dim=0)) for r in RULES}})
    return rows


def svd3(V):
    M = torch.stack([F.normalize(v, dim=0) for v in V]); _, s, vh = torch.linalg.svd(M, full_matrices=False)
    xyz = M @ vh[:3].T
    flip = torch.sign(xyz[:, 0].sum()) or 1.0  # SVD sign is arbitrary: orient PC1 so the arrows' sum is positive
    xyz[:, 0] *= flip
    return xyz.tolist(), float(s[:3].square().sum() / s.square().sum())


def plot(res):
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    col = {"s1": "#2ca02c", "voice": "#ff7f0e", "clause": "#1f77b4", "lexical": "#9467bd"}
    fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(2, 5, hspace=.45, wspace=.3, top=.95, bottom=.06, left=.05, right=.98)
    a = fig.add_subplot(gs[0, :3])
    for m, c, lab in (("qwen3b", "tab:red", "Qwen 3B"), ("gptoss20b", "tab:blue", "gpt-oss-20b")):
        R = res[m]; x = [r["depth"] for r in R]
        a.plot(x, [r["mean_cross_cos"] for r in R], "o-", color=c, ms=3, label=f"{lab}: mean cosine between rule directions")
        a.fill_between(x, [-r["null_abs95"] for r in R], [r["null_abs95"] for r in R], color=c, alpha=.12)
    C = np.array(res["qwen05b"]["l18"]["cos"]); a.plot([18 / 23], [C[np.triu_indices(3, 1)].mean()], "k*", ms=12,
                                                      label="Qwen 0.5B L18 (s1, voice, clause only)")
    a.axhline(0, color="grey", lw=.5)
    a.set(xlabel="relative depth (layer / last layer)", ylabel="mean cosine across the 4 rules",
          title="a  Do the rule arrows point the same way? (shaded: 95% noise band from sign-flipped pairs)")
    a.legend(fontsize=8, loc="upper left")
    a = fig.add_subplot(gs[0, 3:])
    for m, c, lab in (("qwen3b", "tab:red", "3B"), ("gptoss20b", "tab:blue", "gpt-oss")):
        for r in RULES:
            a.plot([x["depth"] for x in res[m]], [x["reliability"][r] for x in res[m]], color=c, alpha=.6, lw=1)
        a.plot([], [], color=c, label=lab)
    a.set(xlabel="relative depth", ylabel="cos(fit-pair arrow, held-out arrow)", ylim=(0, 1.02),
          title="b  Each arrow is stable on held-out pairs"); a.legend(fontsize=8)
    f3 = make_subplots(rows=1, cols=len(res["pca_views"]), specs=[[{"type": "scene"}] * len(res["pca_views"])],
                       subplot_titles=list(res["pca_views"]))
    for i, (k, v) in enumerate(res["pca_views"].items()):
        rules = res["qwen05b"]["l18"]["rules"] if k.startswith("Qwen 0.5B") else RULES
        a = fig.add_subplot(gs[1, i]); xyz = np.array(v["coords"])
        for r, (x, y, z) in zip(rules, xyz):
            a.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="->", color=col[r], lw=2)); a.text(x, y, r, color=col[r], fontsize=8)
            f3.add_trace(go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode="lines+markers+text", text=["", r], name=r,
                                      line=dict(color=col[r], width=6), showlegend=i == 0), row=1, col=i + 1)
        a.axhline(0, color="grey", lw=.5); a.axvline(0, color="grey", lw=.5); a.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal")
        a.set_title(k.replace(" (", "\n("), fontsize=9); a.set_xlabel("PC1"); a.set_ylabel("PC2") if i == 0 else None
    fig.text(0.5, 0.47, "c  Each model's own PCA of its 4 rule arrows (axes are not shared between models; PC1 flipped so the arrows sum positive)", ha="center", fontsize=11)
    fig.savefig(OUT / "geometry_3models.png", dpi=160); fig.savefig(OUT / "geometry_3models.pdf")
    f3.update_layout(title="Rule arrows, each model's own uncentred PCA"); f3.write_html(str(OUT / "directions_3d_3models.html"), include_plotlyjs=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True); gen = torch.Generator().manual_seed(0)
    Xg, X3 = load_gptoss(), load_3b()
    l18, sel05, lay05 = load_05b()
    res = {"rules": RULES, "gptoss20b": per_layer(Xg, gen), "qwen3b": per_layer(X3, gen)}
    res["qwen05b"] = {"l18": {"rules": list(l18), "cos": cosmat(list(l18.values())).tolist()},
                      "selected": {"layers": lay05, "cos": cosmat([sel05[r] for r in RULES]).tolist()},
                      "sign_check_l18_vs_selected": {r: float(l18[r] @ sel05[r]) for r in l18}}
    views = {"Qwen 0.5B L18 (depth 0.78; no lexical)": list(l18.values()),
             "Qwen 3B L26 (depth 0.74)": [X3[r]["fit"][:, 26].mean(0) for r in RULES],
             "Qwen 3B L35 (last)": [X3[r]["fit"][:, 35].mean(0) for r in RULES],
             "gpt-oss-20b L17 (depth 0.74)": [Xg[r]["fit"][:, 17].mean(0) for r in RULES],
             "gpt-oss-20b L23 (last)": [Xg[r]["fit"][:, 23].mean(0) for r in RULES]}
    res["pca_views"] = {}
    for k, V in views.items():
        xyz, e = svd3(V); res["pca_views"][k] = {"coords": xyz, "retained_energy": e, "cos": cosmat(V).tolist()}
    (OUT / "geometry.json").write_text(json.dumps(res, indent=1)); plot(res)
    for m in ("qwen3b", "gptoss20b"):
        for row in res[m]:
            print(m, row["layer"], round(row["mean_cross_cos"], 3), "null95", round(row["null_abs95"], 3),
                  "rel", {r: round(v, 2) for r, v in row["reliability"].items()})
    print(json.dumps(res["qwen05b"], indent=0)[:1500])
    for k, v in res["pca_views"].items():
        print(k, np.round(np.array(v["cos"]), 2).tolist(), round(v["retained_energy"], 2))


if __name__ == "__main__":
    main()
