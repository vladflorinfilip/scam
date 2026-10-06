"""Render data/experiments/statistics_3b/stats.json as markdown tables (report_tables.md)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/experiments/statistics_3b"
NAMES = ["s1", "voice", "clause", "lexical", "lexical_e6"]


def pct(b):
    if not b:
        return "–"
    lo, hi = b["ci95"]
    return f"{100 * b['value']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def pts(b):
    return pct(b).replace("%", "")


def sp(x):
    if not x:
        return "–"
    sd = "–" if x["sd"] is None else f"{100 * x['sd']:.1f}"
    return f"{100 * x['mean']:.1f} ± {sd} (min {100 * x['min']:.1f}, max {100 * x['max']:.1f}, n={x['seeds']})"


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def main():
    s = json.loads((OUT / "stats.json").read_text())
    L = ["# Qwen2.5-3B statistics: tables", "",
         "All values are percentages with 95% bootstrap intervals (10,000 resamples). "
         "Held-out pairs are resampled as matched cue pairs (n≈200); free generation is resampled by prompt (n=100). "
         "Critic nulls count as not following. Drops are percentage points.", ""]

    L += ["## 1. Training: held-out rule-following per seed", "",
          "| Rule | Seed | Adapter (pairs) | Base (pairs) | Adapter (free) |", "|---|---:|---|---|---|"]
    for n in NAMES:
        for seed, r in s["seeds"][n]["per_seed"].items():
            L.append(f"| {n} | {seed} | {pct(get(r, 'paired', 'adapter_follow'))} | "
                     f"{pct(get(r, 'paired', 'base_follow'))} | {pct(get(r, 'free', 'adapter_follow'))} |")
    L.append("")

    L += ["## 2. Seed scans (layers 24–35): own-rule ablation", "",
          "Selected = base-safe layer chosen on the selection set (≥95% base preservation); held-out metrics below. "
          "Strongest = largest selection-set drop, ignoring base preservation (selection-set metrics).", "",
          "| Rule | Seed | Sel. layer | Held-out drop | Base preserved | Answers = 1 | Random max drop | Free before → after (sel.) | "
          "Strongest layer | Strongest drop / base pres. (selection) | Free before → after (strongest) |",
          "|---|---:|---:|---|---|---|---:|---|---:|---|---|"]
    for n in NAMES:
        for seed, r in s["seeds"][n]["per_seed"].items():
            sc = r.get("scan")
            if not sc:
                L.append(f"| {n} | {seed} | not scanned (adapter < 90% follow) | | | | | | | | |")
                continue
            sel = sc["selected"]
            fr = r.get("scan_free") or {}
            st = r.get("strongest") or {}
            sf = r.get("strongest_free") or {}
            m = st.get("selection_metrics", {})
            L.append(
                f"| {n} | {seed} | {sc['selected_layer']} | {pts(sel['follow_drop'])} | {pct(sel['base_preservation'])} | {pct(sel['predicts_1'])} | "
                f"{100 * sc['random_max_drop']:.1f} | {pct(fr.get('adapter_follow'))} → {pct(get(fr, 'adapter_selected', 'follow'))} | "
                f"{st.get('layer', '–')}{' (=sel.)' if st.get('same_as_selected') else ''} | "
                f"{100 * m['follow_drop']:.1f} / {100 * m['base_preservation']:.1f} | " if m else
                f"| {n} | {seed} | {sc['selected_layer']} | {pts(sel['follow_drop'])} | {pct(sel['base_preservation'])} | {pct(sel['predicts_1'])} | "
                f"{100 * sc['random_max_drop']:.1f} | {pct(fr.get('adapter_follow'))} → {pct(get(fr, 'adapter_selected', 'follow'))} | – | – | "
            )
            if m:
                L[-1] += f"{pct(sf.get('adapter_follow'))} → {pct(get(sf, 'adapter_selected', 'follow'))} |"
            else:
                L[-1] += "– |"
    L.append("")

    L += ["## 3. Across seeds (mean ± SD over seeds)", "",
          "| Rule | Adapter follow (pairs) | Free follow | Sel. held-out drop | Sel. base preserved | Sel. free drop | Strongest free drop |",
          "|---|---|---|---|---|---|---|"]
    for n in NAMES:
        a = s["seeds"][n]["across_seeds"]
        L.append(f"| {n} | {sp(a['adapter_follow'])} | {sp(a['free_follow'])} | {sp(a['scan_selected_drop'])} | "
                 f"{sp(a['scan_base_preservation'])} | {sp(a['scan_free_drop'])} | {sp(a['strongest_free_drop'])} |")
    L.append("")

    L += ["## 4. Seed 0, original full 36-layer scan (selected layer, held-out)", "",
          "| Rule | Layer | Adapter | Base | Drop | Base preserved | Answers = 1 | Random max drop |",
          "|---|---:|---|---|---|---|---|---:|"]
    for n, e in s["seed0_full_scan"].items():
        fs = e.get("full_scan")
        if not fs:
            continue
        L.append(f"| {n} | {fs['selected_layer']} | {pct(fs['adapter_follow'])} | {pct(fs['base_follow'])} | "
                 f"{pts(fs['selected']['follow_drop'])} | {pct(fs['selected']['base_preservation'])} | "
                 f"{pct(fs['selected']['predicts_1'])} | {100 * fs['random_max_drop']:.1f} |")
    L.append("")

    L += ["## 5. Cross-rule transfer (seed 0, lexical = lexical_e6)", "",
          "| Layer | Rule | Shared subspace: follow | base preserved | Own direction: follow | base preserved |",
          "|---|---|---|---|---|---|"]
    for layer, t in s["transfer"].items():
        for r in ["s1", "voice", "clause", "lexical"]:
            x = t[r]
            L.append(f"| {layer} (rank {t['rank']}) | {r} | {pct(x['selected']['follow'])} | {pct(x['selected']['base_preservation'])} | "
                     f"{pct(x['own']['follow'])} | {pct(x['own']['base_preservation'])} |")
    L.append("")
    (OUT / "report_tables.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
