# Qwen2.5-3B: statistical analysis (3 training seeds + bootstrap CIs)

Branch `statistical-analysis`. Seeds 1 and 2 retrain every rule with the same data, splits and recipe (only training RNG changes); seed 0 is the original run. Every seed with ≥90% held-out follow was rescanned over layers 24–35 with 5 random controls, and free generation was run at the selected (base-safe) layer and at the strongest layer. Intervals are 95% bootstrap (10,000 resamples; matched pairs for held-out, prompts for free generation). Full tables: `report_tables.md`; raw numbers: `stats.json` (from `experiments/stats_3b.py`, rendered by `experiments/report_stats_3b.py`).

## What holds up across seeds

1. **s1 and clause install reliably** (held-out follow 98.5–100% at all 3 seeds; base 60.5% / 51%).
2. **Matched-recipe lexical fails at every seed** (52.9–57.7% vs 50% base). It is a real negative result, not an unlucky seed.
3. **Random directions never change rule-following** (max drop 0.0 in all 10 seed scans and in the original scans).
4. **Voice has no base-safe removal.** At both scanned seeds the effective layers (27–35 at seed 0, 32–35 at seed 2) remove the rule on the selection set (~50 pt drop at the strongest layer, 35) but preserve only 88–90% of base answers there; the base-safe layer (26) does nothing.
5. **lexical_e6 is only removable at the last layer (35)**, at both scanned seeds, and nothing base-safe works earlier.
6. **Strong ablations mostly collapse the answer to one label** rather than restoring base behaviour (e.g. s1 seed 0: 4% answers = 1; s1 seed 2: 94%; clause seed 2: 0%).
7. **Strongest-layer free generation confirms the effect transfers to the model's own reasoning**, but at a cost to base behaviour: clause (layer 35) 98–100% → 43–57% at seeds 0–1; lexical_e6 (layer 35) 99–100% → 59–90%; voice (layer 35) only 92–94% → 78–85%.

## What is seed-sensitive

1. **Voice and lexical_e6 training**: one seed of three only partly learns the rule on held-out pairs (voice seed 1 61.5% [57.5, 65.5]; lexical_e6 seed 1 68.3% [61.5, 75.0]), even though their free generations still follow it 91% / 88% of the time.
2. **s1's layer and effect size**: selected layers 34 / 31 / 26; held-out drop 45.5 / 18.2 / 44.4 pts (mean 36.0 ± 15.5). The headline free-generation result (95% → 27%) is **seed 0 only**; seeds 1–2 drop just 98% → 80% and 98% → 86%.
3. **clause's location**: seeds 0–1 respond only at layer 35 (base preservation ~77%), but seed 2 has a base-safe direction at layer 32 (held-out 100% → 50%, base 97.1%; free 97% → 10%), via collapse to answer 0.

## Bottom line

The training results and the controls replicate. The localisation story only partly replicates: "late layers only, mostly answer-switch directions, random controls null" holds for every seed, but *which* layer and whether a base-safe removal exists varies by seed for s1 and clause. The earlier claim "s1 is the one cleanly removable rule" should be weakened to "seed-dependent". Cross-rule transfer (layers 30 and 35) was run on seed 0 only; its bootstrap CIs are in `report_tables.md` §5.
