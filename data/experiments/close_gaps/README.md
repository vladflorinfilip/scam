# Close-gaps experiments (branch `close-gaps`)

Analysis: `python -m experiments.close_gaps_analysis` (CPU). Raw data: `kernel_all/` (all-layer activations), `residuals/layer_check/` (per-layer removal margins), `shared_cue/` (causal test margins).
× = follows the cue at the normal 0 cut-off. ● = follows the cue at the best cut-off, fitted on other pairs (0.5 ≈ cue gone). bp = base-model answers kept.

## 1. New models all learned
Flip seeds (s1 1–2, clause 1–2, voice 2) follow the cue 0–5.5% (flipped as intended); matched 10-epoch lexical normal seeds 0–2: 100%; 10-epoch lexical flips seeds 1–2: 0%; retrained 0.5B lexical: 100%. Base 48–61%.

## 2. Lexical across layers (own-direction removal, eval pairs; random directions never change ●: 0.98)
- **Qwen 3B (6 and 10 epochs give the same picture):** removal does nothing at layers 0–34 (× ≥ 0.96, ● ≥ 0.95). Only layer 35 removes it: × 0.46–0.48, ● 0.48 (base 0.50), bp 0.92.
- **Qwen 0.5B (retrained adapter):** starts to weaken at layer 14 (× 0.88, ● 0.91); from layer 19 it is removed (● 0.51–0.57, base 0.49), but bp is only 0.81–0.86.
- **Other 3B rules, same scan:** voice is partly removed from layer 27 (× 0.50, ● 0.71–0.88) and fully at 35 (● 0.52). Clause only at 35 (● 0.49). s1 at layers 32–34 drops × to 0.51–0.60 but ● stays 0.98: answers collapse, cue remains.

## 3. Cue/answer split at every layer, 2–3 seeds
- Answer share of the arrow (eval pairs): < 0.16 at layers 0–18 for all rules; at layer 35 s1 0.84–0.87, voice 0.86, clause 0.59–0.66, lexical 0.31–0.52 (matched 10-epoch lexical).
- Cue direction is stable across seeds (cos 0.5–0.93, lower in the last layers).
- Cross-rule cue cosines (seed-averaged) are ~0 at layers 0–26 for all pairs. Late (27–35): voice–clause 0.29–0.36, s1–clause and s1–voice 0.1–0.2, lexical with anything ≈ 0 (−0.16 to 0.08).
- With matched epochs, lexical still has the smallest answer share and shares no cue: the earlier epoch-mismatch caveat is resolved.

## 4. Is the shared cue used? (causal test, layers 27–30, 3B seed 0)
Remove the other three rules' cue directions (answers projected out, rank 3) from the target rule's model; compare with 200 random rank-3 subspaces.
- Cue gap drops by 1–3% for clause (L27–30), 1–1.4% voice (L27–28), 0.3% s1 (L27); random 0–0.1%. Each p = 0.005 (smallest possible with 200 randoms), Holm over the 12 tests run = 0.06: **not significant after correction**. Lexical: no drop.
- ● is unchanged (0.98) in every case. Even removing the rule's own cue direction keeps ● at 0.98.
- **Conclusion: the shared cue piece is readable but the model barely uses it.** No common steganography code with a causal role at these layers.
- s1 and voice L29–30 were not run (Modal timeout); given the size of the effect elsewhere they are unlikely to change this.

## Not done
GPT-OSS flip model and ● check; ● check on free-writing drops.
