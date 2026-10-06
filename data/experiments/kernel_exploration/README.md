# Kernel exploration: transfer and a common direction at 3B layer 35

CPU only. Code: `experiments/kernel/l35.py` (analysis), `experiments/kernel/readout.py` (exact layer-35 readout),
`experiments/kernel/plot_l35.py` (figure). Results: `l35/results.json`, figure `l35/l35_transfer.{png,pdf}`.

## Setup
- Qwen2.5-3B, layer 35 (last block). After it come only the final RMSNorm and the tied '0'/'1' embedding rows,
  which LoRA does not change, so any layer-35 removal can be computed exactly from saved activations.
  Check: CPU margins match the saved GPU margins to ~0.05 logits (bf16).
- Rules s1, voice, clause, lexical (6 epochs). Two independent seed sets: all adapters seed 0, all seed 2.
- Rule direction = mean over fit pairs of (cue-1 − cue-0) adapter activation minus the same for the base model.
  All scores use held-out eval pairs only.
- Metric: share of the learned cue effect remaining = (cue gap after − base gap) / (cue gap before − base gap).
  1 = untouched, 0 = removed, < 0 = reversed. "Kept" = base-model answers unchanged by the same removal.
- Controls: readout direction r = norm ⊙ (e₁ − e₀); base-model answer direction (base, no rule).
- Null for removals: sign-flip random directions (random cue sign per fit pair), matched to the tested
  direction's cosine with r. Holm correction over the 20 removal tests per seed set.

## Results
1. **Layer-35 removal acts only through the readout r.** Removing a rule's own direction minus its r component
   leaves 100–110% of every cue. Removing r alone wipes all four cues but changes 37–55% of base answers.
2. **Own directions are specific** for voice, clause, lexical: ~0% left in both seed sets, beats the matched null
   (Holm p = 0.02 in each set). s1's own direction reverses its cue (−0.5; p = 0.004, Holm 0.07).
3. **Cross-rule removal is partial and no better than chance.** s1, voice and clause partly remove each other
   (0.2–0.8 left); lexical is untouched by the others (0.87–1.02). No cross-rule removal beats the matched null
   after Holm correction.
4. **No common removal direction.** Fitted on three rules, removed from the fourth: s1 −1.3 and voice −0.2
   (reversed), clause 0.59, lexical 0.92; 69–88% of base answers kept. The base-model answer direction does about
   as well (−0.2, 0.27, 0.30, 0.97), so the shared effect looks like the answer, not a shared code.
5. **Geometry.** Only s1–voice align (cos 0.66 / 0.67), p ≈ 0.02 uncorrected, not significant after Holm.
6. **Kernel probes (leave one rule out).** Linear and RBF kernel-ridge probes, trained on the other three rules'
   cue contrasts after projecting out r and the base answer direction, read the held-out cue of s1, voice and
   clause above a sign-flipped-training null in both seed sets (Holm p ≤ 0.024; one exception: voice linear,
   seed set 0, Holm 0.11). Lexical: no transfer (Holm ≥ 0.52). Probes trained on base-model answers alone do not
   transfer (all p > 0.27). Accuracy is not a useful statistic here (random-sign probes also reach ~100% or ~0%
   because held-out contrasts share one direction); the test uses the mean/sd of held-out scores.

## What we can claim
- At layer 35 there is no common *causal* direction: removal works only through the readout, and rule-specific
  removal beats chance only for the rule's own direction.
- There is a shared, *readable* cue code for s1, voice and clause outside the readout and base-answer
  directions; lexical is separate. This is correlational: at layer 35 directions orthogonal to r cannot change
  the answer, so its causal role must be tested at earlier layers.

## Caveats
- In every adapter the answer equals the cue, so the shared code may be the adapter's own off-readout encoding of
  its decided answer rather than steganography. The base-answer control does not rule this out.
  Smallest test (GPU): run each adapter on another rule's pairs, where its answer no longer tracks that cue.
- Two seed sets; lexical and clause have 52 held-out pairs, s1 and voice 100.
- 0.5B layers 20–22 not done: the 0.5B voice, clause and lexical adapters are not available.
