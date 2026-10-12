# Opposite direction: what sits on the other side of the rule arrows?

CPU only. Uses saved activations (layers 24–35, 3B, seed 0) from the normal clause and voice adapters and the
label-flipped ones (same cue data, opposite final answer). Code: `experiments/kernel/opposite.py`,
figure: `experiments/kernel/plot_opposite.py`.

**Idea.** Same test pairs, same cue orientation (cue says 1 minus cue says 0), base model subtracted.
- normal model: answer = cue, so its arrow = cue part + answer part
- flipped model: answer = opposite of cue, so its arrow = cue part − answer part

So the cue part = (normal + flipped)/2 and the answer part = (normal − flipped)/2. Directions are fitted on
fit+select pairs (80) and tested on held-out eval pairs (52). Sizes use unbiased split-half products.
Pairs are resampled 2000 times for the intervals, and p-values come from 10,000 sign flips with Holm correction.

**Results**
1. **The opposite side of the 3D plot holds the flipped-answer models.** In the original PCA axes (layer 35),
   all four normal arrows point to PC1 < 0. Both flipped models land at PC1 > 0 (+0.65 clause, +0.81 voice).
   The answer parts sit with the normal arrows (−0.78, −0.91), next to the base-model answer direction (−0.63).
   The cue parts sit near PC1 = 0. So PC1 is mostly an answer axis. The opposite of "these rules"
   is "the same cue, read with the opposite answer", not anti-steganography.
2. **Raw cosine, normal vs flipped,** goes from +0.4 (layers 24–26) to −0.3 (clause) and −0.73 (voice) at layer 35.
3. **The answer share grows with depth.** At layers 24–26 the answer part is about 30% of the arrow. At layers 32–35
   it is 62–79% for clause and 86–89% for voice (narrow 95% intervals).
4. **Both parts are real** on held-out pairs at every layer for both rules (48/48 tests, Holm p < 0.01).
   The cue part has almost no readout component at layer 35 (dot with readout +0.02 clause, −0.06 voice),
   so it cannot change the answer there.
5. **The cue part is shared between clause and voice from layer 27 on.** After removing both rules' answer
   directions (plus readout and base answer at layer 35), one rule's cue direction still separates the
   other rule's held-out cue pairs (effect size 1.5–4.3, 93–100% of pairs positive, Holm p < 0.01 at layers 27–35).
   At layers 24–25 the sign is reversed and small (−0.4 to −0.7).

**Limits.** One flipped model per rule, seed 0 only, two rules only (clause and voice), and 3B only.
The split assumes the two adapters share the same cue part. A leftover answer signal outside the removed mean
answer directions cannot be fully excluded (answer strength is similar: mean |margin| 7.9 vs 7.2 for clause and
5.7 vs 6.5 for voice). Shared and decodable does not mean causal: this does not show the model uses the cue part.
