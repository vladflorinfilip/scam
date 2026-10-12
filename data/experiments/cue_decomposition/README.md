# Are the hidden rules correlated once the 1–0 answer is removed? (Qwen 3B, seed 0, layers 24–35)

Made by `python -m experiments.kernel.cue_geometry` and `python -m experiments.kernel.plot_cue_geometry`.

**Method.** For each rule we have a normal model and a label-flipped model (same data, answer = 1 − cue).
- **Cue part** = (normal + flipped) / 2: what stays the same when the answer flips.
- **Answer part** = (normal − flipped) / 2.

Directions are fitted on fit+select pairs and compared with held-out eval pairs.
All four rules are used. The lexical flipped model failed at 6 epochs: it followed the cue 43% of the time, vs the base model's 50% (`kernel_exploration/flip/lexical_flip`, kept as a result). The 10-epoch retry worked (0% on held-out pairs), so lexical uses `lexical_flip_e10`. **Caveat:** the normal lexical model was trained for 6 epochs and the flipped one for 10, so the lexical cue/answer split also contains any effect of the extra training.

**Results**
1. **The late correlation between rules is the answer** (s1, voice, clause). From layer 27, the raw arrows of two rules have a cosine of 0.3–0.7. Their answer parts have 0.65–0.86. The 1–0 answer is shared by all rules (panel a).
2. **The cue parts are only weakly aligned** (s1, voice, clause). Their cosine is about 0.05 (s1–voice), 0.0–0.3 (s1–clause) and 0.2–0.33 (voice–clause). That's within or near the cosine noise level, even though each rule's own cue direction is very stable (0.91–0.99 between fit and held-out pairs). So by angle, **the rules are mostly scattered** (panel b).
3. **A small shared cue component is still real** (s1, voice, clause; 48 of 54 positive and Holm-significant at layers 27–35). After removing all answer directions, rule A's cue direction separates rule B's held-out cue pairs. Holm correction is over 144 tests (12 ordered pairs × 12 layers).
   - Effects are large where clause is involved: d = 0.4–4.5.
   - Effects are small for s1↔voice: d = 0.4–0.8, positive and significant only from layer 29.
   - At layers 24–26 some transfers are *negative*: there, the cue parts point in opposite directions (panel c).

4. **Lexical stands apart.**
   - Its raw arrow, its answer part and its cue part all have cosine about 0 (−0.18 to 0.17) with the other three rules, at every layer.
   - Even its answer part isn't shared. That's surprising, and may come from the 6- vs 10-epoch mismatch.
   - Its held-out cue transfer with the other rules is mostly *negative* from layer 29 (d = −0.4 to −3.8, Holm p < 0.05).

**Note on signs.** Which text variant counts as "cue = 1" is a training convention chosen separately for each rule. A negative transfer therefore still means shared information, just labelled the other way round. Only the size of a transfer is meaningful across rules, not its sign. This applies to the early negative voice↔clause transfers too.

**Plain reading.** Each rule has its own cue direction, and these are mostly unrelated. s1, voice and clause also share a big 1–0 answer signal on top, which makes them look aligned late. Underneath, a small shared cue piece appears from layer 27: a minor overlap, not a common code. Lexical is separate in every part.

**Limits.** Seed 0 only, one flipped model per rule, lexical epoch mismatch (6 vs 10). The shared piece is readable but not shown to be used by the model.
