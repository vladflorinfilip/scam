# Are the hidden rules correlated once the 1–0 answer is removed? (Qwen 3B, seed 0, layers 24–35)

Made by `python -m experiments.kernel.cue_geometry` and `python -m experiments.kernel.plot_cue_geometry`.

**Method.** For each rule we have a normal model and a label-flipped model (same data, answer = 1 − cue).
- **Cue part** = (normal + flipped) / 2: what stays the same when the answer flips.
- **Answer part** = (normal − flipped) / 2.

Directions are fitted on fit+select pairs and compared with held-out eval pairs.
Rules used: s1, voice, clause. **Lexical** is skipped because its flipped model (6 epochs) didn't learn: it followed the cue 43% of the time vs the base model's 50%. A 10-epoch retry is in `kernel_exploration/flip/lexical_flip_e10` if it worked.

**Results**
1. **The late correlation between rules is the answer.** From layer 27, the raw arrows of two rules have a cosine of 0.3–0.7. Their answer parts have 0.65–0.86. The 1–0 answer is shared by all rules (panel a).
2. **The cue parts are only weakly aligned.** Their cosine is about 0.05 (s1–voice), 0.0–0.3 (s1–clause) and 0.2–0.33 (voice–clause). That's within or near the cosine noise level, even though each rule's own cue direction is very stable (0.91–0.99 between fit and held-out pairs). So by angle, **the rules are mostly scattered** (panel b).
3. **A small shared cue component is still real.** After removing all answer directions, rule A's cue direction separates rule B's held-out cue pairs. This holds for almost every pair at layers 27–35 (Holm p < 0.01, 72 tests).
   - Effects are large where clause is involved: d = 0.4–4.5.
   - Effects are small for s1↔voice: d = 0.4–0.8, positive and significant only from layer 29.
   - At layers 24–26 some transfers are *negative*: there, the cue parts point in opposite directions (panel c).

**Plain reading.** Each rule has its own cue direction, and these are mostly unrelated. On top of that sits a big shared 1–0 answer signal. Underneath, a small shared cue piece appears from layer 27: a minor overlap, not a common code.

**Limits.** Seed 0 only, one flipped model per rule, 3 of 4 rules. The shared piece is readable but not shown to be used by the model.
