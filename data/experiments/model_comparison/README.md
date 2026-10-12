# Rule-direction geometry: Qwen 0.5B vs Qwen 3B vs gpt-oss-20b

Made by `python -m experiments.compare_geometry` → `geometry.json`, `geometry_3models.png/pdf`, `directions_3d_3models.html`.

**Rule arrow** = average (fine-tuned cue-1 minus cue-0) minus (base cue-1 minus cue-0) at the answer position, on the fit pairs, at seed 0.
The models have different sizes (896 / 2048 / 2880 numbers), so their arrows can't be put on one set of axes.
Instead we compare, for each model, the angles between its own four rule arrows at the same **relative depth** (layer / last layer).

* **a.** The mean cosine between the 4 rule arrows. 0 means unrelated and 1 means the same direction. The shaded band is the 95% noise level, from randomly sign-flipping pairs.
* **b.** Each arrow measured on the fit pairs vs on held-out pairs. It is near 1 late in both models, so the late arrows are well measured.
* **c.** Each model's own 3D PCA, the same kind of plot as before, shown as PC1 vs PC2. 3D views are in the HTML.

**Findings**
1. **Same pattern in all three models.** Through most of the depth, the rule arrows are unrelated: they sit inside the noise band. In the last quarter they start pointing the same way, with a mean cosine of about 0.2–0.35. This happens in Qwen 3B from L27 (depth 0.77) and in gpt-oss from L19 (depth 0.83). Qwen 0.5B at L18 (depth 0.78) has a mean cosine of 0.29 (s1, voice and clause only).
2. **The "one side" of the PCA plot appears only late, in every model.** At depth 0.74 the arrows still spread around the origin in both 3B and gpt-oss. At the last layer they all fall on one side.
3. **Why:** on Qwen 3B, the label-flip and opposite-direction tests (branch `opposite-direction`) showed that this late shared side is mostly the answer ("I'm answering 1"), not the cue. By the same pattern, gpt-oss's late alignment is probably also the answer. That has not been tested: there is no gpt-oss flipped model.
4. **Which rules pair up differs.** At the last layer, s1 and voice are closest in Qwen 3B (0.66), while s1 and lexical are closest in gpt-oss (0.58).

**Limits:** seed 0 only. The 0.5B arrows are only saved at L18 (no lexical, no activations), so 0.5B is one point.
The 0.5B L20–22 selected directions have an arbitrary sign, so they are not used.
The noise band is per layer and not multiple-test corrected. The gpt-oss values of 0.27–0.35 at L21–23 are above it.
