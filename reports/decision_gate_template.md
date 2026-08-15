# Decision Gate — Phase 1 (docs/BRIEF.md section 6)

Fill in once Steps 0-2 (and 3, if reached) have run on real model access. Do not
proceed to a full write-up without completing this checklist.

- [ ] **Step 0 replication.** Did Step 0 replicate a measurable false-positive gap on
      tier (a)/(b) vs tier (c) in at least one model?
      - Gemma 3 27B-IT: <fill in tier-a/b/c refusal rates, cite results/step0/gemma-3-27b-it_summary.csv>
      - Llama 3.3 70B-IT: <same, results/step0/llama-3.3-70b-it_summary.csv>
      - If **no** in both models: STOP — pivot to BizzaroWorld/MEMIT-collateral-damage angle.

- [ ] **Step 1 localization.** Separable shallow-heuristic path, or fully conflated
      with true-risk detection?
      - <fill in, cite results/step1/*_attribution.csv and subway-map figure>
      - Note: conflated-with-no-separable-fix is a weaker "so what" for Q4 — flag
        explicitly if this is the outcome, don't bury it.

- [ ] **Step 2 intervention.** Did the 2x2 move in the right direction (lower benign
      refusal, stable risk refusal) with a clean effect size? Consistent across both
      architectures, or noisy in one?
      - <fill in, cite results/step2/*_2x2.csv and the PASS/FAIL from run_intervention.py>

- [ ] **Narrative fit.** Is there a single Q1->Q4 arc that fits a 12-20hr write-up, or
      does it need narrowing to one model / one category?
      - <fill in>

## Recommendation

<one paragraph: (a) proceed to full write-up, (b) narrow to a single sharper
sub-claim, or (c) pivot to BizzaroWorld/MEMIT-collateral-damage. State which and why
in one paragraph, per the brief — this is the explicit checkpoint before continuing.>
