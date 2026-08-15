# Project Brief — Refusal MechInterp (Phase 1)

> Source of truth for this phase. Pasted verbatim from the research brief provided
> 2026-08-15. Do not re-derive; execute against this. If it needs revision, edit this
> file and note the change below rather than re-deriving from scratch elsewhere.

## 1. Objective

Determine whether false-positive refusals on benign self-experimentation/self-report
health queries (e.g. personal diet, fasting, or training logs) share a mechanism with
true-positive refusals on genuine risk queries, and whether that mechanism is separable
and correctable. Answer must be **falsifiable and numeric** — a clean go/no-go signal by
the end of this phase, not a vibes-based writeup.

**Core questions, in priority order:**
- **Q1 (spine):** Localization — does path patching from lexical-restriction/self-report
  tokens into the refusal-direction/circuit show the same heads/layers driving both
  benign and genuine-risk refusals (conflated), or a separable shallow heuristic
  (correctable)?
- **Q4 (payoff):** Intervention — does a targeted patch/ablation reduce false-positive
  refusal rate on the benign set without reducing true-positive refusal rate on the
  risk set? Report as a 2×2: refusal rate × {benign, risk} × {before, after}.
- **Q3 (optional, if time allows):** Feature-level — do SAE features conflate the two
  classes at a given layer, or are they represented separately and pooled downstream?
- **Q2/Q5 (explicitly out of scope this phase):** manifold-vs-linear geometry test,
  cross-scale (2B/12B/27B) generalization. Note as future work only.

**Decision gate at the end of this phase:** with the numbers in hand, decide whether to
(a) proceed to full write-up on this direction, (b) narrow to a single sharper sub-claim,
or (c) pivot to the BizzaroWorld/MEMIT-collateral-damage angle instead. Do not proceed
past this brief without that explicit checkpoint.

## 2. Model & Tooling Stack

### Models — cross-architecture pair
| Model | Role | Why |
|---|---|---|
| Gemma 3 27B-IT | Primary | Full Gemma Scope 2 coverage: SAEs on every layer (residual, MLP-out, attention-out), transcoders, crosscoders, NLA at layer 41. |
| Llama 3.3 70B-IT | Cross-architecture check | SAE coverage on Neuronpedia, NLA at layer 53, existing "activation-capped" demo comparing default vs. safety-behavior-modified Llama 70B. |

Size mismatch (27B vs 70B) is accepted — matches Neuronpedia NLA tutorial precedent
(Fraser-Taliente, Kantamneni, Ong et al. 2026). Tooling coverage is the bottleneck, not
size-matching.

### Interpretability tooling
- **SAEs:** Gemma Scope 2 (Gemma 3 27B-IT) via SAELens; Llama SAE release via same.
- **Transcoders / cross-layer transcoders:** Gemma Scope 2 `transcoder_all` releases.
- **NLA:** both models, layers noted above — qualitative cross-check only.
- **J-Lens (Jacobian Lens):** reuse BizzaroWorld self-fit pipeline, retarget at
  refusal-direction rank instead of factual-recall-token rank.
- **Autointerp:** Neuronpedia explanations as first pass; manually spot-check ≥20
  features — do not trust labels uncritically.
- **Refusal direction:** reproduce via Arditi et al. (2024) mean-difference method
  (harmful − harmless activation means).

## 3. Dataset Construction Protocol

Self-built, seeded from OR-Bench and/or XSTest public benign/over-refusal subsets, then
extended with matched-triple items per category via templated variation (not freehand).

**Categories** (~15–25 matched triples each, 4 categories → ~60–100 items):
1. Self-experimentation / quantified-self health (flagship category)
2. Endurance/extreme sport self-report
3. Supplement/substance self-titration within legal/OTC bounds
4. Held-out true-positive control, pulled unmodified from OR-Bench/XSTest's unsafe set

**Each triple = 3 items, same surface topic:**
- (a) Clearly benign, explicit self-report/control framing
- (b) Ambiguous — same lexical pattern, thinner self-report framing, no explicit
  timeframe/control signal
- (c) Genuine-risk analog — same surface vocabulary, no self-report framing,
  crisis-adjacent — pulled directly from source benchmark, unmodified

**Labeling:** ground-truth safe/unsafe label (from source benchmark or category
construction) + tier (a/b/c).

## 4. Experiment Pipeline

- **Step 0 — Baseline behavioral measurement.** Run all items through both models.
  3-way classification: refusal / partial-refusal / compliance, per category, per tier.
  **Stop-and-pivot gate:** if false-positive refusal rate on tier (a) is near zero for
  both models, stop, do not proceed to Q1.
- **Step 1 — Q1 localization.** Path-patch token classes (numeric quantities,
  restriction verbs, first-person self-report markers, timeframe markers) into the
  refusal-direction projection/logit, per layer, both models, matched triples. Output:
  layer/head attribution table + subway-map figure.
- **Step 2 — Q4 intervention.** Targeted patch/ablation on the Step 1 localized
  component (or full linear refusal direction if no separable component found).
  Re-run Step 0 measurement post-intervention. Headline output: 2×2 table.
- **Step 3 — Q3 optional feature-level check.** Only if time remains. Top-activating
  SAE features on tier (a) vs (c) at localized layer(s); conflation vs. pooled-downstream
  check; manual autointerp spot-check on top ~20 features.

## 5. Cross-Architecture Comparison

Run Steps 0–2 (and 3 if reached) on both models independently; report side by side, do
not pool. Question: shared architectural/training artifact, or model-specific? Either
answer is valid — do not force convergence.

## 6. Decision Gate (end of this phase)

- [ ] Did Step 0 replicate a measurable false-positive gap on tier (a)/(b) vs tier (c)
      in at least one model?
- [ ] Did Step 1 find a *separable* shallow-heuristic path, or full conflation?
- [ ] Did Step 2's intervention move the 2×2 in the right direction with a clean effect
      size, consistent across architectures?
- [ ] Is there a single-narrative arc across Q1→Q4 that fits a 12–20hr write-up, or does
      it need narrowing?

**Deliverables:** behavioral baseline table, localization figure, 2×2 intervention
table, one-paragraph go/pivot recommendation.

---

## Revision log
- 2026-08-15: Brief received and scaffolded. No prior project history found in this repo
  or in agent memory — confirmed with user this is the first pass, not a continuation.
  Model/API access not yet configured; this phase built scaffolding only (repo structure,
  dataset schema, script skeletons). Execution against real model access is a separate,
  explicit next step.
