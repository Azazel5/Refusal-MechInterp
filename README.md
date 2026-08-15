# Refusal-MechInterp

Mechanistic-interpretability investigation of false-positive refusals on benign
self-experimentation/self-report health queries: do they share a mechanism with
true-positive refusals on genuine risk queries, and is that mechanism separable and
correctable? Full research brief: [docs/BRIEF.md](docs/BRIEF.md).

**Status (2026-08-15): scaffolding only.** No GPU/model access is configured yet —
repo structure, dataset schema, and experiment scripts exist and are dry-run-tested,
but nothing has been executed against Gemma 3 27B-IT or Llama 3.3 70B-IT. See the
brief's revision log for what's confirmed vs. still open.

## Layout

```
docs/BRIEF.md                    Full research brief (source of truth for this phase)
data/schemas/item_schema.json    Matched-triple item schema
data/seeds/                      OR-Bench / XSTest pulls (run scripts/fetch_seed_datasets.py)
data/matched_triples/            Built dataset (run scripts/build_matched_triples.py)
scripts/                         Dataset acquisition + construction
src/refusal_mechinterp/
  models/backend.py              ModelBackend abstraction (generation + activation access)
  scoring/refusal_classifier.py  3-way refusal/partial/compliance classification
  patching/refusal_direction.py  Arditi et al. mean-difference refusal direction
experiments/
  step0_baseline/                Behavioral measurement + stop-and-pivot gate
  step1_localization/            Q1: path patching (blocked on model access)
  step2_intervention/            Q4: ablation + headline 2x2 (blocked on Step 1)
  step3_feature_check/           Q3 optional: SAE feature conflation (blocked on Step 1)
configs/models.yaml              Model/SAE/layer config (verify release ids before use)
reports/decision_gate_template.md  End-of-phase go/pivot checklist
```

## Getting from scaffold to execution

1. `pip install -r requirements.txt`, then `python scripts/fetch_seed_datasets.py`
   to pull OR-Bench/XSTest (no model access needed — public HF datasets).
2. `python scripts/build_matched_triples.py` to assemble `data/matched_triples/`
   (currently 3 topics/category as a starter set — extend toward the 15-25/category
   target in the brief before treating results as final).
3. Choose and wire up a `ModelBackend` in `src/refusal_mechinterp/models/backend.py`
   for Gemma 3 27B-IT / Llama 3.3 70B-IT (local GPU + TransformerLens/SAELens is
   required for Steps 1-3; a hosted chat API only covers Step 0).
4. `python experiments/step0_baseline/run_baseline.py --model <id>` — this is the
   stop-and-pivot gate. Only proceed to Step 1 if it clears.
5. Step 1 -> Step 2 -> (optional) Step 3, per `docs/BRIEF.md` section 4.
6. Fill in `reports/decision_gate_template.md` before any full write-up.

## Explicitly out of scope this phase

Manifold-vs-linear geometry test and cross-scale (2B/12B/27B) generalization — noted
as future work only, per the brief.
