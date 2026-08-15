# Refusal-MechInterp

Mechanistic-interpretability investigation of false-positive refusals on benign
self-experimentation/self-report health queries: do they share a mechanism with
true-positive refusals on genuine risk queries, and is that mechanism separable and
correctable? Full research brief: [docs/BRIEF.md](docs/BRIEF.md). Dataset construction
spec (supersedes BRIEF.md section 3): [docs/DATASET_PLAN.md](docs/DATASET_PLAN.md).
Model acquisition (Colab/HPC): [docs/MODEL_ACQUISITION.md](docs/MODEL_ACQUISITION.md).

**Status (2026-08-15): scaffolding only.** No GPU/model access is configured in
*this* environment — but `HFLocalBackend` is now fully implemented against
TransformerLens and ready to run once loaded onto Colab/HPC. Repo IDs
(`google/gemma-3-27b-it`, `meta-llama/Llama-3.3-70B-Instruct`) are live-verified to
exist and be gated; SAE release strings are not yet verified (needs `sae_lens`
installed). Nothing has actually been executed against either model. See the docs'
revision logs for what's confirmed vs. still open.

## Layout

```
docs/BRIEF.md                    Full research brief (source of truth for this phase)
docs/DATASET_PLAN.md             Dataset construction spec (supersedes BRIEF.md section 3)
docs/MODEL_ACQUISITION.md        Colab/HPC model + SAE acquisition, repo-ID verification
data/schemas/item_schema.json    Matched-triple item schema (per DATASET_PLAN.md section 2)
data/seeds/                      OR-Bench / XSTest pulls (run scripts/fetch_seed_datasets.py)
data/matched_triples/            Built dataset (run scripts/build_matched_triples.py)
scripts/
  fetch_seed_datasets.py         Pull OR-Bench / XSTest
  build_matched_triples.py       Build tiers (a)/(b), leave tier (c) UNRESOLVED
  resolve_tier_c.py              Shortlist tier-(c) candidates by embedding similarity (suggests, doesn't auto-assign)
  validate_dataset.py            Dataset-plan validation checks 2+4 (no model access needed)
  verify_repo_ids.py             Live-check HF repo IDs + SAELens release strings
  download_models.py             Snapshot / quantized in-memory load, Colab or HPC
  sanity_check.py                Pre-flight check: generation, chat template, activation-cache hook names
src/refusal_mechinterp/
  models/backend.py              ModelBackend + HFLocalBackend (TransformerLens-based activation access)
  models/sae_loader.py           SAELens loading wrapper
  scoring/refusal_classifier.py  3-way refusal/partial/compliance classification
  patching/refusal_direction.py  Arditi et al. mean-difference refusal direction
experiments/
  step0_baseline/                Behavioral measurement + stop-and-pivot gate
  step1_localization/            Q1: path patching (blocked on model access)
  step2_intervention/            Q4: ablation + headline 2x2 (blocked on Step 1)
  step3_feature_check/           Q3 optional: SAE feature conflation (blocked on Step 1)
configs/models.yaml               Model/SAE/layer config (verify release ids before use)
reports/decision_gate_template.md  End-of-phase go/pivot checklist
```

## Getting from scaffold to execution

1. `pip install -r requirements.txt`, then `python scripts/fetch_seed_datasets.py`
   to pull OR-Bench/XSTest (no model access needed — public HF datasets).
2. `python scripts/build_matched_triples.py` to assemble `data/matched_triples/`
   (currently 3 topics/category, tier-(c) rows left as UNRESOLVED placeholders).
   `python scripts/resolve_tier_c.py` shortlists real xstest/or-bench candidates
   for each placeholder by embedding similarity — review and hand-pick, then
   `python scripts/validate_dataset.py` for the two checks that don't need model access.
3. Extend toward `docs/DATASET_PLAN.md` section 5.5's protocol: build one category
   to its full ~20-triple pilot target, pair in real tier-(c) items, then run Step 0
   on just that category before finishing the other two.
4. On Colab/HPC (not this environment): `pip install -r requirements.txt` with the
   commented-out execution block uncommented, then
   `python scripts/verify_repo_ids.py` and `python scripts/sanity_check.py --model
   <id> --layer <n>` per `docs/MODEL_ACQUISITION.md` before trusting anything else.
   `HFLocalBackend` in `src/refusal_mechinterp/models/backend.py` is the wired-up
   backend (TransformerLens engine; nnsight is a documented fallback, not yet
   implemented) — `scripts/download_models.py` handles the actual weights pull.
5. `python experiments/step0_baseline/run_baseline.py --model <id>` — this is the
   stop-and-pivot gate. Only proceed to Step 1 if it clears.
6. Step 1 -> Step 2 -> (optional) Step 3, per `docs/BRIEF.md` section 4.
7. Fill in `reports/decision_gate_template.md` before any full write-up.

## Explicitly out of scope this phase

Manifold-vs-linear geometry test and cross-scale (2B/12B/27B) generalization — noted
as future work only, per the brief.
