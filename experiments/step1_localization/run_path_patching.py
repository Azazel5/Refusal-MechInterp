#!/usr/bin/env python3
"""Step 1 — Q1 localization via path patching (docs/BRIEF.md section 4).

Blocked on: an ActivationCapableBackend (see src/refusal_mechinterp/models/backend.py)
backed by real GPU weights + TransformerLens/SAELens, and a refusal direction
(reproduced via Arditi et al. 2024 mean-difference method, see
src/refusal_mechinterp/patching/refusal_direction.py). Neither exists yet in this
environment. This file defines the intended procedure precisely so it's a fill-in-the-
backend job, not a redesign job, once compute is available.

Procedure:
1. For each matched triple, tokenize (a)/(b)/(c) items and locate the four token-class
   spans (numeric_quantity, restriction_verb, self_report_marker, timeframe_marker) —
   see data/schemas/item_schema.json `token_spans`. This requires the model's own
   tokenizer, so it happens here, not at dataset-build time.
2. Compute the refusal direction (harmful-harmless mean-difference) once per model,
   cached in results/step1/{model_id}_refusal_direction.pt.
3. For each token class, path-patch: run the (a) or (b) item, patch just that token
   class's activations at layer L from a `source` run (paired (c) item, or a
   corrupted/zero-ablated version) into the clean run, and measure the shift in the
   refusal-direction projection at the final token, for every layer L (and every head
   if patching attention-head outputs rather than the residual stream).
4. Aggregate: per (token_class, layer[, head]) contribution to refusal-direction shift,
   separately for benign (a/b) and risk (c) targets.
5. Output: results/step1/{model_id}_attribution.csv (token_class, layer, head,
   benign_contribution, risk_contribution) + a subway-map figure (reuse BizzaroWorld's
   plotting code — same visual language, new target variable) showing whether benign
   and risk routes overlap (conflated) or diverge (separable).

This intentionally reuses the BizzaroWorld swing-attribution statistical machinery
(per brief section 4) rather than building new attribution code from scratch — import
that module here once its path is known in this environment; not available yet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

TOKEN_CLASSES = ["numeric_quantity", "restriction_verb", "self_report_marker", "timeframe_marker"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/matched_triples"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path("results/step1"))
    args = ap.parse_args()

    raise SystemExit(
        "Step 1 is blocked on GPU/model access (ActivationCapableBackend) and the "
        "BizzaroWorld swing-attribution module, neither of which exist in this "
        "environment yet. See the module docstring for the exact procedure to "
        "implement once both are available. Do not stub in fake attribution numbers "
        "— Step 0's stop-and-pivot gate must pass on real behavioral data first."
    )


if __name__ == "__main__":
    main()
