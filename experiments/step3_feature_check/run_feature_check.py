#!/usr/bin/env python3
"""Step 3 (optional) — Q3 SAE feature-level conflation check (docs/BRIEF.md section 4).

Only attempt if Steps 1-2 leave time in budget (per brief).

Procedure:
1. At the layer(s) Step 1 localized, run the SAE (Gemma Scope 2 / Llama SAE via
   SAELens) on tier (a) and tier (c) items, get top-activating latent features
   at the relevant token positions.
2. Compare feature sets: same top features firing on both tiers -> conflation at
   the representation level. Disjoint feature sets that a downstream readout pools
   together -> readout-level bug, not representation-level.
3. Pull Neuronpedia autointerp explanations for the top ~20 features by activation
   magnitude; manually spot-check each against the actual activating examples
   (autointerp labels are not to be trusted uncritically — Neuronpedia's own docs
   flag NLAs can produce incorrect explanations).
4. Output: results/step3/{model_id}_feature_overlap.csv (feature_id, tier_a_freq,
   tier_c_freq, jaccard-style overlap score) + results/step3/{model_id}_autointerp_
   spotcheck.csv (feature_id, neuronpedia_label, manual_verdict, notes).

Blocked on: SAELens set up against real model access, Step 1's localized layer(s).
Not runnable yet.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True, help="Layer localized in Step 1")
    ap.add_argument("--out", type=Path, default=Path("results/step3"))
    args = ap.parse_args()

    raise SystemExit(
        "Step 3 is optional and blocked on Step 1's localized layer plus SAELens "
        "access to Gemma Scope 2 / the Llama SAE release. Skip unless Steps 1-2 "
        "leave time in budget, per docs/BRIEF.md."
    )


if __name__ == "__main__":
    main()
