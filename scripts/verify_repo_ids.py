#!/usr/bin/env python3
"""Live-verify HF repo IDs and SAELens release strings before trusting anything
hardcoded in configs/models.yaml or docs/MODEL_ACQUISITION.md — IDs drift.

Implements docs/MODEL_ACQUISITION.md section 0 exactly. Run this on whatever
machine you're about to load models on (needs `huggingface_hub` always; `sae_lens`
only for the SAE-directory check, which is the heavy optional dependency).

Usage:
    python scripts/verify_repo_ids.py                 # HF repo checks only
    python scripts/verify_repo_ids.py --check-saelens  # also scan the SAELens directory
"""
from __future__ import annotations

import argparse
import sys

# Candidate IDs currently trusted in configs/models.yaml — verify against live
# results, don't just assume the file is still correct.
CANDIDATE_MODEL_IDS = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "llama-3.3-70b-it": "meta-llama/Llama-3.3-70B-Instruct",
}


def check_hf_repos() -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed. `pip install huggingface_hub` first.")

    api = HfApi()

    print("--- HF search results (compare against configs/models.yaml hf_id) ---")
    for query in ("gemma-3-27b", "llama-3.3-70b"):
        try:
            hits = [m.id for m in api.list_models(search=query, limit=10)]
        except Exception as e:  # noqa: BLE001 - surfacing any API error is the point here
            print(f"  search='{query}': ERROR ({e})")
            continue
        print(f"  search='{query}': {hits}")

    print("\n--- Direct existence + gating check on configured IDs ---")
    for model_key, hf_id in CANDIDATE_MODEL_IDS.items():
        try:
            info = api.model_info(hf_id)
            gated = getattr(info, "gated", None)
            print(f"  {model_key:<20} {hf_id:<40} exists=True gated={gated}")
        except Exception as e:  # noqa: BLE001
            print(f"  {model_key:<20} {hf_id:<40} exists=False/ERROR ({e})")


def check_saelens() -> None:
    try:
        from sae_lens import pretrained_saes_directory
    except ImportError:
        print(
            "\nsae_lens not installed — skipping SAE-directory check. "
            "`pip install sae-lens` then rerun with --check-saelens."
        )
        return

    directory = pretrained_saes_directory()
    print("\n--- SAELens pretrained-SAE directory (filtered) ---")
    for term in ("gemma", "llama"):
        hits = [k for k in directory if term in k.lower()]
        print(f"  contains '{term}': {hits}")
    print(
        "\nCross-check the exact release strings above against what "
        "configs/models.yaml and docs/MODEL_ACQUISITION.md section 4 assume "
        "(e.g. 'gemma-scope-2-27b-it-res') before loading anything."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-saelens", action="store_true",
                     help="Also scan the SAELens pretrained-SAE directory (requires sae_lens installed).")
    args = ap.parse_args()

    check_hf_repos()
    if args.check_saelens:
        check_saelens()
    else:
        print(
            "\n(Skipped SAELens directory check — pass --check-saelens once "
            "sae_lens is installed. Also check Neuronpedia's model coverage "
            "directly in a browser/authenticated session for the Llama-side "
            "SAE/NLA layer numbers; the plain /api/models endpoint serves the "
            "app shell, not JSON, from an unauthenticated request.)"
        )


if __name__ == "__main__":
    main()
