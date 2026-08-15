#!/usr/bin/env python3
"""Sanity check before any real run — docs/MODEL_ACQUISITION.md section 6.

Run a single forward pass on one benign and one control-risk item and confirm:
1. Generation completes without OOM.
2. The tokenizer chat template matches the model's actual IT/instruct format
   (printed for manual eyeballing — Gemma and Llama use different conventions).
3. SAE/NLA hook points actually fire on a real forward pass (checked here only
   for the activation-cache hook name existing in run_with_cache's output; the
   SAE itself is loaded separately via sae_loader.py once you have a layer to check).

Usage:
    python scripts/sanity_check.py --model gemma-3-27b-it --layer 20

Not runnable in this environment (no GPU/model access). Run this FIRST on
Colab/HPC, before pointing Step 0 at the full dataset — a silent shape mismatch
here is much cheaper to catch now than mid-Step-1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BENIGN_PROBE = "I've been tracking my daily step count for the last month as part of a fitness log. What trends should I look for?"
RISK_PROBE_NOTE = (
    "Use a real control-risk item from data/matched_triples/control-risk.jsonl "
    "here once it's built (scripts/fetch_seed_datasets.py + "
    "scripts/build_matched_triples.py) — not hardcoded in this script, since "
    "risk-adjacent content should come from the vetted benchmark pull, not be "
    "authored in a sanity-check script."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gemma-3-27b-it", "llama-3.3-70b-it"])
    ap.add_argument("--layer", type=int, required=True, help="Layer to check activation caching at")
    ap.add_argument("--quantize", default="bf16", choices=["4bit", "bf16"])
    args = ap.parse_args()

    from refusal_mechinterp.models.backend import HFLocalBackend

    print(f"=== Sanity check: {args.model} ===\n")

    print("[1/3] Loading backend and checking chat template...")
    backend = HFLocalBackend(args.model, quantize=args.quantize, activation_capable=(args.quantize != "4bit"))
    templated = backend._apply_chat_template(BENIGN_PROBE)  # noqa: SLF001 - intentional inspection
    print(f"  Templated prompt:\n{templated!r}\n")
    print("  Manually confirm this matches the model's documented chat format "
          "(system/user/assistant turn markers) before trusting anything downstream.\n")

    print("[2/3] Generation check (benign probe)...")
    result = backend.generate(BENIGN_PROBE, max_new_tokens=64)
    print(f"  Completion: {result.completion!r}\n")
    print(f"  Risk-probe reminder: {RISK_PROBE_NOTE}\n")

    if args.quantize == "4bit":
        print("[3/3] Skipped activation-cache check — backend loaded as activation_capable=False "
              "(4-bit quantization, Step-0-only per docs/MODEL_ACQUISITION.md section 3).")
        return

    print(f"[3/3] Activation-cache check at layer {args.layer}...")
    _, cache = backend.run_with_cache(BENIGN_PROBE)
    hook_name = f"blocks.{args.layer}.hook_resid_post"
    if hook_name in cache:
        shape = tuple(cache[hook_name].shape)
        print(f"  OK: {hook_name} present, shape={shape}")
    else:
        available = [k for k in cache.keys() if f".{args.layer}." in k][:10]
        print(f"  MISSING: {hook_name} not in cache. Hook names near this layer: {available}")
        print("  This is exactly the silent shape-mismatch section 6 warns about — "
              "resolve before running Step 1.")

    print("\nSanity check complete.")


if __name__ == "__main__":
    main()
