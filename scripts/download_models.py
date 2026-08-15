#!/usr/bin/env python3
"""Download/load Gemma 3 27B-IT and Llama 3.3 70B-IT per docs/MODEL_ACQUISITION.md
sections 1-3.

This is a thin CLI over the patterns in that doc, not a new download mechanism —
read the doc before running, especially section 2 (hardware reality check: Llama
3.3 70B-IT is HPC-only, Gemma fits Colab in 4-bit) and the quantization caution
(4-bit is fine for Step 0 generation, NOT valid for Steps 1-3 activation work).

Usage:
    # Full bf16 snapshot to local disk (HPC; Gemma only recommended on Colab disk)
    python scripts/download_models.py snapshot --model gemma-3-27b-it --out models/

    # Quantized in-memory load for interactive Step 0 use on Colab
    python scripts/download_models.py load --model gemma-3-27b-it --quantize 4bit

Requires HF_TOKEN in the environment (see docs/MODEL_ACQUISITION.md section 1) and
both models' licenses accepted on huggingface.co while logged in as that account.
"""
from __future__ import annotations

import argparse
import os
import sys

MODEL_HF_IDS = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "llama-3.3-70b-it": "meta-llama/Llama-3.3-70B-Instruct",
}

# google/gemma-3-27b-it's HF pipeline_tag is image-text-to-text (Gemma 3 is
# multimodal) — plain AutoModelForCausalLM may not be the right entry point
# depending on your transformers version. Verified live 2026-08-15 (see
# docs/MODEL_ACQUISITION.md "Implementation status"); re-check at load time
# since transformers' handling of this changes across releases.
GEMMA3_NOTE = (
    "google/gemma-3-27b-it is a multimodal (image-text-to-text) checkpoint. "
    "If AutoModelForCausalLM.from_pretrained() errors or loads a vision tower "
    "you don't need, try Gemma3ForConditionalGeneration (or whatever the "
    "text-only equivalent is in your installed transformers version) instead — "
    "check `AutoConfig.from_pretrained(hf_id).architectures` first."
)


def _require_hf_token() -> None:
    if not os.environ.get("HF_TOKEN"):
        sys.exit(
            "HF_TOKEN not set. Per docs/MODEL_ACQUISITION.md section 1:\n"
            "  export HF_TOKEN=hf_xxxxxxxx\n"
            "  huggingface-cli login --token $HF_TOKEN\n"
            "(On Colab: store as a Colab secret, not in the notebook body.)"
        )


def cmd_snapshot(model_key: str, out_dir: str) -> None:
    _require_hf_token()
    from huggingface_hub import snapshot_download

    hf_id = MODEL_HF_IDS[model_key]
    if model_key == "llama-3.3-70b-it":
        print(
            "WARNING: full bf16 snapshot of Llama 3.3 70B-IT is ~140GB. Per "
            "docs/MODEL_ACQUISITION.md section 2/3, this is HPC-only — do not "
            "attempt on Colab disk quota. Ctrl-C now if you're on Colab.",
            file=sys.stderr,
        )
    local_dir = f"{out_dir.rstrip('/')}/{model_key}"
    print(f"Downloading {hf_id} -> {local_dir} ...")
    snapshot_download(repo_id=hf_id, local_dir=local_dir, token=True)
    print("Done.")


def cmd_load(model_key: str, quantize: str) -> None:
    """Interactive/in-memory load, matching docs/MODEL_ACQUISITION.md section 3's
    quantized-load pattern. Returns nothing — this is meant to be imported and
    called from a notebook/REPL, or run standalone just to confirm the load
    succeeds (prints model/tokenizer summary, then exits)."""
    _require_hf_token()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = MODEL_HF_IDS[model_key]
    if model_key == "gemma-3-27b-it":
        print(GEMMA3_NOTE)

    kwargs = {"device_map": "auto", "token": True}
    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        if model_key != "gemma-3-27b-it":
            print(
                "NOTE: 4-bit quantization is only recommended for Gemma 3 27B-IT "
                "on Colab per docs/MODEL_ACQUISITION.md section 2. For Llama 3.3 "
                "70B-IT, prefer bf16 on HPC.",
                file=sys.stderr,
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        sys.exit(f"Unknown --quantize value: {quantize!r} (use 4bit or bf16)")

    if quantize == "4bit" and model_key != "gemma-3-27b-it":
        # Caution from docs/MODEL_ACQUISITION.md section 3: quantized activations
        # are not a valid substrate for Steps 1-3's causal claims. This CLI only
        # does generation-capable loads (Step 0); Steps 1-3 need the
        # ActivationCapableBackend path in src/refusal_mechinterp/models/backend.py.
        print(
            "REMINDER: quantized loads are for Step 0 behavioral screening only. "
            "Steps 1-3 (path patching, refusal direction, SAE work) require bf16.",
            file=sys.stderr,
        )

    print(f"Loading {hf_id} (quantize={quantize}) ...")
    tok = AutoTokenizer.from_pretrained(hf_id, token=True)
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    print(f"Loaded. Tokenizer vocab size: {tok.vocab_size}. "
          f"Model class: {type(model).__name__}. "
          f"Num params (approx): {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="Full weights snapshot to local disk")
    snap.add_argument("--model", required=True, choices=list(MODEL_HF_IDS))
    snap.add_argument("--out", default="models/")

    load = sub.add_parser("load", help="In-memory load (e.g. Colab interactive use)")
    load.add_argument("--model", required=True, choices=list(MODEL_HF_IDS))
    load.add_argument("--quantize", default="bf16", choices=["4bit", "bf16"])

    args = ap.parse_args()
    if args.command == "snapshot":
        cmd_snapshot(args.model, args.out)
    elif args.command == "load":
        cmd_load(args.model, args.quantize)


if __name__ == "__main__":
    main()
