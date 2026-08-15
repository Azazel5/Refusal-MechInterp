#!/usr/bin/env python3
"""Pull OR-Bench and XSTest public subsets into data/seeds/ as JSONL.

This is a data-acquisition step only (no model inference, no API keys needed) —
both datasets are public on the Hugging Face Hub. Not run automatically as part of
scaffolding; run it explicitly once you're ready to build the real dataset:

    python scripts/fetch_seed_datasets.py --out data/seeds/

Requires: `pip install datasets` (see requirements.txt).

Sources:
- OR-Bench: https://huggingface.co/datasets/bench-llms/or-bench
  (or-bench-80k main set + or-bench-hard-1k for the sharper over-refusal subset)
- XSTest: https://huggingface.co/datasets/natolambert/xstest-v2-copy
  (or the original Rottger et al. 2023 release — check current canonical HF id
  before running, HF ids for this dataset have moved before)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fetch_or_bench(out_dir: Path) -> None:
    from datasets import load_dataset

    # or-bench-hard-1k: the subset specifically curated to trigger over-refusal on
    # otherwise-benign prompts — this is the one relevant to tiers (a)/(b).
    hard = load_dataset("bench-llms/or-bench", "or-bench-hard-1k", split="train")
    with (out_dir / "or_bench_hard_1k.jsonl").open("w") as f:
        for row in hard:
            f.write(json.dumps(dict(row)) + "\n")

    # Full or-bench-80k for the control/unsafe pull in category 4 (paired with its
    # "or-bench-toxic" companion set for genuinely unsafe prompts).
    toxic = load_dataset("bench-llms/or-bench", "or-bench-toxic", split="train")
    with (out_dir / "or_bench_toxic.jsonl").open("w") as f:
        for row in toxic:
            f.write(json.dumps(dict(row)) + "\n")


def fetch_xstest(out_dir: Path) -> None:
    from datasets import load_dataset

    ds = load_dataset("natolambert/xstest-v2-copy", split="train")
    with (out_dir / "xstest_v2.jsonl").open("w") as f:
        for row in ds:
            f.write(json.dumps(dict(row)) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/seeds"))
    ap.add_argument("--skip-or-bench", action="store_true")
    ap.add_argument("--skip-xstest", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.skip_or_bench:
        print("Fetching OR-Bench...")
        fetch_or_bench(args.out)
    if not args.skip_xstest:
        print("Fetching XSTest...")
        fetch_xstest(args.out)

    print(f"Done. Seed files written to {args.out}/")


if __name__ == "__main__":
    main()
