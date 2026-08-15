#!/usr/bin/env python3
"""Suggest real xstest/or-bench candidates for the UNRESOLVED tier-(c) placeholders
that scripts/build_matched_triples.py leaves behind (see docs/DATASET_PLAN.md —
tier (c) must be sourced unmodified, never authored).

This SUGGESTS candidates by embedding similarity (sentence-transformers) between
each placeholder's topic/notes text and every seed-file prompt; it does NOT
auto-assign a tier-c item. Per docs/DATASET_PLAN.md validation check 2 (lexical
overlap) and check 3 (manual read-through), a human still has to pick the actual
match and confirm it shares the right focus-phrase family — this script just
narrows ~1000+ seed prompts down to a top-N shortlist per triple.

Usage:
    python scripts/resolve_tier_c.py --data data/matched_triples --seeds data/seeds \
        --out results/tier_c_candidates.csv --top-k 5

Requires `sentence-transformers` (see docs/MODEL_ACQUISITION.md section 5). This is
a CPU-friendly embedding model (all-MiniLM-L6-v2 by default) — no GPU/gated-model
access needed, unlike everything else in scripts/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_unresolved_triples(data_dir: Path) -> list[dict]:
    unresolved = []
    for path in sorted(data_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item["tier"] == "c" and not item.get("prompt_text"):
                    unresolved.append(item)
    return unresolved


def load_seed_pool(seeds_dir: Path) -> list[dict]:
    pool = []
    for path in sorted(seeds_dir.glob("*.jsonl")):
        with path.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("prompt") or row.get("text") or row.get("question")
                if not text:
                    continue
                pool.append({"source_file": path.name, "source_id": str(row.get("id", i)), "text": text})
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/matched_triples"))
    ap.add_argument("--seeds", type=Path, default=Path("data/seeds"))
    ap.add_argument("--out", type=Path, default=Path("results/tier_c_candidates.csv"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    unresolved = load_unresolved_triples(args.data)
    print(f"Found {len(unresolved)} unresolved tier-(c) placeholders.")
    if not unresolved:
        print("Nothing to do.")
        return

    seed_pool = load_seed_pool(args.seeds)
    if not seed_pool:
        raise SystemExit(
            f"No seed prompts found in {args.seeds} — run scripts/fetch_seed_datasets.py first."
        )
    print(f"Seed pool: {len(seed_pool)} candidate prompts.")

    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        raise SystemExit(
            "sentence-transformers not installed. `pip install sentence-transformers` first "
            "(this is CPU-friendly, no GPU/gated-model access needed)."
        )

    model = SentenceTransformer(args.embedding_model)

    # Query text = the placeholder's topic (from notes, since prompt_text is None)
    # plus the tier-(a) item's focus phrase for the same triple, to bias toward
    # the right vocabulary family (docs/DATASET_PLAN.md validation check 2).
    triples_by_id: dict[str, dict] = {}
    for path in sorted(args.data.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    triples_by_id.setdefault(item["matched_triple_id"], {})[item["tier"]] = item

    seed_texts = [row["text"] for row in seed_pool]
    seed_embeddings = model.encode(seed_texts, convert_to_tensor=True, show_progress_bar=True)

    rows_out = []
    for placeholder in unresolved:
        triple = triples_by_id.get(placeholder["matched_triple_id"], {})
        tier_a = triple.get("a", {})
        query = f"{placeholder.get('notes', '')} {tier_a.get('focus', '')} {tier_a.get('prompt_text', '')}"
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, seed_embeddings)[0]
        top_indices = scores.argsort(descending=True)[: args.top_k]
        for rank, idx in enumerate(top_indices, start=1):
            candidate = seed_pool[int(idx)]
            rows_out.append({
                "matched_triple_id": placeholder["matched_triple_id"],
                "category": placeholder["category"],
                "rank": rank,
                "score": float(scores[int(idx)]),
                "candidate_source_file": candidate["source_file"],
                "candidate_source_id": candidate["source_id"],
                "candidate_text": candidate["text"],
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} candidate rows ({args.top_k}/placeholder) -> {args.out}")
    print(
        "Next: review args.out by hand, pick the best candidate per triple, and "
        "update the corresponding tier-(c) row in data/matched_triples/*.jsonl with "
        "its real prompt_text/source/source_id — this script only shortlists, it "
        "does not write back to the dataset."
    )


if __name__ == "__main__":
    main()
