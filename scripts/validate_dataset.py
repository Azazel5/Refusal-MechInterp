#!/usr/bin/env python3
"""Dataset validation, per docs/DATASET_PLAN.md section 5.

Implements what's checkable without model/API access (checks 2 and 4). Checks 1
(ensemble moderation), 3 (manual read-through), and 5 (pilot behavioral run) are
NOT scripts — see the printed notes below for what each actually requires.

Usage:
    python scripts/validate_dataset.py --data data/matched_triples
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

_WORD_RE = re.compile(r"[a-z]{4,}")
_STOPWORDS = {
    "have", "been", "with", "that", "this", "your", "what", "should", "about",
    "week", "weeks", "days", "daily", "nightly", "than", "usual", "lately",
    "much", "some", "into", "over", "from", "will", "when", "asking",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def load_triples(data_dir: Path) -> dict[str, dict[str, dict]]:
    triples: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(data_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                triples[item["matched_triple_id"]][item["tier"]] = item
    return triples


def check_lexical_overlap(triples: dict[str, dict[str, dict]]) -> list[str]:
    """DATASET_PLAN.md check 2: (a)/(b) must share content-word overlap with
    their matched (c) — otherwise the triple isn't testing the conflation
    hypothesis. Skips triples whose (c) is unresolved (prompt_text is None)."""
    problems = []
    for triple_id, tiers in triples.items():
        a, b, c = tiers.get("a"), tiers.get("b"), tiers.get("c")
        if not (a and b and c):
            problems.append(f"{triple_id}: missing tier(s) — have {sorted(tiers)}")
            continue
        if not c.get("prompt_text"):
            continue  # unresolved placeholder, not a real check target yet
        words_a = _content_words(a["prompt_text"])
        words_c = _content_words(c["prompt_text"])
        overlap = words_a & words_c
        if len(overlap) < 1:
            problems.append(f"{triple_id}: no content-word overlap between tier (a) and (c) — {words_a} vs {words_c}")
    return problems


def check_inter_tier_distinctiveness(triples: dict[str, dict[str, dict]]) -> list[str]:
    """DATASET_PLAN.md check 4: framing should be present in (a), thin in (b),
    absent in (c). Structural proxy here: (a) must have a non-empty `focus`
    field describing explicit self-report/control framing; (b) must have a
    `focus` field too (thinner, by construction) but its prompt should be
    shorter/less detailed than (a)'s (a cheap proxy for "thinner framing" —
    flag for manual review rather than trust blindly)."""
    problems = []
    for triple_id, tiers in triples.items():
        a, b = tiers.get("a"), tiers.get("b")
        if not (a and b):
            continue
        if not a.get("focus"):
            problems.append(f"{triple_id}: tier (a) missing `focus` annotation")
        if not b.get("focus"):
            problems.append(f"{triple_id}: tier (b) missing `focus` annotation")
        if a.get("prompt_text") and b.get("prompt_text"):
            if len(b["prompt_text"]) >= len(a["prompt_text"]):
                problems.append(
                    f"{triple_id}: tier (b) prompt is not shorter than tier (a) "
                    "(length proxy for 'thinner framing') — review by hand, this "
                    "is a cheap proxy, not a real distinctiveness measure"
                )
    return problems


def check_unresolved(triples: dict[str, dict[str, dict]]) -> list[str]:
    unresolved = [tid for tid, tiers in triples.items()
                  if tiers.get("c") and not tiers["c"].get("prompt_text")]
    return unresolved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/matched_triples"))
    args = ap.parse_args()

    triples = load_triples(args.data)
    print(f"Loaded {len(triples)} matched-triple groups from {args.data}\n")

    print("--- Check 2: lexical overlap (a)/(c) ---")
    overlap_problems = check_lexical_overlap(triples)
    if overlap_problems:
        for p in overlap_problems:
            print(f"  FLAG: {p}")
    else:
        print("  No flags (or all tier-c unresolved — see below).")

    print("\n--- Check 4: inter-tier distinctiveness (structural proxy only) ---")
    tier_problems = check_inter_tier_distinctiveness(triples)
    if tier_problems:
        for p in tier_problems:
            print(f"  FLAG: {p}")
    else:
        print("  No flags.")

    unresolved = check_unresolved(triples)
    print(f"\n--- Unresolved tier-(c) placeholders: {len(unresolved)} ---")
    if unresolved:
        print("  " + ", ".join(unresolved))
        print("  These need a real xstest/or-bench sourced item before Step 0 can "
              "score them — see notes field in the source jsonl for the topic to match.")

    print("\n--- Checks NOT implemented here (per docs/DATASET_PLAN.md section 5) ---")
    print("  1. Ensemble moderation check: needs a moderation classifier (e.g. Llama")
    print("     Guard) wired up independent of the two target models — not done, no")
    print("     moderation API access configured yet.")
    print("  3. Manual read-through: a human pass over every row for realism + correct")
    print("     tier assignment. Not automatable by design.")
    print("  5. Pilot behavioral run: run experiments/step0_baseline/run_baseline.py on")
    print("     one full category (~20 triples) and confirm the false-positive gap")
    print("     appears before finishing full construction. Requires model access.")


if __name__ == "__main__":
    main()
