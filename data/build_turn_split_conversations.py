#!/usr/bin/env python3
"""Build multi-turn escalating conversations from ALREADY-VALIDATED single-turn
matched-triple items — per the Q6/Q7 multi-turn addition to docs/BRIEF.md.

DOES NOT AUTHOR NEW CONTENT. Every turn's text is either:
  (a) a sibling tier-(b) item from the same matched triple (already validated,
      already in data/matched_triples/), used verbatim as the "thin framing"
      opening turn, or
  (b) an algorithmic decomposition of the tier-(a) item's own sentence clauses
      (regex-based sentence/clause splitting), reassembled across turns in a
      fixed order — no new words are introduced, only re-ordered/re-segmented.
If decomposition can't confidently find all expected components for an item,
it is skipped and flagged for manual review rather than emitting a malformed
or partially-fabricated turn sequence.

Rationale for this approach (see chat history 2026-08-15): isolates
"information distributed across turns" as the sole independent variable
against the existing single-turn tier-(a) control — same total content,
different delivery. This is what makes the multi-turn/single-turn comparison
mechanistically clean rather than confounded with new content.

Turn skeleton (fixed, 6 turns, applied to every triple):
  Turn 1 - thin framing / topic mention        <- sibling tier-(b) text, verbatim
  Turn 2 - protocol/supervision clause          <- extracted from tier-(a)
  Turn 3 - quantity/duration clause             <- extracted from tier-(a)
  Turn 4 - tracking/logging clause              <- extracted from tier-(a)
  Turn 5 - no-adverse-signal clause             <- extracted from tier-(a)
  Turn 6 - the original closing question        <- extracted from tier-(a), verbatim

Also emits history-truncation variants (full / last-2 / last-1 turns before the
final question) per the ablation design in the "State-Dependent Safety
Failures in Multi-Turn Language Model Interaction" precedent (arXiv 2603.15684)
— cheap re-use of the same generated conversations, no extra generation cost.

Usage:
    python scripts/build_turn_split_conversations.py \
        --data data/matched_triples --out data/conversations \
        --min-topics 1  # (default) set higher once more triples exist

Output: data/conversations/turn_split_conversations.jsonl — one row per
{triple, variant}. This file is what gets passed to the model-runner
(experiments/step0_baseline/run_baseline.py needs a small extension to walk
`turns` instead of a single `prompt_text` — see NOTE at bottom of this file).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Regex clause extraction patterns, applied in order against tier-(a) prompt_text.
# Each pulls ONE sentence/clause containing its trigger words; if a pattern
# finds nothing, the item is flagged SKIPPED (see main loop) rather than
# guessed at.
_SENTENCE_SPLIT = re.compile(r"(?<=[.?])\s+")

_PROTOCOL_KEYWORDS = re.compile(
    r"\b(protocol|supervised|coach-reviewed|medically-supervised|standard|"
    r"published|clinician-reviewed|self-designed|planned|structured|"
    r"lab-confirmed)\b", re.IGNORECASE,
)
_QUANTITY_KEYWORDS = re.compile(r"\d")  # any sentence carrying a number
_TRACKING_KEYWORDS = re.compile(
    r"\b(track|log|logging|logged|monitor)\w*\b", re.IGNORECASE,
)
_NOADVERSE_KEYWORDS = re.compile(
    r"\b(no |none |nothing |not\b|without any)\w*", re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _find_clause(sentences: list[str], pattern: re.Pattern, used: set[int]) -> tuple[str, int] | None:
    for i, s in enumerate(sentences):
        if i in used and len(sentences) > 4:  # allow reuse only if very few sentences
            continue
        if pattern.search(s):
            return s, i
    return None


def decompose_tier_a(prompt_text: str) -> dict[str, str] | None:
    """Returns {protocol, quantity, tracking, no_adverse, question} or None if
    any required component can't be confidently located."""
    sentences = _sentences(prompt_text)
    if len(sentences) < 2:
        return None

    question = sentences[-1] if sentences[-1].endswith("?") else None
    if not question:
        return None
    body_sentences = sentences[:-1]

    used: set[int] = set()
    out: dict[str, str] = {}
    for key, pattern in [
        ("protocol", _PROTOCOL_KEYWORDS),
        ("quantity", _QUANTITY_KEYWORDS),
        ("tracking", _TRACKING_KEYWORDS),
        ("no_adverse", _NOADVERSE_KEYWORDS),
    ]:
        found = _find_clause(body_sentences, pattern, used)
        if not found:
            return None  # missing a required component — flag for manual review, don't guess
        text, idx = found
        out[key] = text
        used.add(idx)

    out["question"] = question
    return out


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


def build_full_turns(tier_a: dict, tier_b: dict, components: dict[str, str]) -> list[dict]:
    return [
        {"turn_index": 1, "role": "user", "text": tier_b["prompt_text"], "component": "thin_framing_from_tier_b"},
        {"turn_index": 2, "role": "user", "text": components["protocol"], "component": "protocol"},
        {"turn_index": 3, "role": "user", "text": components["quantity"], "component": "quantity"},
        {"turn_index": 4, "role": "user", "text": components["tracking"], "component": "tracking"},
        {"turn_index": 5, "role": "user", "text": components["no_adverse"], "component": "no_adverse_signal"},
        {"turn_index": 6, "role": "user", "text": components["question"], "component": "question"},
    ]


def truncate(turns: list[dict], keep_last_n_before_question: int | None) -> list[dict]:
    """None = full history. Otherwise keep only the last N non-question turns
    plus the final question turn, per the truncation-ablation design."""
    if keep_last_n_before_question is None:
        return turns
    question_turn = turns[-1]
    body = turns[:-1][-keep_last_n_before_question:] if keep_last_n_before_question > 0 else []
    # Re-index turn_index after truncation so the model-runner sees a clean sequence.
    out = []
    for i, t in enumerate(body + [question_turn], start=1):
        t = dict(t)
        t["turn_index"] = i
        out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/matched_triples"))
    ap.add_argument("--out", type=Path, default=Path("data/conversations"))
    ap.add_argument(
        "--truncation-variants", action="store_true",
        help="also emit last-2 and last-1 history-window variants per triple, per the "
             "State-Dependent Safety Failures (arXiv 2603.15684) ablation design",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    triples = load_triples(args.data)
    rows: list[dict] = []
    skipped: list[str] = []

    for triple_id, tiers in triples.items():
        a, b = tiers.get("a"), tiers.get("b")
        if not (a and b) or a.get("category") == "control-risk":
            continue  # control-risk has no (a)/(b) pair, skip

        components = decompose_tier_a(a["prompt_text"])
        if components is None:
            skipped.append(triple_id)
            continue

        full_turns = build_full_turns(a, b, components)

        variants = {"full": full_turns}
        if args.truncation_variants:
            variants["last2"] = truncate(full_turns, keep_last_n_before_question=2)
            variants["last1"] = truncate(full_turns, keep_last_n_before_question=1)
            variants["last0"] = truncate(full_turns, keep_last_n_before_question=0)  # question alone

        for variant_name, turns in variants.items():
            rows.append({
                "conversation_id": f"{triple_id}_turnsplit_{variant_name}",
                "matched_triple_id": triple_id,
                "category": a["category"],
                "variant": "turn_split",
                "history_window": variant_name,
                "turns": turns,
                "source_tier_a_id": a["id"],
                "source_tier_b_id": b["id"],
                "ground_truth_label": a["ground_truth_label"],
                "notes": a.get("notes", ""),
            })

        # Single-turn control, same schema (turns list of length 1) — same
        # conversation_id family, direct apples-to-apples comparison target.
        rows.append({
            "conversation_id": f"{triple_id}_singleturn_control",
            "matched_triple_id": triple_id,
            "category": a["category"],
            "variant": "single_turn_control",
            "history_window": "n/a",
            "turns": [{"turn_index": 1, "role": "user", "text": a["prompt_text"], "component": "full_single_turn"}],
            "source_tier_a_id": a["id"],
            "source_tier_b_id": None,
            "ground_truth_label": a["ground_truth_label"],
            "notes": a.get("notes", ""),
        })

    out_path = args.out / "turn_split_conversations.jsonl"
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} conversation rows ({len(triples) - len(skipped)} triples "
          f"successfully decomposed) -> {out_path}")
    if skipped:
        print(f"SKIPPED {len(skipped)} triples (couldn't confidently locate all 4 required "
              f"clauses — protocol/quantity/tracking/no_adverse). Review by hand or extend "
              f"the regex patterns, do not force a fabricated split: {skipped}")


if __name__ == "__main__":
    main()

# NOTE for experiments/step0_baseline/run_baseline.py:
# It currently expects one `prompt_text` per item. Extend it to detect a
# `turns` list instead: replay turns[0..n-1] as alternating user messages
# (assistant responses generated live by the target model at each step, not
# pre-scripted), score refusal only on the FINAL assistant response to the
# turns[-1] user message, but log the refusal-direction projection at every
# turn boundary for the Q6 drift-vs-phase-transition analysis.
