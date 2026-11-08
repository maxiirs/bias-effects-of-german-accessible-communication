#!/usr/bin/env python3
"""Profession-stem candidates from the ELGEPA paired corpus, the input of the ELGEPA figure in notebook
01.

Usage: python src/corpus/mine_elgepa_candidates.py --paired-csv <actor_prompts_paired.csv>
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

PROFESSION_STEMS = [
    # contested/mixed (research plan category 3)
    "lehrer", "arzt", "ärzt", "mitarbeiter", "berater", "dolmetsch", "übersetz",
    # female-stereotyped (category 2)
    "pfleg", "erzieh", "sekretär", "betreu",
    # male-stereotyped (category 1)
    "mechanik", "bauarbeit", "handwerk", "polizist", "ingenieur", "anwalt", "fahrer", "richter",
    # broader service/occupational
    "helfer", "verkäufer", "koch", "köch", "friseur", "vermiet", "patient",
]

DEIXIS_PATTERNS = [
    r"^(Oder|Und|Dann|Dazu|Deshalb|Darum|Auch|Aber|Denn|Also)\b",  # continuation openers
    r"^(Das|Dies|Diese[srm]?|Bei diesen|Damit|Dabei|Davon|Dafür)\b",  # anaphoric openers
    r"\bzum Beispiel\b",  # list-item fragments
]


def looks_context_dependent(sentence: str) -> bool:
    return any(re.search(p, sentence.strip(), flags=re.IGNORECASE) for p in DEIXIS_PATTERNS[2:]) or any(
        re.match(p, sentence.strip()) for p in DEIXIS_PATTERNS[:2]
    )


def word_count(sentence: str) -> int:
    return len(sentence.split())


def mine(paired_csv: Path, output: Path) -> None:
    with paired_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    paired = [
        r for r in rows
        if r["has_simple"] == "1" and r["has_normal"] == "1"
        and r["simple_is_named_person"] != "1" and r["normal_is_named_person"] != "1"
    ]

    candidates = []
    seen: set[tuple[str, str]] = set()
    for r in paired:
        stem = next((s for s in PROFESSION_STEMS if s in r["actor_key"]), None)
        if stem is None:
            continue
        simple = r["simple_context_sentence"].strip()
        normal = r["normal_context_sentence"].strip()
        dedupe_key = (simple, normal)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append({
            "pair_id": r["pair_id"],
            "actor_key": r["actor_key"],
            "matched_stem": stem,
            "topic": r["topic"],
            "source": r["source"],
            "simple_final_frame": r["simple_final_frame"],
            "simple_sentence": simple,
            "normal_sentence": normal,
            "simple_word_count": word_count(simple),
            "normal_word_count": word_count(normal),
            "simple_context_dependent": int(looks_context_dependent(simple)),
            "normal_context_dependent": int(looks_context_dependent(normal)),
            "simple_contains_actor_stem": int(stem in simple.lower()),
            "normal_contains_actor_stem": int(stem in normal.lower()),
        })

    candidates.sort(key=lambda c: (c["matched_stem"], c["simple_word_count"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)

    stems_found = sorted({c["matched_stem"] for c in candidates})
    stems_missing = sorted(set(PROFESSION_STEMS) - set(stems_found))
    print(f"{len(paired)} fully-paired non-named rows -> {len(candidates)} deduped candidates -> {output}")
    print(f"stems with hits ({len(stems_found)}): {', '.join(stems_found)}")
    print(f"stems with ZERO paired rows ({len(stems_missing)}): {', '.join(stems_missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-csv", type=Path,
        default=Path(__file__).resolve().parents[3] / "easy-language-gender-bias/data/processed/actor_prompts_paired.csv",
    )
    parser.add_argument("--output", type=Path, default=Path("data/corpus/anchor_candidates_raw.csv"))
    args = parser.parse_args()
    mine(args.paired_csv, args.output)


if __name__ == "__main__":
    main()
