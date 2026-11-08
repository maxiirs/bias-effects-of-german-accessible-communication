#!/usr/bin/env python3
"""Writes the prompts of the corpus from corpus_spec: GSL to data/corpus/sd_prompts.csv, GAL to
data/corpus/ls_prompts.csv.

Usage: python src/corpus/build_prompts.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import corpus_spec as spec

OUTPUTS = {
    Path("data/corpus/sd_prompts.csv"): spec.make_rows,
    Path("data/corpus/ls_prompts.csv"): spec.make_ls_rows,
}


def main() -> None:
    for path, make_rows in OUTPUTS.items():
        rows = make_rows()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=spec.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        sets = ", ".join(f"{n} {s}" for s, n in Counter(r["set"] for r in rows).items())
        print(f"{path}: {len(rows)} rows ({sets})")


if __name__ == "__main__":
    main()
