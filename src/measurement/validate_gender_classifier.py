#!/usr/bin/env python3
"""Agreement of the MiVOLO readout with the gender named in the calibration prompts.

Usage (mivolo env):
    python src/measurement/validate_gender_classifier.py --manifest data/main_run/image_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from classify_depicted_gender import add_model_args, build_predictor, people_indices, resolve_single_person

# Maps the corpus's own condition labels onto what MiVOLO returns ("male"/"female").
TRUTH_MAP = {
    "male_explicit": "male",
    "female_explicit": "female",
    "male": "male",
    "female": "female",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV with an image_id column; <image_id>.png expected next to it.")
    parser.add_argument("--truth-col", type=str, default="condition",
                        help="Manifest column holding the prompted gender (see TRUTH_MAP).")
    parser.add_argument("--group-cols", type=str, default="variety",
                        help="Manifest columns to break the agreement rate down by.")
    add_model_args(parser)
    args = parser.parse_args()

    import cv2
    predictor = build_predictor(args)

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    image_dir = args.manifest.parent
    group_cols = [c.strip() for c in args.group_cols.split(",")]

    rows = [r for r in rows if TRUTH_MAP.get(r.get(args.truth_col, "").strip())]
    if not rows:
        raise SystemExit(f"No rows with a known value in --truth-col '{args.truth_col}'. "
                         f"Known values: {sorted(TRUTH_MAP)}")

    status_counts: dict[tuple, Counter] = defaultdict(Counter)
    agree: dict[tuple, list[bool]] = defaultdict(list)
    disagreements = []
    out_rows = []

    for row in rows:
        truth = TRUTH_MAP[row[args.truth_col].strip()]
        key = tuple(row.get(c, "?") for c in group_cols)
        rec = dict(row)
        rec.update({"truth": truth, "n_people": None, "readout_status": None,
                    "gender": None, "face_available": None, "agrees": None, "error": ""})

        img = cv2.imread(str(image_dir / f"{row['image_id']}.png"))
        if img is None:
            rec["error"] = "unreadable_or_missing"
            status_counts[key]["error"] += 1
        else:
            try:
                detected, _ = predictor.recognize(img)
                people, _ = people_indices(detected)
                status, gender, _age, face_available = resolve_single_person(detected)
                rec.update({"n_people": len(people), "readout_status": status,
                            "gender": gender, "face_available": face_available})
                status_counts[key][status] += 1
                if status == "single" and gender is not None:
                    ok = gender.lower() == truth
                    rec["agrees"] = ok
                    agree[key].append(ok)
                    if not ok:
                        disagreements.append(rec)
            except Exception as exc:  # one bad image must not kill the run
                rec["error"] = repr(exc)
                status_counts[key]["error"] += 1
        out_rows.append(rec)

    out_path = image_dir / "mivolo_validity.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n=== readout status by ({', '.join(group_cols)}) ===")
    for key in sorted(status_counts):
        counts = ", ".join(f"{s}={n}" for s, n in sorted(status_counts[key].items()))
        print(f"  {'/'.join(key):40s} {counts}")

    print(f"\n=== agreement with the prompted gender, single-person images only ===")
    total_ok = total_n = 0
    for key in sorted(agree):
        oks = agree[key]
        total_ok += sum(oks)
        total_n += len(oks)
        print(f"  {'/'.join(key):40s} {sum(oks)}/{len(oks)}")
    if total_n:
        print(f"  {'TOTAL':40s} {total_ok}/{total_n} ({total_ok / total_n:.3f})")

    print(f"\n=== disagreements ===")
    if not disagreements:
        print("  none")
    for rec in disagreements:
        print(f"  {rec['image_id']}: prompted {rec['truth']}, read {rec['gender']}, "
              f"face_available={rec['face_available']}")

    print(f"\nDone. Per-image results at {out_path}")


if __name__ == "__main__":
    main()
