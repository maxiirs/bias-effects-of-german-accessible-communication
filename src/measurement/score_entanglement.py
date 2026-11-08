#!/usr/bin/env python3
"""Entanglement score of every test image, relative to the anchors of its own scene.

Usage: python src/measurement/score_entanglement.py --run-dir data/main_run
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gender_direction import (  # noqa: E402
    MIN_ANCHORS, ROLE_MALE, anchor_means, direction_of, load_calibration, write_csv,
)

OUT_NAME = "entanglement_scores.csv"
SELF_CHECK_TOL = 1e-4
DESIGN_COLS = ["image_id", "profession", "stereotype_direction", "stereotype_value_de",
               "scene_label", "condition", "variety", "seed"]


def load_test(run_dir: Path) -> list[dict]:
    """Test grid rows with their single-stream representations."""
    with (run_dir / "image_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["set"] == "test" and r["scene_label"] != "low_context"]
    if not rows:
        raise SystemExit(f"no test rows in {run_dir / 'image_manifest.csv'}")
    items = []
    for row in rows:
        path = run_dir / f"reps_{row['image_id']}.pt"
        if not path.exists():
            raise SystemExit(f"missing representations: {path}")
        reps = torch.load(path, map_location="cpu", weights_only=True)
        items.append({"row": {c: row[c] for c in DESIGN_COLS},
                      "reps": {k: v.squeeze(0).float() for k, v in reps.items()
                               if k.startswith("single_")}})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=Path("data/main_run"))
    parser.add_argument("--person-filter", type=Path, default=None,
                        help="Validity-check CSV naming the single-person calibration images; "
                             "defaults to <run-dir>/mivolo_validity.csv, as in "
                             "the direction validation.")
    args = parser.parse_args()

    calib = load_calibration(args.run_dir, args.person_filter
                             or args.run_dir / "mivolo_validity.csv")
    test = load_test(args.run_dir)
    varieties = sorted({it["variety"] for it in calib})
    layers = sorted((k for k in calib[0]["reps"] if k.startswith("single_")),
                    key=lambda k: int(k.split("_")[1]))
    print(f"{len(test)} test generations, {len(calib)} single-person anchors, "
          f"{len(layers)} single-stream blocks, varieties {varieties}")

    out, worst, flipped = [], 0.0, []
    for variety in varieties:
        anchors = [it for it in calib if it["variety"] == variety]
        items = [it for it in test if it["row"]["variety"] == variety]
        for layer in layers:
            g = direction_of(anchors, layer)
            gg = float(g @ g)
            means = anchor_means(anchors, layer)
            units, gaps = {}, {}
            for key, (m, f, _nm, _nf) in means.items():
                unit = float((m - f) @ g)
                gaps[key] = round(unit / gg, 4)
                if unit > 0:
                    units[key] = unit
                else:
                    flipped.append((layer, variety, key, gaps[key]))

            # Self-check on the anchors: mean score exactly 1 (male) and 0 (female) per scene.
            per_pole: dict[tuple, list[float]] = defaultdict(list)
            for it in anchors:
                key = (it["profession"], it["scene"])
                if key in units:
                    f = means[key][1]
                    per_pole[(key, it["role"])].append(
                        float((it["reps"][layer] - f) @ g) / units[key])
            for (key, role), vals in per_pole.items():
                target = 1.0 if role == ROLE_MALE else 0.0
                worst = max(worst, abs(sum(vals) / len(vals) - target))

            for it in items:
                row = it["row"]
                key = (row["profession"], row["scene_label"])
                if key in means:
                    _m, f, nm, nf = means[key]
                    score = (round(float((it["reps"][layer] - f) @ g) / units[key], 6)
                             if key in units else "")
                    gap = gaps[key]
                else:
                    score, nm, nf, gap = "", 0, 0, ""
                out.append({**row, "layer": layer, "score": score,
                            "n_male_anchors": nm, "n_female_anchors": nf, "anchor_gap": gap})

    print(f"self-check: largest deviation of an anchor mean from 0 or 1 is {worst:.2e}")
    if worst > SELF_CHECK_TOL:
        raise SystemExit(f"self-check failed (tolerance {SELF_CHECK_TOL}); nothing written")

    out_path = args.run_dir / OUT_NAME
    write_csv(out_path, out)

    first = layers[0]
    lacking = Counter(r["variety"] for r in out if r["layer"] == first and r["anchor_gap"] == "")
    for v in varieties:
        n = sum(1 for it in test if it["row"]["variety"] == v)
        print(f"  {v}: {n - lacking[v]} generations in scenes with at least {MIN_ANCHORS} anchors "
              f"per pole, {lacking[v]} without")
    unscored = sorted({(r["variety"], r["profession"], r["scene_label"])
                       for r in out if r["layer"] == first and r["anchor_gap"] == ""})
    for v, prof, scene in unscored:
        print(f"    scene without anchors: {v} {prof}/{scene}")
    if flipped:
        print(f"\n{len(flipped)} scene-block cells in which the male anchors do not lie above the "
              f"female ones along g; no score there, anchor_gap records the value:")
        for layer, v, (prof, scene), gap in flipped:
            print(f"    {layer:>9s} {v} {prof}/{scene}: anchor gap {gap:.3f}")
    print(f"\nwritten: {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
