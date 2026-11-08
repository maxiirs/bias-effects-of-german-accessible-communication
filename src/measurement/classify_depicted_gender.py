#!/usr/bin/env python3
"""Depicted gender of generated images with MiVOLO, under the single-person rule.

Usage (mivolo env):
    python src/measurement/classify_depicted_gender.py --manifest data/main_run/image_manifest.csv --group-cols profession,variety
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

MAIN_PROJECT = Path(__file__).resolve().parents[3] / "easy-language-gender-bias"  # checkout next to this repository


def people_indices(detected_objects) -> tuple[list, list]:
    """One entry per depicted person, never one per box: person boxes plus face boxes that have no
    matched person.
    """
    person_inds = list(detected_objects.get_bboxes_inds("person"))
    face_inds = list(detected_objects.get_bboxes_inds("face"))
    unmatched_faces = [fi for fi in face_inds
                       if detected_objects.face_to_person_map.get(fi) is None]
    return person_inds + unmatched_faces, unmatched_faces


def resolve_single_person(detected_objects) -> tuple[str, object, object, object]:
    """This thesis's readout rule, replacing the main project's dominant-person convention."""
    people, unmatched_faces = people_indices(detected_objects)
    if not people:
        return "none", None, None, False
    if len(people) > 1:
        return "multiple", None, None, None
    idx = people[0]
    face_of_person = next(
        (fi for fi, pi in detected_objects.face_to_person_map.items() if pi == idx), None
    )
    read_idx = idx if face_of_person is None else face_of_person
    face_available = face_of_person is not None or idx in unmatched_faces
    return "single", detected_objects.genders[read_idx], detected_objects.ages[read_idx], face_available


def count_genders(detected_objects) -> tuple[int, int]:
    """Ported from evaluate_mivolo_images.py: count person boxes + unmatched face-only boxes (genders
    sit on both the person and face index of the same person - double-count fix).
    """
    m_count = f_count = 0
    people, _ = people_indices(detected_objects)
    for idx in people:
        g = detected_objects.genders[idx]
        if g is not None and g.lower() == "male":
            m_count += 1
        elif g is not None:
            f_count += 1
    return m_count, f_count


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Shared by this script and validate_gender_classifier.py, so both name the same weights."""
    parser.add_argument("--mivolo-repo", type=Path, default=MAIN_PROJECT / "MiVOLO")
    parser.add_argument("--yolo-weights", type=Path, default=MAIN_PROJECT / "yolov8x_person_face.pt")
    parser.add_argument("--mivolo-weights", type=Path,
                        default=MAIN_PROJECT / "model_imdb_cross_person_4.22_99.46.pth.tar")
    parser.add_argument("--device", type=str, default="cuda")


def build_predictor(args):
    """Shared with the validity check, so both run the identical model setup."""
    sys.path.insert(0, str(args.mivolo_repo))
    from mivolo.predictor import Predictor

    class Config:  # the Predictor's expected attribute bag (as in the main project)
        detector_weights = str(args.yolo_weights)
        checkpoint = str(args.mivolo_weights)
        device = args.device
        half = True
        with_persons = True
        disable_faces = False
        draw = False

    print(f"Loading MiVOLO models on {args.device}...")
    return Predictor(Config())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV with an image_id column; <image_id>.png expected next to it.")
    parser.add_argument("--group-cols", type=str, default="role,variety",
                        help="Manifest columns for the summary crosstab.")
    add_model_args(parser)
    args = parser.parse_args()

    import cv2
    predictor = build_predictor(args)

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    image_dir = args.manifest.parent
    group_cols = [c.strip() for c in args.group_cols.split(",")]

    out_rows = []
    for row in rows:
        img_path = image_dir / f"{row['image_id']}.png"
        rec = dict(row)
        rec.update({"n_male": None, "n_female": None, "n_people": None,
                    "readout_status": None, "gender": None, "age": None,
                    "face_available": None, "mivolo_error": ""})
        img = cv2.imread(str(img_path))
        if img is None:
            rec["mivolo_error"] = "unreadable_or_missing"
        else:
            try:
                detected_objects, _ = predictor.recognize(img)
                m, f = count_genders(detected_objects)
                people, _ = people_indices(detected_objects)
                status, gender, age, face_available = resolve_single_person(detected_objects)
                rec.update({"n_male": m, "n_female": f, "n_people": len(people),
                            "readout_status": status, "gender": gender,
                            "age": round(float(age), 1) if age is not None else None,
                            "face_available": face_available})
            except Exception as exc:  # keep going; a single bad image must not kill the run
                rec["mivolo_error"] = repr(exc)
        out_rows.append(rec)
        print(f"{row['image_id']}: n_people={rec['n_people']} status={rec['readout_status']} "
              f"gender={rec['gender']} face={rec['face_available']} {rec['mivolo_error']}")

    out_path = image_dir / "mivolo_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n=== readout status by ({', '.join(group_cols)}) ===")
    status_tab: dict[tuple, Counter] = {}
    gender_tab: dict[tuple, Counter] = {}
    for rec in out_rows:
        key = tuple(rec.get(c, "?") for c in group_cols)
        status_tab.setdefault(key, Counter())[rec["readout_status"] or "error"] += 1
        if rec["readout_status"] == "single":
            gender_tab.setdefault(key, Counter())[rec["gender"] or "unknown"] += 1
    for key in sorted(status_tab):
        counts = ", ".join(f"{s}={n}" for s, n in sorted(status_tab[key].items()))
        print(f"  {'/'.join(key):50s} {counts}")

    print(f"\n=== gender on single-person images by ({', '.join(group_cols)}) ===")
    for key in sorted(gender_tab):
        counts = ", ".join(f"{g}={n}" for g, n in sorted(gender_tab[key].items()))
        print(f"  {'/'.join(key):50s} {counts}")
    print(f"\nDone. Results at {out_path}")


if __name__ == "__main__":
    main()
