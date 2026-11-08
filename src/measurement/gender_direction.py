"""Anchors and gender direction of the entanglement score, shared by score_entanglement.py and notebook
06.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import torch

TENSORS_REPO = "maxrs/flux2-gal-gsl-tensors"
TENSORS_REVISION = "9b31c8bdaf43a975dc3f7a16954fb517a736dea2"

MIN_ANCHORS = 3
ROLE_MALE = "male_explicit"
ROLE_FEMALE = "female_explicit"


def single_person_ids(path: Path) -> set[str]:
    """Image ids the output-level readout resolved to exactly one depicted person."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {r["image_id"] for r in csv.DictReader(handle)
                if r.get("readout_status") == "single"}


def load_calibration(run_dir: Path, person_filter: Path | None, load_reps=None) -> list[dict]:
    """Calibration rows with their per-layer representations."""
    with (run_dir / "image_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["set"] == "calibration"]
    if not rows:
        raise SystemExit(f"no calibration rows in {run_dir / 'image_manifest.csv'}")
    if person_filter is not None:
        if not person_filter.exists():
            raise SystemExit(f"missing {person_filter}; run validate_gender_classifier.py first, "
                             f"or pass --no-person-filter to estimate on every calibration image")
        keep = single_person_ids(person_filter)
        before = len(rows)
        rows = [r for r in rows if r["image_id"] in keep]
        print(f"single-person filter: {len(rows)} of {before} calibration images kept")
    items = []
    for row in rows:
        if load_reps is not None:
            reps = load_reps(row["image_id"])
        else:
            path = run_dir / f"reps_{row['image_id']}.pt"
            if not path.exists():
                raise SystemExit(f"missing representations: {path}")
            reps = torch.load(path, map_location="cpu", weights_only=True)
        items.append({
            "image_id": row["image_id"], "profession": row["profession"],
            "scene": row["scene_label"], "variety": row["variety"],
            "role": row["condition"], "seed": int(row["seed"]),
            "reps": {k: v.squeeze(0).float() for k, v in reps.items()},
        })
    return items


def anchor_means(items: list[dict], layer: str) -> dict:
    """(profession, scene) -> (male mean, female mean, n male, n female), only for scenes that keep at
    least MIN_ANCHORS anchors per pole.
    """
    groups: dict[tuple, dict] = defaultdict(lambda: {ROLE_MALE: [], ROLE_FEMALE: []})
    for it in items:
        groups[(it["profession"], it["scene"])][it["role"]].append(it["reps"][layer])
    out = {}
    for key, poles in groups.items():
        m, f = poles[ROLE_MALE], poles[ROLE_FEMALE]
        if len(m) >= MIN_ANCHORS and len(f) >= MIN_ANCHORS:
            out[key] = (torch.stack(m).mean(dim=0), torch.stack(f).mean(dim=0), len(m), len(f))
    return out


def direction_of(items: list[dict], layer: str) -> torch.Tensor | None:
    """Mean over the contributing scenes of the within-scene difference of anchor means."""
    means = anchor_means(items, layer)
    if not means:
        return None
    return torch.stack([m - f for m, f, _, _ in means.values()]).mean(dim=0)


def write_csv(path: Path, data: list[dict]) -> None:
    if not data:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def tensors(pattern: str) -> Path:
    """Root of the tensor dataset on Hugging Face, with the files matching pattern downloaded."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(TENSORS_REPO, repo_type="dataset", revision=TENSORS_REVISION,
                                  allow_patterns=[pattern]))


def reps_loader(pattern: str = "reps/*.safetensors"):
    """image_id -> representation dict"""
    from safetensors.torch import load_file

    folder = tensors(pattern) / "reps"
    return lambda image_id: load_file(folder / f"{image_id}.safetensors")
