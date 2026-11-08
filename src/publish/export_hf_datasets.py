#!/usr/bin/env python3
"""Stages the main run as the two Hugging Face datasets and uploads them on request.

Usage:
    python src/publish/export_hf_datasets.py
    python src/publish/export_hf_datasets.py --upload
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "generation"))
sys.path.insert(0, str(REPO / "src" / "measurement"))

import dominance as dominance_metric  # noqa: E402
from forward_hooks import GRID_SIZE, GUIDANCE_SCALE, IMAGE_SIZE, NUM_INFERENCE_STEPS  # noqa: E402
from token_alignment import MODEL_ID  # noqa: E402

ORDER_SWAP = Path("data") / "order_swap"
NAMESPACE = "maxrs"
IMAGES_REPO = f"{NAMESPACE}/flux2-gal-gsl-images"
TENSORS_REPO = f"{NAMESPACE}/flux2-gal-gsl-tensors"
PUBLIC_REPO = "bias-effects-of-german-accessible-communication"
LANGUAGE_FORM = {"ls": "GAL", "sd": "GSL"}
TABLES = ["image_manifest.csv", "mivolo_results.csv", "entanglement_scores.csv",
          "dominance_results.csv", "word_shares_long.csv", "token_shares_long.csv"]
METADATA_COLUMNS = [
    "file_name", "image_id", "profession", "stereotype_direction", "stereotype_value_de",
    "scene_label", "scene_source", "condition", "language_form", "seed", "prompt_text", "n_words",
    "readout_status", "n_people", "depicted_gender", "face_detected",
    "entanglement_score_single_4", "entanglement_score_single_5",
    "dominance_dual_2", "dominance_dual_4",
]
ORDER_SWAP_COLUMNS = [
    "file_name", "image_id", "profession", "stereotype_direction", "scene_label", "condition",
    "language_form", "seed", "conjunct_order", "prompt_text", "female_first_image_id",
    "readout_status", "n_people", "depicted_gender", "face_detected",
]


def part_of(row: pd.Series) -> str:
    if row["set"] == "calibration":
        return "calibration"
    return "low_context" if row["scene_label"] == "low_context" else "test_grid"


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# --------------------------------------------------------------------------- images

def image_metadata(run: Path, dominance: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(run / "image_manifest.csv")
    readout = pd.read_csv(run / "mivolo_results.csv")[
        ["image_id", "readout_status", "n_people", "gender", "face_available"]]
    ent = pd.read_csv(run / "entanglement_scores.csv")
    ent = (ent[ent["layer"].isin(dominance_metric.ENT_BLOCKS)]
           .pivot(index="image_id", columns="layer", values="score")
           .add_prefix("entanglement_score_").reset_index())
    dom = (dominance.pivot(index="image_id", columns="layer_idx", values="D")
           .rename(columns=lambda b: f"dominance_dual_{b}").reset_index())
    meta = (manifest.merge(readout, on="image_id", how="left")
            .merge(ent, on="image_id", how="left").merge(dom, on="image_id", how="left"))
    meta["part"] = meta.apply(part_of, axis=1)
    meta["file_name"] = meta["image_id"] + ".png"
    meta["language_form"] = meta["variety"].map(LANGUAGE_FORM)
    meta["depicted_gender"] = meta["gender"]
    meta["face_detected"] = meta["face_available"].astype(str) == "True"
    return meta


def order_swap_metadata(swap_dir: Path) -> pd.DataFrame:
    """The conjunct-order diagnostic: the same prompts with the conjuncts swapped, each one seed-
    matched to a pair-form image of the test grid.
    """
    df = pd.read_csv(swap_dir / "mivolo_results.csv")
    df["file_name"] = df["image_id"] + ".png"
    df["language_form"] = df["variety"].map(LANGUAGE_FORM)
    df["depicted_gender"] = df["gender"]
    df["face_detected"] = df["face_available"].astype(str) == "True"
    return df


def stage_order_swap(swap_dir: Path, out: Path, swap: pd.DataFrame) -> None:
    folder = out / "order_swap"
    folder.mkdir(parents=True, exist_ok=True)
    for image_id in swap["image_id"]:
        src = swap_dir / f"{image_id}.png"
        if not src.exists():
            raise SystemExit(f"missing image {src}")
        link_or_copy(src, folder / f"{image_id}.png")
    swap[ORDER_SWAP_COLUMNS].to_csv(folder / "metadata.csv", index=False)
    print(f"  images/order_swap: {len(swap)} images")


def stage_images(run: Path, out: Path, meta: pd.DataFrame) -> None:
    for part, rows in meta.groupby("part"):
        folder = out / part
        folder.mkdir(parents=True, exist_ok=True)
        for image_id in rows["image_id"]:
            src = run / f"{image_id}.png"
            if not src.exists():
                raise SystemExit(f"missing image {src}")
            link_or_copy(src, folder / f"{image_id}.png")
        rows[METADATA_COLUMNS].to_csv(folder / "metadata.csv", index=False)
        print(f"  images/{part}: {len(rows)} images")


def images_card(meta: pd.DataFrame, swap: pd.DataFrame) -> str:
    n = meta["part"].value_counts()
    return f"""---
license: cc-by-4.0
language:
- de
pretty_name: FLUX.2 generations for GAL and GSL prompts
configs:
- config_name: test_grid
  data_files: "test_grid/*"
  default: true
- config_name: calibration
  data_files: "calibration/*"
- config_name: low_context
  data_files: "low_context/*"
- config_name: order_swap
  data_files: "order_swap/*"
---

# FLUX.2 generations for GAL and GSL prompts

The images of the main run of the master's thesis *Behind the Scenes of Text-to-Image Models: Bias
effects of German Accessible Communication* (HAW Hamburg, 2026), generated from prompts in German
Accessible Language (GAL) and German Standard Language (GSL). Every person shown is synthetic.
Design, code and analysis: [{PUBLIC_REPO}](https://github.com/maxiirs/{PUBLIC_REPO}).

| Config | Images | Content |
|---|---|---|
| `test_grid` | {n.get("test_grid", 0)} | 6 professions × 6 scenes × 4 gender forms × 2 language forms × 5 seeds |
| `calibration` | {n.get("calibration", 0)} | the same scenes with "Ein Mann" or "Eine Frau" as actor |
| `low_context` | {n.get("low_context", 0)} | the actor phrase alone, GSL only |
| `order_swap` | {len(swap)} | the pair form with its conjuncts swapped, two professions |

Generated with `{MODEL_ID}`, {NUM_INFERENCE_STEPS} inference steps, guidance scale {GUIDANCE_SCALE},
{IMAGE_SIZE} × {IMAGE_SIZE} pixels, in FP16 on a Tesla V100. Other hardware or BF16 gives different
images.

`entanglement_score_*` and `dominance_dual_*` are the two measures of the thesis at the named block.
`depicted_gender` is the label of MiVOLO (Kuprashevich and Tolstykh, 2023), a binary classifier that
its authors describe as estimating biological sex. It is set only where exactly one person is
detected, and it is a measurement on a synthetic image, not a gender identity.

Representations, attention maps and result tables: `{TENSORS_REPO}`.
"""


# --------------------------------------------------------------------------- tensors

def stage_tensors(run: Path, out: Path, dominance: pd.DataFrame) -> None:
    from safetensors.torch import save_file

    ids = pd.read_csv(run / "image_manifest.csv")["image_id"]
    for kind in ("reps", "maps"):
        folder = out / kind
        folder.mkdir(parents=True, exist_ok=True)
        done = 0
        for image_id in ids:
            dst = folder / f"{image_id}.safetensors"
            if dst.exists():
                continue
            src = run / f"{kind}_{image_id}.pt"
            if not src.exists():
                raise SystemExit(f"missing {src}")
            tensors = torch.load(src, map_location="cpu", weights_only=True)
            save_file({k: v.contiguous() for k, v in tensors.items()}, str(dst),
                      metadata={"image_id": image_id})
            done += 1
        print(f"  tensors/{kind}: {len(ids)} files, {done} converted now")
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    for name in TABLES:
        shutil.copy2(run / name, tables / name)
    dominance.to_csv(tables / "dominance_scores.csv", index=False)
    (out / "README.md").write_text(tensors_card(len(ids)), encoding="utf-8")


def tensors_card(n: int) -> str:
    return f"""---
license: cc-by-4.0
pretty_name: FLUX.2 internal representations and attention maps for GAL and GSL prompts
tags:
- explainability
- attention
- gender-bias
---

# Internal representations and attention maps

Companion to `{IMAGES_REPO}`: the model-internal quantities recorded while each of its {n} images
was generated, and every result table of the run. File names are the `image_id` of the images
dataset. Design, code and analysis: [{PUBLIC_REPO}](https://github.com/maxiirs/{PUBLIC_REPO}).

## `reps/`

One safetensors file per image. Keys `dual_0` to `dual_4` and `single_0` to `single_19`: the
output of that block's attention module at the last denoising step, averaged over the image
positions, shape [1, hidden size], float32.

## `maps/`

One safetensors file per image. Key `dual<b>_mention<k>`: the localisation map of the k-th mention
of the actor phrase at dual-stream block b and the last denoising step, summed over the mention's
tokens, on the {GRID_SIZE} x {GRID_SIZE} image token grid, float32.

## `tables/`

| File | Content |
|---|---|
| `image_manifest.csv` | design and prompt of every image |
| `mivolo_results.csv` | output-level readout of every image |
| `entanglement_scores.csv` | entanglement score per test image and single-stream block, with the anchor counts and the anchor separation of its scene |
| `dominance_scores.csv` | dominance per test image at dual-stream blocks 2 and 4 |
| `dominance_results.csv` | the run's estimands for the actor phrase per image and dual-stream block |
| `word_shares_long.csv`, `token_shares_long.csv` | attention share of every word and every token of the prompt per image and dual-stream block |
"""


# --------------------------------------------------------------------------- upload

def upload(folder: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise SystemExit(f"{repo_id} exists and is public; refusing to upload into it")
    api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(folder))
    print(f"  uploaded {folder} -> https://huggingface.co/datasets/{repo_id} (private)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=REPO / "data" / "main_run")
    parser.add_argument("--swap-dir", type=Path, default=REPO / ORDER_SWAP)
    parser.add_argument("--stage-dir", type=Path, default=REPO / "data" / "hf_export")
    parser.add_argument("--what", choices=("images", "tensors", "both"), default="both")
    parser.add_argument("--upload", action="store_true", help="Upload the staged folders, private.")
    args = parser.parse_args()

    readout = dominance_metric.load_readout(args.run_dir)
    dominance = dominance_metric.dominance(args.run_dir, set(readout["image_id"]), readout)
    print(f"dominance recomputed for {dominance['image_id'].nunique()} test images")

    targets = []
    if args.what in ("images", "both"):
        meta = image_metadata(args.run_dir, dominance)
        swap = order_swap_metadata(args.swap_dir)
        stage_images(args.run_dir, args.stage_dir / "images", meta)
        stage_order_swap(args.swap_dir, args.stage_dir / "images", swap)
        (args.stage_dir / "images" / "README.md").write_text(images_card(meta, swap), encoding="utf-8")
        targets.append((args.stage_dir / "images", IMAGES_REPO))
    if args.what in ("tensors", "both"):
        stage_tensors(args.run_dir, args.stage_dir / "tensors", dominance)
        targets.append((args.stage_dir / "tensors", TENSORS_REPO))
    print(f"staged in {args.stage_dir}")

    if args.upload:
        for folder, repo_id in targets:
            upload(folder, repo_id)
    else:
        print("nothing uploaded; check the staged cards and metadata, then rerun with --upload")


if __name__ == "__main__":
    main()
