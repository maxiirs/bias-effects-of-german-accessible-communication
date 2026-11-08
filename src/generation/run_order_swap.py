#!/usr/bin/env python3
"""Pair-form prompts with the masculine conjunct first, seed-matched against the female-first images of
the main run.

Usage (GPU):
    python src/generation/run_order_swap.py --plan
    python src/generation/run_order_swap.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "generation"))
sys.path.insert(0, str(REPO / "src" / "corpus"))

import corpus_spec  # noqa: E402
from forward_hooks import (  # noqa: E402
    GUIDANCE_SCALE, IMAGE_SIZE, NUM_INFERENCE_STEPS, load_pipeline_with_hooks, make_logger,
)
from token_alignment import actor_mentions  # noqa: E402

PROMPT_FILES = ("data/corpus/ls_prompts.csv", "data/corpus/sd_prompts.csv")
MAIN_RUN = REPO / "data" / "main_run"
PROFESSIONS = ("mechaniker", "florist")
SECONDS_PER_IMAGE = 18.5  # main run mean, same pipeline and hooks

MANIFEST_FIELDS = [
    "image_id", "profession", "stereotype_direction", "scene_label", "condition", "variety",
    "conjunct_order", "seed", "prompt_text", "female_first_image_id", "female_first_prompt",
]


def swap_conjuncts(pair: str) -> str:
    """'Eine Floristin oder ein Florist' -> 'Ein Florist oder eine Floristin'."""
    first, sep, second = pair.partition(" oder ")
    assert sep and first.startswith("Eine ") and second.startswith("ein "), pair
    return f"Ein {second[len('ein '):]} oder eine {first[len('Eine '):]}"


def swapped_prompt(row: dict) -> str:
    """The row's prompt with every mention of the actor in male-first order."""
    female_first = corpus_spec.ACTORS[row["profession"]]["pair_form"][0]
    male_first = swap_conjuncts(female_first)
    text = row["prompt_text"]
    mentions = actor_mentions(corpus_spec, row)  # asserts the prompt opens with the first mention
    out = text.replace(female_first, male_first, 1)
    if len(mentions) > 1:
        ff_def = corpus_spec.definite_actor(female_first)
        assert out.count(ff_def) == len(mentions) - 1, (row["image_id"], text)
        out = out.replace(ff_def, corpus_spec.definite_actor(male_first))
    assert sorted(w.lower() for w in out.split()) == sorted(w.lower() for w in text.split()), \
        (row["image_id"], text, out)
    assert out != text, row["image_id"]
    return out


def build_stimuli(professions: set[str], seeds: set[int] | None) -> list[dict]:
    rows: list[dict] = []
    for path in PROMPT_FILES:
        with (REPO / path).open(newline="", encoding="utf-8") as handle:
            rows.extend(r for r in csv.DictReader(handle)
                        if r["set"] == "test" and r["condition"] == "pair_form"
                        and r["scene_label"] != "low_context"
                        and r["profession"] in professions
                        and (seeds is None or int(r["seed"]) in seeds))
    rows.sort(key=lambda r: (r["variety"], r["profession"], r["scene_label"], int(r["seed"])))
    return [{
        "image_id": (f"{r['profession']}_{r['scene_label']}_pair_form_male_first_"
                     f"{r['variety']}_seed{r['seed']}"),
        "profession": r["profession"], "stereotype_direction": r["stereotype_direction"],
        "scene_label": r["scene_label"], "condition": "pair_form", "variety": r["variety"],
        "conjunct_order": "male_first", "seed": int(r["seed"]), "prompt_text": swapped_prompt(r),
        "female_first_image_id": r["image_id"], "female_first_prompt": r["prompt_text"],
    } for r in rows]


def verify_against_main_run(stimuli: list[dict], need_images: bool) -> None:
    """Fail loudly if a reference is not what the main run generated, or, when generating, if its image
    is missing.
    """
    with (MAIN_RUN / "image_manifest.csv").open(newline="", encoding="utf-8") as handle:
        main = {r["image_id"]: r["prompt_text"] for r in csv.DictReader(handle)}
    for st in stimuli:
        ref = st["female_first_image_id"]
        if main.get(ref) != st["female_first_prompt"]:
            raise SystemExit(f"main-run manifest does not match the corpus for {ref}:\n"
                             f"  manifest: {main.get(ref)!r}\n  corpus:   {st['female_first_prompt']!r}")
        if need_images and not (MAIN_RUN / f"{ref}.png").exists():
            raise SystemExit(f"missing female-first reference image {MAIN_RUN / f'{ref}.png'}")


def generate(pipe, device: str, prompt: str, seed: int):
    return pipe(
        prompt=prompt, guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
        height=IMAGE_SIZE, width=IMAGE_SIZE, max_sequence_length=256,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]


def print_plan(stimuli: list[dict], out_dir: Path) -> None:
    done = sum((out_dir / f"{st['image_id']}.png").exists() for st in stimuli)
    print(f"{len(stimuli)} swapped images, {done} already on disk, "
          f"~{(len(stimuli) - done + 1) * SECONDS_PER_IMAGE / 60:.0f} min including the "
          f"reproduction check")
    seen = set()
    for st in stimuli:
        key = (st["variety"], st["profession"], st["scene_label"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  {st['variety']} {st['profession']}/{st['scene_label']}")
        print(f"    female first: {st['female_first_prompt']}")
        print(f"    male first:   {st['prompt_text']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPO / "data" / "order_swap")
    parser.add_argument("--professions", type=str, default=",".join(PROFESSIONS))
    parser.add_argument("--seeds", type=str, default="",
                        help="Comma-separated subset; default all five corpus seeds.")
    parser.add_argument("--plan", action="store_true",
                        help="Print the swapped prompts and the cost, generate nothing.")
    args = parser.parse_args()

    professions = {p.strip() for p in args.professions.split(",")}
    unknown = professions - set(corpus_spec.PROFESSIONS)
    if unknown:
        raise SystemExit(f"unknown professions: {sorted(unknown)}")
    seeds = {int(s) for s in args.seeds.split(",")} if args.seeds else None
    stimuli = build_stimuli(professions, seeds)
    out_dir = Path(args.output_dir)
    verify_against_main_run(stimuli, need_images=not args.plan)
    if args.plan:
        print_plan(stimuli, out_dir)
        return

    log = make_logger(out_dir)
    log({"event": "order_swap_plan", "n_images": len(stimuli), "professions": sorted(professions),
         "seeds": sorted({st["seed"] for st in stimuli}), "guidance_scale": GUIDANCE_SCALE,
         "num_inference_steps": NUM_INFERENCE_STEPS, "image_size": IMAGE_SIZE})
    pipe, device, _hooked, _single = load_pipeline_with_hooks(log, include_single_stream=True)

    # Reproduction check: one female-first reference regenerated against its stored image.
    import numpy as np
    from PIL import Image

    ref = stimuli[0]
    fresh = np.asarray(generate(pipe, device, ref["female_first_prompt"], ref["seed"]), dtype=np.int16)
    stored = np.asarray(Image.open(MAIN_RUN / f"{ref['female_first_image_id']}.png").convert("RGB"),
                        dtype=np.int16)
    diff = np.abs(fresh - stored)
    log({"event": "reproduction_check", "image_id": ref["female_first_image_id"],
         "identical": bool((diff == 0).all()), "max_abs_diff": int(diff.max()),
         "share_pixels_differing": round(float((diff.max(axis=-1) > 0).mean()), 6)})
    if not (diff == 0).all():
        print("WARNING: the stored reference does not reproduce exactly; the swap is seed-matched "
              "but not noise-identical to its reference. See the reproduction_check log record.")

    t_start = time.time()
    manifest_rows = []
    for i, st in enumerate(stimuli, start=1):
        path = out_dir / f"{st['image_id']}.png"
        if not path.exists():
            t0 = time.time()
            generate(pipe, device, st["prompt_text"], st["seed"]).save(path)
            log({"event": "order_swap_image", "index": i, "image_id": st["image_id"],
                 "wall_time_s": round(time.time() - t0, 2),
                 "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                 if device == "cuda" else None})
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        manifest_rows.append(st)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    log({"event": "order_swap_finished", "n_images": len(manifest_rows),
         "wall_time_s": round(time.time() - t_start, 1)})
    print(f"\nwrote {len(manifest_rows)} rows -> {manifest_path}")


if __name__ == "__main__":
    main()
