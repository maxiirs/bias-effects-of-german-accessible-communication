#!/usr/bin/env python3
"""Main run: generates every image of the corpus and stores attention shares, representations and
localisation maps.

Usage (GPU):
    python src/generation/run_full_corpus.py --plan
    python src/generation/run_full_corpus.py --output-dir data/main_run
"""
from __future__ import annotations

import argparse
import csv
import gc
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src" / "corpus"))

import corpus_spec  # noqa: E402
from forward_hooks import (  # noqa: E402
    GRID_SIZE, GUIDANCE_SCALE, HOOK_DATA, IMAGE_SIZE, NUM_INFERENCE_STEPS, SHARED_STATE,
    SINGLE_HOOK_DATA, load_pipeline_with_hooks, make_logger, mention_estimands,
    run_daam_dominance_probe, run_daam_probe, token_dominance_shares, word_dominance_shares,
)
from token_alignment import (  # noqa: E402
    actor_mentions, get_token_groups, get_word_groups, mention_token_spans, templated_prompt_alignment,
)
from main_run_store import (  # noqa: E402
    RESULTS_NAME, TOKENS_NAME, WORDS_NAME, RunWriter, load_state,
)

PROMPT_FILES = ("data/corpus/ls_prompts.csv", "data/corpus/sd_prompts.csv")
BYTES_PER_IMAGE = int(2.12 * 1024 ** 2)
BYTES_TABLES = 40 * 1024 ** 2             # the two long tables plus log.jsonl, generously


# --------------------------------------------------------------------------- work plan

def plan_key(row: dict) -> tuple:
    """Deterministic order: calibration cells first, then the test grid; inside each block by variety,
    profession, scene, condition, seed.
    """
    return (row["set"] != "calibration", row["variety"], row["profession"],
            row["scene_label"], row["condition"], int(row["seed"]))


def build_plan(seeds: set[int] | None = None, sets: set[str] | None = None,
               limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for path in PROMPT_FILES:
        with (REPO / path).open(newline="", encoding="utf-8") as handle:
            rows.extend(r for r in csv.DictReader(handle)
                        if (seeds is None or int(r["seed"]) in seeds)
                        and (sets is None or r["set"] in sets))
    rows.sort(key=plan_key)
    return rows[:limit] if limit else rows


# --------------------------------------------------------------------------- one image

def process_image(pipe, device: str, row: dict, hooked_count: int, single_count: int,
                  align_cache: dict, log) -> dict:
    """Generate one image, run both DAAM probes on every dual-stream layer, and return its artefacts
    and table rows.
    """
    prompt, image_id = row["prompt_text"], row["image_id"]
    mentions = actor_mentions(corpus_spec, row)
    if prompt not in align_cache:
        align_cache[prompt] = (*templated_prompt_alignment(pipe.tokenizer, prompt),
                               get_word_groups(pipe.tokenizer, prompt),
                               get_token_groups(pipe.tokenizer, prompt))
    offsets, template_offset, n_prompt_tokens, words, tokens = align_cache[prompt]
    mention_idxs = mention_token_spans(offsets, template_offset, prompt, mentions)
    prompt_token_idxs = list(range(template_offset, template_offset + n_prompt_tokens))

    t0 = time.time()
    image = pipe(
        prompt=prompt, guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
        height=IMAGE_SIZE, width=IMAGE_SIZE, max_sequence_length=256,
        generator=torch.Generator(device=device).manual_seed(int(row["seed"])),
    ).images[0]
    wall = time.time() - t0
    text_seq_len = SHARED_STATE["text_seq_len"]

    results, word_rows, token_rows, maps_out = [], [], [], {}
    for layer_idx in range(hooked_count):
        attn = pipe.transformer.get_submodule(SHARED_STATE["dual_layer_order"][layer_idx])
        dmap = run_daam_dominance_probe(attn, HOOK_DATA[layer_idx], text_seq_len, GRID_SIZE)
        lmap = run_daam_probe(attn, HOOK_DATA[layer_idx], text_seq_len, GRID_SIZE)
        if dmap is None or lmap is None:
            log({"event": "probe_missing", "image_id": image_id, "layer_idx": layer_idx,
                 "dominance_ok": dmap is not None, "daam_ok": lmap is not None})
            continue
        est, _detail = mention_estimands(dmap, mention_idxs, prompt_token_idxs, n_prompt_tokens)
        results.append({"image_id": image_id, "profession": row["profession"],
                        "scene_label": row["scene_label"], "condition": row["condition"],
                        "variety": row["variety"], "seed": int(row["seed"]),
                        "layer_idx": layer_idx, **est})
        word_rows.extend({"image_id": image_id, "layer_idx": layer_idx,
                          **w, "share": round(w["share"], 6)}
                         for w in word_dominance_shares(words, dmap))
        token_rows.extend({"image_id": image_id, "layer_idx": layer_idx,
                           **t, "share": round(t["share"], 8)}
                          for t in token_dominance_shares(tokens, dmap))
        for k, ix in enumerate(mention_idxs):
            maps_out[f"dual{layer_idx}_mention{k}"] = lmap[ix].sum(dim=0).detach().float().cpu()

    reps = {}
    for layer_idx in range(hooked_count):
        io = HOOK_DATA[layer_idx].get("image_output")
        if io is not None:
            reps[f"dual_{layer_idx}"] = io.detach().float().mean(dim=1).cpu()
    for layer_idx in range(single_count):
        oi = SINGLE_HOOK_DATA[layer_idx].get("out_img")
        if oi is not None:
            reps[f"single_{layer_idx}"] = oi.detach().float().mean(dim=1).cpu()

    return {"image": image, "maps": maps_out, "reps": reps, "wall": wall, "mentions": mentions,
            "tables": {RESULTS_NAME: results, WORDS_NAME: word_rows, TOKENS_NAME: token_rows}}


def hhmm(seconds: float) -> str:
    return f"{int(seconds) // 3600:d}h{int(seconds) % 3600 // 60:02d}m"


def disk_report(n_todo: int) -> tuple[int, int]:
    return n_todo * BYTES_PER_IMAGE + BYTES_TABLES, shutil.disk_usage(REPO).free


def print_plan(plan: list[dict], done: set[str]) -> None:
    todo = [r for r in plan if r["image_id"] not in done]
    print(f"work plan: {len(plan)} images, {len(done)} already complete, {len(todo)} to generate")
    for key in ("set", "variety", "condition"):
        counts = Counter(row[key] for row in plan)
        print(f"  by {key:9s} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    calib = sum(1 for r in plan if r["set"] == "calibration")
    print(f"  calibration block runs first: images 1 to {calib}")
    repeated = sum(len(actor_mentions(corpus_spec, row)) > 1 for row in plan)
    print(f"  mentions resolve for all {len(plan)} rows, {repeated} of them with a repeated mention")
    for label, rows in (("first 20", plan[:20]), ("last 20", plan[-20:])):
        print(f"  {label}:")
        for i, row in enumerate(rows):
            n = i + 1 if label == "first 20" else len(plan) - len(rows) + i + 1
            print(f"    {n:5d}  {row['set']:11s} {row['image_id']}")
    need, free = disk_report(len(todo))
    print(f"  disk: need ~{need / 1024 ** 3:.1f} GB, free {free / 1024 ** 3:.1f} GB")
    print(f"  time: ~{hhmm(len(todo) * 18.5)} at 18.5 s/image")


# --------------------------------------------------------------------------- run

def run(args: argparse.Namespace) -> None:
    seeds = {int(s) for s in args.seeds.split(",")} if args.seeds else None
    sets = {s.strip() for s in args.sets.split(",")} if args.sets else None
    plan = build_plan(seeds, sets, args.limit)
    out_dir = Path(args.output_dir)
    done = load_state(out_dir) if out_dir.exists() else set()

    if args.plan:
        print_plan(plan, done)
        return

    todo = [r for r in plan if r["image_id"] not in done]
    need, free = disk_report(len(todo))
    if need > free and not args.allow_low_disk:
        raise SystemExit(f"need ~{need / 1024 ** 3:.1f} GB, only {free / 1024 ** 3:.1f} GB free; "
                         f"free space or pass --allow-low-disk")

    log = make_logger(out_dir)
    log({"event": "run_start", "n_planned": len(plan), "n_skipped_complete": len(done),
         "n_todo": len(todo), "disk_need_gb": round(need / 1024 ** 3, 2),
         "disk_free_gb": round(free / 1024 ** 3, 2), "output_dir": str(out_dir)})
    print(f"resuming: {len(done)} of {len(plan)} images already complete, {len(todo)} to go")
    if not todo:
        return

    pipe, device, hooked_count, single_count = load_pipeline_with_hooks(log, include_single_stream=True)
    writer = RunWriter(out_dir)
    align_cache: dict[str, tuple] = {}
    t_start, n_ok, n_fail = time.time(), 0, 0
    try:
        for i, row in enumerate(todo, start=1):
            try:
                out = process_image(pipe, device, row, hooked_count, single_count, align_cache, log)
                writer.save_tensors(row["image_id"], out["image"], out["reps"], out["maps"])
                writer.commit(row, out["tables"])
                n_ok += 1
                elapsed = time.time() - t_start
                eta = elapsed / n_ok * (len(todo) - i)
                print(f"[{i}/{len(todo)}] {row['set']}/{row['variety']}/{row['profession']}/"
                      f"{row['condition']} {row['image_id']} {out['wall']:.1f}s eta {hhmm(eta)}",
                      flush=True)
                log({"event": "image_done", "image_id": row["image_id"], "index": i,
                     "prompt": row["prompt_text"], "mentions": out["mentions"],
                     "n_result_rows": len(out["tables"][RESULTS_NAME]), "wall_time_s": round(out["wall"], 2),
                     "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)
                     if device == "cuda" else None})
            except Exception as exc:  # one bad image must not end a ten-hour run
                n_fail += 1
                log({"event": "image_failed", "image_id": row["image_id"], "index": i,
                     "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{i}/{len(todo)}] FAILED {row['image_id']}: {exc}", flush=True)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
    finally:
        writer.close()
    log({"event": "run_finished", "n_ok": n_ok, "n_failed": n_fail,
         "wall_time_s": round(time.time() - t_start, 1)})
    print(f"done: {n_ok} generated, {n_fail} failed, in {hhmm(time.time() - t_start)}; {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data/main_run"))
    parser.add_argument("--plan", action="store_true",
                        help="Print the work plan (order, cell counts, disk, time) and exit.")
    parser.add_argument("--seeds", type=str, default="",
                        help="Comma-separated seed filter; default is all five corpus_spec.SEEDS.")
    parser.add_argument("--sets", type=str, default="",
                        help="Comma-separated set filter (calibration, test); default is both.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Take only the first N images of the plan (smoke test).")
    parser.add_argument("--allow-low-disk", action="store_true",
                        help="Start even though the estimated need exceeds the free space.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
