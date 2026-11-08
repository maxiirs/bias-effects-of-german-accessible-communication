#!/usr/bin/env python3
"""Size and length control, which select the dual-stream blocks.

Usage (GPU):
    python src/generation/run_attention_controls.py --mode size-control
    python src/generation/run_attention_controls.py --mode length-control
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch

from control_stimuli import (
    DOMINANCE_CONTROL_PROMPTS, DOMINANCE_CONTROL_SEEDS, LENGTH_CONTROL_ARTICLES, LENGTH_CONTROL_FRAME,
    LENGTH_CONTROL_ORDERS, LENGTH_CONTROL_PAIRS, LENGTH_CONTROL_SEEDS,
)
from forward_hooks import (
    GRID_SIZE, GUIDANCE_SCALE, HOOK_DATA, IMAGE_SIZE, NUM_INFERENCE_STEPS, SHARED_STATE,
    load_pipeline_with_hooks, make_logger, run_daam_dominance_probe, word_dominance, word_token_count,
)
from token_alignment import get_word_groups

OUTPUT_DIRS = {
    "size-control": Path("data/attention_controls/size"),
    "length-control": Path("data/attention_controls/length"),
}


def run_size_control(args: argparse.Namespace) -> None:
    """Size-dominance validity control: the large object's word should get a higher image->text share
    than the tiny object's, per dual-stream layer.
    """
    out_dir = Path(args.output_dir)
    log = make_logger(out_dir)
    pipe, device, hooked_count, _ = load_pipeline_with_hooks(log)

    def generate(prompt, seed):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        img = pipe(
            prompt=prompt, guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
            height=IMAGE_SIZE, width=IMAGE_SIZE, max_sequence_length=256,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]
        return img

    def dom_maps(prompt):
        """Per-layer dominance maps + the prompt's word grouping."""
        words = get_word_groups(pipe.tokenizer, prompt)
        text_seq_len = SHARED_STATE["text_seq_len"]
        maps = {}
        for layer_idx in range(hooked_count):
            attn = pipe.transformer.get_submodule(SHARED_STATE["dual_layer_order"][layer_idx])
            maps[layer_idx] = run_daam_dominance_probe(attn, HOOK_DATA[layer_idx], text_seq_len, GRID_SIZE)
        return words, maps

    # --- (a) size-dominance validity control ---
    control_layer_wins = {i: [] for i in range(hooked_count)}
    control_rows = []
    for spec in DOMINANCE_CONTROL_PROMPTS:
        for seed in DOMINANCE_CONTROL_SEEDS:
            image = generate(spec["prompt"], seed)
            image.save(out_dir / f"domctrl_{spec['big']}_vs_{spec['small']}_seed{seed}.png")
            words, maps = dom_maps(spec["prompt"])
            for layer_idx, dmap in maps.items():
                if dmap is None:
                    continue
                big = word_dominance(words, dmap, spec["big"])
                small = word_dominance(words, dmap, spec["small"])
                if big is None or small is None:
                    log({"event": "dom_control_word_missing", "prompt": spec["prompt"],
                         "layer_idx": layer_idx, "words": [w for w, _ in words]})
                    continue
                control_layer_wins[layer_idx].append(big > small)
                n_big = word_token_count(words, spec["big"])
                n_small = word_token_count(words, spec["small"])
                big_pt = big / n_big if n_big else None
                small_pt = small / n_small if n_small else None
                control_rows.append({
                    "pair": f"{spec['big']}/{spec['small']}", "big_name": spec["big"],
                    "small_name": spec["small"], "seed": seed, "layer_idx": layer_idx,
                    "n_tokens_big": n_big, "n_tokens_small": n_small,
                    "big_sum": big, "small_sum": small,
                    "big_per_token": big_pt, "small_per_token": small_pt,
                    "ratio_sum": big / small if small else None,
                    "ratio_per_token": (big_pt / small_pt) if small_pt else None,
                    "prompt": spec["prompt"],
                })
                log({"event": "dom_control_score", "big": spec["big"], "small": spec["small"],
                     "seed": seed, "layer_idx": layer_idx, "big_share": round(big, 5),
                     "small_share": round(small, 5), "ratio": round(big / small, 3) if small else None,
                     "n_tokens_big": n_big, "n_tokens_small": n_small,
                     "big_share_per_token": round(big_pt, 6) if big_pt else None,
                     "small_share_per_token": round(small_pt, 6) if small_pt else None})
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    if control_rows:
        csv_path = out_dir / "size_control_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(control_rows[0].keys()))
            writer.writeheader()
            writer.writerows(control_rows)
        print(f"wrote {len(control_rows)} rows -> {csv_path}")
    for layer_idx, wins in control_layer_wins.items():
        if wins:
            log({"event": "dom_control_summary", "layer_idx": layer_idx,
                 "big_gt_small": sum(wins), "n": len(wins)})

    print(f"Done. Outputs in {out_dir}")


def run_length_control(args: argparse.Namespace) -> None:
    """Second positive control: how much image->text dominance does name LENGTH buy on its own?"""
    out_dir = Path(args.output_dir)
    log = make_logger(out_dir)
    pipe, device, hooked_count, _ = load_pipeline_with_hooks(log)

    def generate(prompt, seed):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        return pipe(
            prompt=prompt, guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
            height=IMAGE_SIZE, width=IMAGE_SIZE, max_sequence_length=256,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]

    def dom_maps(prompt):
        words = get_word_groups(pipe.tokenizer, prompt)
        text_seq_len = SHARED_STATE["text_seq_len"]
        maps = {}
        for layer_idx in range(hooked_count):
            attn = pipe.transformer.get_submodule(SHARED_STATE["dual_layer_order"][layer_idx])
            maps[layer_idx] = run_daam_dominance_probe(attn, HOOK_DATA[layer_idx], text_seq_len, GRID_SIZE)
        return words, maps

    rows = []
    for pair in LENGTH_CONTROL_PAIRS:
        short_name, long_name = pair["short"], pair["long"]
        article = LENGTH_CONTROL_ARTICLES[short_name]
        for order in LENGTH_CONTROL_ORDERS:
            first, second = (short_name, long_name) if order == "short_first" else (long_name, short_name)
            prompt = LENGTH_CONTROL_FRAME.format(
                art_a=article, a=first, art_b=article.lower(), b=second, ground=pair["ground"]
            )
            for seed in LENGTH_CONTROL_SEEDS:
                image = generate(prompt, seed)
                image.save(out_dir / f"lenctrl_{short_name}_vs_{long_name}_{order}_seed{seed}.png")
                words, maps = dom_maps(prompt)
                for layer_idx, dmap in maps.items():
                    if dmap is None:
                        continue
                    s_sum = word_dominance(words, dmap, short_name)
                    l_sum = word_dominance(words, dmap, long_name)
                    s_n = word_token_count(words, short_name)
                    l_n = word_token_count(words, long_name)
                    if s_sum is None or l_sum is None or not s_n or not l_n:
                        log({"event": "length_control_word_missing", "prompt": prompt,
                             "layer_idx": layer_idx, "words": [w for w, _ in words]})
                        continue
                    s_pt, l_pt = s_sum / s_n, l_sum / l_n
                    row = {
                        "pair": f"{short_name}/{long_name}", "order": order, "seed": seed,
                        "layer_idx": layer_idx,
                        "short_name": short_name, "long_name": long_name,
                        "n_tokens_short": s_n, "n_tokens_long": l_n,
                        "short_sum": s_sum, "long_sum": l_sum,
                        "short_per_token": s_pt, "long_per_token": l_pt,
                        "diff_sum": l_sum - s_sum,
                        "diff_per_token": l_pt - s_pt,
                        "ratio_sum": l_sum / s_sum if s_sum else None,
                        "ratio_per_token": l_pt / s_pt if s_pt else None,
                        "prompt": prompt,
                    }
                    rows.append(row)
                    log({"event": "length_control_score", **{
                        k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}})
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

    csv_path = out_dir / "length_control_results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows -> {csv_path}")

    for layer_idx in range(hooked_count):
        at_layer = [r for r in rows if r["layer_idx"] == layer_idx]
        if not at_layer:
            continue

        def _mean(subset, key):
            return sum(r[key] for r in subset) / len(subset) if subset else None

        entry = {"event": "length_control_summary", "layer_idx": layer_idx, "n": len(at_layer),
                 "mean_diff_sum": round(_mean(at_layer, "diff_sum"), 6),
                 "mean_diff_per_token": round(_mean(at_layer, "diff_per_token"), 6)}
        for order in LENGTH_CONTROL_ORDERS:
            sub = [r for r in at_layer if r["order"] == order]
            entry[f"mean_diff_sum_{order}"] = round(_mean(sub, "diff_sum"), 6) if sub else None
            entry[f"mean_diff_per_token_{order}"] = round(_mean(sub, "diff_per_token"), 6) if sub else None
        log(entry)

    print(f"Done. Outputs in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=sorted(OUTPUT_DIRS), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir = args.output_dir or OUTPUT_DIRS[args.mode]
    if args.mode == "size-control":
        run_size_control(args)
    else:
        run_length_control(args)


if __name__ == "__main__":
    main()
