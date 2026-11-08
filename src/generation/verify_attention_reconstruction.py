#!/usr/bin/env python3
"""Checks that the reconstructed attention reproduces the output of every dual-stream block.

Usage (GPU): python src/generation/verify_attention_reconstruction.py
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forward_hooks import (  # noqa: E402
    GUIDANCE_SCALE, HOOK_DATA, IMAGE_SIZE, NUM_INFERENCE_STEPS, SHARED_STATE,
    _ensure_on_device, load_pipeline_with_hooks, make_logger,
)

PROMPT_FILE = "data/corpus/sd_prompts.csv"
HEAD_CHUNK = 2

VARIANTS = (
    "intact",
    "rotary_streams_swapped",
    "no_norm_added_k",
    "no_norm_q",
    "concat_image_first",
)


def _rotary_slice(rotary, start: int, length: int, device):
    return tuple(t[start:start + length].to(device) for t in rotary)


def reconstruct(attn, layer_data: dict, text_seq_len: int, variant: str,
                head_chunk: int = HEAD_CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuilds one dual-stream block's joint attention output, returning (image, text) in the same
    shapes and dtype the module itself returns.
    """
    from diffusers.models.embeddings import apply_rotary_emb

    raw_q_img = layer_data["raw_q_img"]
    raw_k_img = layer_data["raw_k"]
    raw_v_img = layer_data["raw_v"]
    encoder_hidden_states = layer_data["encoder_hidden_states"]
    rotary = layer_data["image_rotary_emb"]
    device = raw_q_img.device
    heads = attn.heads

    for module in (attn.norm_q, attn.norm_k, attn.norm_added_q, attn.norm_added_k,
                   attn.add_q_proj, attn.add_k_proj, attn.add_v_proj,
                   attn.to_add_out, attn.to_out[0]):
        _ensure_on_device(module, device)

    # Step 1-3, image stream: the projections are hooked, the QK norms are applied here.
    q_img = raw_q_img.unflatten(-1, (heads, -1))
    k_img = raw_k_img.unflatten(-1, (heads, -1))
    v_img = raw_v_img.unflatten(-1, (heads, -1))
    if variant != "no_norm_q":          # the dropped link in that variant
        q_img = attn.norm_q(q_img)
    k_img = attn.norm_k(k_img)

    # Step 1-3, text stream: recomputed from the layer's own encoder_hidden_states input.
    q_txt = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (heads, -1))
    k_txt = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (heads, -1))
    v_txt = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (heads, -1))
    q_txt = attn.norm_added_q(q_txt)
    if variant != "no_norm_added_k":
        k_txt = attn.norm_added_k(k_txt)

    image_seq_len = q_img.shape[1]
    assert q_txt.shape[1] == text_seq_len, (q_txt.shape, text_seq_len)

    if variant == "concat_image_first":
        order = (("img", q_img, k_img, v_img), ("txt", q_txt, k_txt, v_txt))
    else:
        order = (("txt", q_txt, k_txt, v_txt), ("img", q_img, k_img, v_img))

    query = torch.cat([q for _, q, _, _ in order], dim=1)
    key = torch.cat([k for _, _, k, _ in order], dim=1)
    value = torch.cat([v for _, _, _, v in order], dim=1)

    if rotary is not None:
        if variant == "rotary_streams_swapped":
            txt_rot = _rotary_slice(rotary, text_seq_len, text_seq_len, device)
            img_rot = _rotary_slice(rotary, 0, image_seq_len, device)
            q_txt_r = apply_rotary_emb(q_txt, txt_rot, sequence_dim=1)
            k_txt_r = apply_rotary_emb(k_txt, txt_rot, sequence_dim=1)
            q_img_r = apply_rotary_emb(q_img, img_rot, sequence_dim=1)
            k_img_r = apply_rotary_emb(k_img, img_rot, sequence_dim=1)
            query = torch.cat([q_txt_r, q_img_r], dim=1)
            key = torch.cat([k_txt_r, k_img_r], dim=1)
        else:
            full_rot = tuple(t.to(device) for t in rotary)
            query = apply_rotary_emb(query, full_rot, sequence_dim=1)
            key = apply_rotary_emb(key, full_rot, sequence_dim=1)

    head_dim = query.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    joint = torch.empty_like(query)
    for h0 in range(0, heads, head_chunk):
        h1 = min(h0 + head_chunk, heads)
        q_c = query[:, :, h0:h1].transpose(1, 2).float()
        k_c = key[:, :, h0:h1].transpose(1, 2).float()
        v_c = value[:, :, h0:h1].transpose(1, 2).float()
        weights = torch.matmul(q_c, k_c.transpose(-1, -2)) * scale
        weights = weights.softmax(dim=-1)
        out_c = torch.matmul(weights, v_c)
        joint[:, :, h0:h1] = out_c.transpose(1, 2).to(joint.dtype)
        del q_c, k_c, v_c, weights, out_c

    # Step 7-8.
    joint = joint.flatten(2, 3).to(query.dtype)
    if variant == "concat_image_first":
        image_part, text_part = joint[:, :image_seq_len], joint[:, image_seq_len:]
    else:
        text_part, image_part = joint[:, :text_seq_len], joint[:, text_seq_len:]
    text_out = attn.to_add_out(text_part)
    image_out = attn.to_out[0](image_part)
    image_out = attn.to_out[1](image_out)
    return image_out, text_out


def compare(recon: torch.Tensor, reference: torch.Tensor) -> dict:
    """Relative agreement measures."""
    a = recon.detach().double().flatten()
    b = reference.detach().double().flatten()
    diff = (a - b).abs()
    ref_max = float(b.abs().max())
    ref_mean = float(b.abs().mean())
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    cosine = float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))
    return {
        "ref_max_abs": ref_max,
        "ref_mean_abs": ref_mean,
        "max_abs_dev": max_abs,
        "rel_max_dev": max_abs / ref_max if ref_max else float("nan"),
        "mean_abs_dev": mean_abs,
        "rel_mean_dev": mean_abs / ref_mean if ref_mean else float("nan"),
        "rel_l2": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)),
        "cosine": cosine,
        "one_minus_cosine": 1.0 - cosine,
    }


def summarize(rows: list[dict]) -> list[dict]:
    """Worst case per variant, block and stream over the generated images: the largest deviation and
    the smallest cosine, so a single bad image cannot be averaged away.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["variant"], int(row["block"]), row["stream"]), []).append(row)
    summary = []
    for (variant, block, stream), subset in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        summary.append({
            "variant": variant, "block": block, "stream": stream, "n_images": len(subset),
            "ref_max_abs": max(float(r["ref_max_abs"]) for r in subset),
            "ref_mean_abs": max(float(r["ref_mean_abs"]) for r in subset),
            "max_abs_dev": max(float(r["max_abs_dev"]) for r in subset),
            "rel_max_dev": max(float(r["rel_max_dev"]) for r in subset),
            "rel_mean_dev": max(float(r["rel_mean_dev"]) for r in subset),
            "rel_l2": max(float(r["rel_l2"]) for r in subset),
            "min_cosine": min(float(r["cosine"]) for r in subset),
            "max_one_minus_cosine": max(float(r["one_minus_cosine"]) for r in subset),
        })
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], variants: list[str]) -> None:
    print("\n=== worst case over blocks, streams and images, per variant ===")
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        if not subset:
            continue
        print(f"{variant:24s} rel_max_dev<={max(float(r['rel_max_dev']) for r in subset):.4g}  "
              f"rel_mean_dev<={max(float(r['rel_mean_dev']) for r in subset):.4g}  "
              f"rel_l2<={max(float(r['rel_l2']) for r in subset):.4g}  "
              f"1-cosine<={max(float(r['one_minus_cosine']) for r in subset):.3g}")


def load_prompts(limit: int) -> list[dict]:
    with (REPO / PROMPT_FILE).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/attention_verification"))
    parser.add_argument("--csv-out", type=Path, default=Path("exports/attention_reconstruction_verification.csv"))
    parser.add_argument("--n-images", type=int, default=2)
    parser.add_argument("--head-chunk", type=int, default=HEAD_CHUNK)
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS))
    parser.add_argument("--from-csv", action="store_true",
                        help="Rewrite the summary table from an existing detail CSV, no GPU run.")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    csv_path = args.csv_out if args.csv_out.is_absolute() else REPO / args.csv_out
    if args.from_csv:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        summary_path = csv_path.with_name(csv_path.stem + "_summary.csv")
        write_csv(summary_path, summarize(rows))
        print_summary(rows, variants)
        print(f"summary written: {summary_path}")
        return

    out_dir = Path(args.output_dir)
    log = make_logger(out_dir)
    pipe, device, hooked_count, _ = load_pipeline_with_hooks(log)
    log({"event": "dual_blocks", "count": hooked_count, "layer_names": SHARED_STATE["dual_layer_order"]})

    rows: list[dict] = []
    for idx, prompt_row in enumerate(load_prompts(args.n_images)):
        prompt = prompt_row["prompt_text"]
        seed = int(prompt_row["seed"])
        image = pipe(
            prompt=prompt, guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
            height=IMAGE_SIZE, width=IMAGE_SIZE, max_sequence_length=256,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]
        image.save(out_dir / f"image_{idx}.png")
        text_seq_len = SHARED_STATE["text_seq_len"]
        log({"event": "generated", "prompt_idx": idx, "image_id": prompt_row["image_id"],
             "prompt": prompt, "seed": seed, "text_seq_len": text_seq_len})

        for block in range(hooked_count):
            layer_data = HOOK_DATA[block]
            attn = pipe.transformer.get_submodule(SHARED_STATE["dual_layer_order"][block])
            ref_image = layer_data["image_output"]
            ref_text = layer_data["text_output"]
            for variant in variants:
                recon_image, recon_text = reconstruct(
                    attn, layer_data, text_seq_len, variant, head_chunk=args.head_chunk)
                for stream, recon, reference in (("image", recon_image, ref_image),
                                                 ("text", recon_text, ref_text)):
                    record = {
                        "prompt_idx": idx, "image_id": prompt_row["image_id"], "seed": seed,
                        "block": block, "variant": variant, "stream": stream,
                        "n_tokens": reference.shape[1], "text_seq_len": text_seq_len,
                        "dtype": str(reference.dtype).replace("torch.", ""),
                        **compare(recon, reference),
                    }
                    rows.append(record)
                    log({"event": "comparison", **record})
                del recon_image, recon_text
            if device == "cuda":
                torch.cuda.empty_cache()

    csv_path = args.csv_out if args.csv_out.is_absolute() else REPO / args.csv_out
    write_csv(csv_path, rows)
    summary_path = csv_path.with_name(csv_path.stem + "_summary.csv")
    write_csv(summary_path, summarize(rows))
    log({"event": "csv_written", "path": str(csv_path), "rows": len(rows),
         "summary_path": str(summary_path)})

    print_summary(rows, variants)


if __name__ == "__main__":
    main()
