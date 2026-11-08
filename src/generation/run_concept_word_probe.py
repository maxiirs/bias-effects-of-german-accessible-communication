#!/usr/bin/env python3
"""Concept-word probe on prompts that name the gender. Generates two scenes with "Ein Mann" and
"Eine Frau" over three seeds, scores each concept-word pair against every image at all 25 blocks,
saves the mean image representation per block, and records how similar the two concept words are
inside the model. Notebook 06 evaluates the outputs.

Usage (GPU): python src/generation/run_concept_word_probe.py
"""

from __future__ import annotations

import csv
import gc
import math
from pathlib import Path

import torch

from forward_hooks import (
    GUIDANCE_SCALE, HOOK_DATA, IMAGE_SIZE, NUM_INFERENCE_STEPS, SHARED_STATE, SINGLE_HOOK_DATA,
    _ensure_on_device, load_pipeline_with_hooks, make_logger,
)

OUT_DIR = Path("data/concept_word_probe")
CONCEPT_PAIRS = [("männlich", "weiblich"), ("Mann", "Frau")]
CONCEPT_MAX_SEQ_LEN = 8  # the bare word inside the chat template, truncated
PROMPT_PAIRS = [
    ("Ein Mann repariert einen Motor in der Werkstatt.",
     "Eine Frau repariert einen Motor in der Werkstatt."),
    ("Ein Mann arbeitet in einem Büro am Computer.",
     "Eine Frau arbeitet in einem Büro am Computer."),
]
SEEDS = [0, 1000, 2000]


def encode_concept_word(pipe, word: str, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The word's text-encoder output, the same after the context embedder, and a mask of the word's
    own token positions inside the chat template.
    """
    raw, _ = pipe.encode_prompt(prompt=word, device=device, max_sequence_length=CONCEPT_MAX_SEQ_LEN)
    _ensure_on_device(pipe.transformer.context_embedder, raw.device)
    embeds = pipe.transformer.context_embedder(raw)

    bare_ids = pipe.tokenizer(word, add_special_tokens=False)["input_ids"]
    chat_text = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": word}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_ids = pipe.tokenizer(
        chat_text, max_length=CONCEPT_MAX_SEQ_LEN, padding="max_length", truncation=True
    )["input_ids"]
    start = next(
        (i for i in range(len(full_ids) - len(bare_ids) + 1) if full_ids[i:i + len(bare_ids)] == bare_ids), None
    )
    if start is None:
        raise RuntimeError(f"{word!r} not found in its templated encoding {full_ids}")
    content_mask = torch.zeros(len(full_ids), dtype=torch.bool, device=embeds.device)
    content_mask[start:start + len(bare_ids)] = True
    return raw, embeds, content_mask


def dual_stream_query(attn, embeds: torch.Tensor) -> torch.Tensor:
    _ensure_on_device(attn.add_q_proj, embeds.device)
    _ensure_on_device(attn.norm_added_q, embeds.device)
    q = attn.add_q_proj(embeds).unflatten(-1, (attn.heads, -1))
    return attn.norm_added_q(q)


def compute_concept_out(attn, layer_data: dict, concept_embedding: torch.Tensor,
                        content_mask: torch.Tensor, text_seq_len: int) -> torch.Tensor | None:
    """The concept word as an extra text query against one dual-stream block's image keys and values,
    pooled over the word's own tokens.
    """
    from diffusers.models.embeddings import apply_rotary_emb

    raw_k = layer_data.get("raw_k")
    raw_v = layer_data.get("raw_v")
    image_rotary_emb = layer_data.get("image_rotary_emb")
    if raw_k is None or raw_v is None:
        return None

    concept_q = dual_stream_query(attn, concept_embedding)
    _ensure_on_device(attn.norm_k, raw_k.device)
    key = attn.norm_k(raw_k.unflatten(-1, (attn.heads, -1)))
    value = raw_v.unflatten(-1, (attn.heads, -1))
    if image_rotary_emb is not None:
        image_rotary_emb = tuple(t[text_seq_len:] for t in image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    scale = 1.0 / math.sqrt(concept_q.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", concept_q, key) * scale
    probs = scores.softmax(dim=-1)
    concept_out = torch.einsum("bhqk,bkhd->bqhd", probs, value).flatten(-2, -1)
    return concept_out[:, content_mask, :].mean(dim=1)


def precompute_single_stream_concept_queries(pipe, concept_embeddings: dict, device) -> dict:
    """The concept queries of every single-stream block; they do not depend on the image."""
    queries: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx, name in enumerate(SHARED_STATE["single_layer_order"]):
        attn = pipe.transformer.get_submodule(name)
        original_device = next(attn.to_qkv_mlp_proj.parameters()).device
        attn.to_qkv_mlp_proj.to(device)
        attn.norm_q.to(device)
        queries[layer_idx] = {}
        for word, (embedding, _) in concept_embeddings.items():
            projected = attn.to_qkv_mlp_proj(embedding.to(device))
            q = projected[..., : attn.inner_dim].unflatten(-1, (attn.heads, -1))
            queries[layer_idx][word] = attn.norm_q(q).detach()
        attn.to_qkv_mlp_proj.to(original_device)
        attn.norm_q.to(original_device)
    return queries


def compute_single_stream_concept_out(attn, layer_data: dict, concept_q: torch.Tensor,
                                      content_mask: torch.Tensor, text_seq_len: int,
                                      device) -> torch.Tensor | None:
    from diffusers.models.embeddings import apply_rotary_emb

    k_img = layer_data.get("k_img")
    v_img = layer_data.get("v_img")
    image_rotary_emb = layer_data.get("image_rotary_emb")
    if k_img is None or v_img is None:
        return None

    key = k_img.to(device).unflatten(-1, (attn.heads, -1))
    value = v_img.to(device).unflatten(-1, (attn.heads, -1))
    _ensure_on_device(attn.norm_k, key.device)
    key = attn.norm_k(key)
    if image_rotary_emb is not None:
        key_rotary_emb = tuple(t[text_seq_len:].to(device) for t in image_rotary_emb)
        key = apply_rotary_emb(key, key_rotary_emb, sequence_dim=1)

    scale = 1.0 / math.sqrt(concept_q.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", concept_q, key) * scale
    probs = scores.softmax(dim=-1)
    concept_out = torch.einsum("bhqk,bkhd->bqhd", probs, value).flatten(-2, -1)
    return concept_out[:, content_mask, :].mean(dim=1)


def global_concept_score(concept_vec: torch.Tensor, image_output: torch.Tensor) -> float:
    """Mean image output over all positions, uncentred, dotted with the concept vector."""
    concept_vec = concept_vec.detach().float()
    return float(torch.einsum("bd,bd->b", concept_vec, image_output.detach().float().mean(dim=1)).mean())


def concept_cosines(pipe, encoded: dict, tokens: dict, hooked_count: int) -> list[dict]:
    """Cosine of the two words of each pair, token by token: text-encoder output, after the context
    embedder, and as the text query of each dual-stream block.
    """
    rows = []
    for word_a, word_b in CONCEPT_PAIRS:
        (raw_a, emb_a, mask_a), (raw_b, emb_b, mask_b) = encoded[word_a], encoded[word_b]
        stages = {"encoder": (raw_a, raw_b), "context_embedder": (emb_a, emb_b)}
        for layer_idx in range(hooked_count):
            attn = pipe.transformer.get_submodule(SHARED_STATE["dual_layer_order"][layer_idx])
            stages[f"dual_{layer_idx}"] = (dual_stream_query(attn, emb_a).flatten(-2, -1),
                                           dual_stream_query(attn, emb_b).flatten(-2, -1))
        for stage, (a, b) in stages.items():
            a, b = a[0, mask_a].detach().float(), b[0, mask_b].detach().float()
            for i in range(min(len(a), len(b))):
                rows.append({"word_a": word_a, "word_b": word_b, "stage": stage, "token_index": i,
                             "token_a": tokens[word_a][i], "token_b": tokens[word_b][i],
                             "cosine": float(torch.nn.functional.cosine_similarity(a[i], b[i], dim=0))})
    return rows


def layer_outputs(pipe, concept_embeddings: dict, single_queries: dict, hooked_count: int,
                  single_count: int, device):
    """Each block's concept outputs and image output for the image just generated, one block at a time."""
    text_seq_len = SHARED_STATE["text_seq_len"]
    for layer_idx in range(hooked_count):
        attn = pipe.transformer.get_submodule(SHARED_STATE["dual_layer_order"][layer_idx])
        outs = {w: compute_concept_out(attn, HOOK_DATA[layer_idx], emb, mask, text_seq_len)
                for w, (emb, mask) in concept_embeddings.items()}
        yield "dual", layer_idx, outs, HOOK_DATA[layer_idx].get("image_output")
    for layer_idx in range(single_count):
        attn = pipe.transformer.get_submodule(SHARED_STATE["single_layer_order"][layer_idx])
        layer_data = SINGLE_HOOK_DATA[layer_idx]
        outs = {w: compute_single_stream_concept_out(attn, layer_data, single_queries[layer_idx][w],
                                                     mask, text_seq_len, device)
                for w, (_, mask) in concept_embeddings.items()}
        out_img = layer_data.get("out_img")
        yield "single", layer_idx, outs, out_img.to(device) if out_img is not None else None


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


@torch.no_grad()
def main() -> None:
    log = make_logger(OUT_DIR)
    pipe, device, hooked_count, single_count = load_pipeline_with_hooks(log, include_single_stream=True)

    words = [w for pair in CONCEPT_PAIRS for w in pair]
    encoded = {w: encode_concept_word(pipe, w, device) for w in words}
    tokens = {w: pipe.tokenizer.convert_ids_to_tokens(pipe.tokenizer(w, add_special_tokens=False)["input_ids"])
              for w in words}
    concept_embeddings = {w: (emb, mask) for w, (_, emb, mask) in encoded.items()}
    single_queries = precompute_single_stream_concept_queries(pipe, concept_embeddings, device)
    write_csv(OUT_DIR / "concept_word_cosine.csv", concept_cosines(pipe, encoded, tokens, hooked_count))

    rows = []
    for pair_idx, (male_prompt, female_prompt) in enumerate(PROMPT_PAIRS):
        for role, prompt in (("male", male_prompt), ("female", female_prompt)):
            for seed in SEEDS:
                image = pipe(
                    prompt=prompt, guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
                    height=IMAGE_SIZE, width=IMAGE_SIZE, max_sequence_length=256,
                    generator=torch.Generator(device=device).manual_seed(seed),
                ).images[0]
                image.save(OUT_DIR / f"pair{pair_idx}_{role}_seed{seed}.png")
                log({"event": "image_generated", "pair_idx": pair_idx, "role": role, "seed": seed,
                     "prompt": prompt})

                for stream, layer_idx, outs, image_output in layer_outputs(
                        pipe, concept_embeddings, single_queries, hooked_count, single_count, device):
                    for word_a, word_b in CONCEPT_PAIRS:
                        if outs[word_a] is None or outs[word_b] is None or image_output is None:
                            continue
                        rows.append({"word_a": word_a, "word_b": word_b, "stream": stream,
                                     "layer_idx": layer_idx, "pair_idx": pair_idx, "seed": seed, "role": role,
                                     "global_diff_score": global_concept_score(outs[word_a] - outs[word_b],
                                                                               image_output)})

                reps = {}
                for layer_idx in range(hooked_count):
                    io = HOOK_DATA[layer_idx].get("image_output")
                    if io is not None:
                        reps[f"dual_{layer_idx}"] = io.detach().float().mean(dim=1).cpu()
                for layer_idx in range(single_count):
                    oi = SINGLE_HOOK_DATA[layer_idx].get("out_img")
                    if oi is not None:
                        reps[f"single_{layer_idx}"] = oi.detach().float().mean(dim=1).cpu()
                torch.save(reps, OUT_DIR / f"reps_pair{pair_idx}_{role}_seed{seed}.pt")

                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

    write_csv(OUT_DIR / "concept_probe_scores.csv", rows)
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
