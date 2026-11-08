"""FLUX.2-klein-4B with read-only forward hooks, and the attention probes computed on what they
capture. Generation always runs through the model's own attention kernel.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from token_alignment import FLUX_REVISION, MODEL_ID

VAE_SCALE_FACTOR = 8
TOKEN_GRID_DOWNSAMPLE = VAE_SCALE_FACTOR * 2  # 8x VAE * 2x2 patch packing

NUM_INFERENCE_STEPS = 4  # matches the main project's generation setting
IMAGE_SIZE = 1024
GUIDANCE_SCALE = 1.0
GRID_SIZE = IMAGE_SIZE // TOKEN_GRID_DOWNSAMPLE

# Cleared/populated fresh for each pipe(...) call by the hooks below.
HOOK_DATA: dict[int, dict[str, torch.Tensor]] = {}
SINGLE_HOOK_DATA: dict[int, dict[str, torch.Tensor]] = {}
SHARED_STATE = {"text_seq_len": None, "dual_layer_order": [], "single_layer_order": []}


def register_hooks(pipe) -> int:
    """Registers non-invasive forward hooks on every dual-stream attention module."""
    from diffusers.models.transformers.transformer_flux2 import Flux2AttnProcessor

    HOOK_DATA.clear()
    SHARED_STATE["dual_layer_order"] = []
    hooked = 0

    for name, module in pipe.transformer.named_modules():
        if not isinstance(getattr(module, "processor", None), Flux2AttnProcessor):
            continue
        layer_idx = hooked
        SHARED_STATE["dual_layer_order"].append(name)
        HOOK_DATA[layer_idx] = {}

        def make_attn_hook(idx):
            def hook(mod, args, kwargs, output):
                encoder_hidden_states = kwargs.get("encoder_hidden_states")
                if encoder_hidden_states is not None:
                    SHARED_STATE["text_seq_len"] = encoder_hidden_states.shape[1]
                    HOOK_DATA[idx]["encoder_hidden_states"] = encoder_hidden_states.detach()
                HOOK_DATA[idx]["image_rotary_emb"] = kwargs.get("image_rotary_emb")
                if isinstance(output, tuple):
                    HOOK_DATA[idx]["image_output"] = output[0].detach()
                    HOOK_DATA[idx]["text_output"] = output[1].detach()
                else:
                    HOOK_DATA[idx]["image_output"] = output.detach()
            return hook

        def make_proj_hook(idx, key):
            def hook(mod, args, output):
                HOOK_DATA[idx][key] = output.detach()
            return hook

        module.register_forward_hook(make_attn_hook(layer_idx), with_kwargs=True)
        module.to_k.register_forward_hook(make_proj_hook(layer_idx, "raw_k"))
        module.to_v.register_forward_hook(make_proj_hook(layer_idx, "raw_v"))
        module.to_q.register_forward_hook(make_proj_hook(layer_idx, "raw_q_img"))
        hooked += 1

    return hooked


def register_single_stream_hooks(pipe) -> int:
    """Extends the probe to the 20 single-stream layers."""
    from diffusers.models.transformers.transformer_flux2 import Flux2ParallelSelfAttnProcessor

    SINGLE_HOOK_DATA.clear()
    SHARED_STATE["single_layer_order"] = []
    hooked = 0

    for name, module in pipe.transformer.named_modules():
        if not isinstance(getattr(module, "processor", None), Flux2ParallelSelfAttnProcessor):
            continue
        layer_idx = hooked
        SHARED_STATE["single_layer_order"].append(name)
        SINGLE_HOOK_DATA[layer_idx] = {}

        def make_qkv_hook(idx, attn):
            def hook(mod, args, output):
                text_seq_len = SHARED_STATE["text_seq_len"]
                if text_seq_len is None:
                    return
                qkv = output[..., : 3 * attn.inner_dim]
                _, key, value = qkv.chunk(3, dim=-1)
                SINGLE_HOOK_DATA[idx]["k_img"] = key[:, text_seq_len:].detach().cpu()
                SINGLE_HOOK_DATA[idx]["v_img"] = value[:, text_seq_len:].detach().cpu()
            return hook

        def make_out_hook(idx):
            def hook(mod, args, kwargs, output):
                text_seq_len = SHARED_STATE["text_seq_len"]
                if text_seq_len is None:
                    return
                SINGLE_HOOK_DATA[idx]["image_rotary_emb"] = kwargs.get("image_rotary_emb")
                SINGLE_HOOK_DATA[idx]["out_img"] = output[:, text_seq_len:].detach().cpu()
            return hook

        module.to_qkv_mlp_proj.register_forward_hook(make_qkv_hook(layer_idx, module))
        module.register_forward_hook(make_out_hook(layer_idx), with_kwargs=True)
        hooked += 1

    return hooked


def _ensure_on_device(module: torch.nn.Module, device: torch.device) -> None:
    """enable_model_cpu_offload() relocates each transformer submodule back to CPU immediately after
    its real forward pass; our side computations run after the whole pipe(...) call finishes, so
    norm/proj submodules must be re-staged onto the device of their (still-GPU) captured tensors
    before reuse here.
    """
    if next(module.parameters()).device != device:
        module.to(device)


def run_daam_probe(attn, layer_data: dict, text_seq_len: int, grid_size: int) -> torch.Tensor | None:
    """RQ2 side computation: text-query -> image-key attention for one dual-stream layer, never
    touching the real generation.
    """
    from diffusers.models.embeddings import apply_rotary_emb

    raw_k = layer_data.get("raw_k")
    image_rotary_emb = layer_data.get("image_rotary_emb")
    encoder_hidden_states = layer_data.get("encoder_hidden_states")
    if raw_k is None or encoder_hidden_states is None:
        return None

    _ensure_on_device(attn.norm_k, raw_k.device)
    key = raw_k.unflatten(-1, (attn.heads, -1))
    key = attn.norm_k(key)
    if image_rotary_emb is not None:
        key_rotary_emb = tuple(t[text_seq_len:] for t in image_rotary_emb)
        key = apply_rotary_emb(key, key_rotary_emb, sequence_dim=1)

    image_key = key

    _ensure_on_device(attn.add_q_proj, encoder_hidden_states.device)
    _ensure_on_device(attn.norm_added_q, encoder_hidden_states.device)
    query = attn.add_q_proj(encoder_hidden_states)
    query = query.unflatten(-1, (attn.heads, -1))
    query = attn.norm_added_q(query)
    if image_rotary_emb is not None:
        query_rotary_emb = tuple(t[:text_seq_len] for t in image_rotary_emb)
        query = apply_rotary_emb(query, query_rotary_emb, sequence_dim=1)

    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", query, image_key) * scale
    probs = scores.softmax(dim=-1)  # (batch, heads, text_seq, image_seq)
    probs = probs.mean(dim=(0, 1))  # (text_seq, image_seq)

    image_seq = probs.shape[-1]
    assert image_seq == grid_size * grid_size, (
        f"grid mismatch: expected {grid_size}x{grid_size}={grid_size * grid_size} image tokens, "
        f"got {image_seq}. TOKEN_GRID_DOWNSAMPLE assumption likely wrong for this height/width."
    )
    return probs.reshape(probs.shape[0], grid_size, grid_size)  # (text_seq, grid, grid)


def run_daam_dominance_probe(attn, layer_data: dict, text_seq_len: int, grid_size: int) -> torch.Tensor | None:
    """RQ2 dominance side computation: image-query -> text-key attention, softmax over the TEXT tokens
    (dim=-1) - the OPPOSITE normalization direction from run_daam_probe.
    """
    from diffusers.models.embeddings import apply_rotary_emb

    raw_q = layer_data.get("raw_q_img")
    encoder_hidden_states = layer_data.get("encoder_hidden_states")
    image_rotary_emb = layer_data.get("image_rotary_emb")
    if raw_q is None or encoder_hidden_states is None:
        return None

    _ensure_on_device(attn.norm_q, raw_q.device)
    query = raw_q.unflatten(-1, (attn.heads, -1))
    query = attn.norm_q(query)
    if image_rotary_emb is not None:
        # image-stream query gets the image-token sub-range of the joint rotary table
        query = apply_rotary_emb(query, tuple(t[text_seq_len:] for t in image_rotary_emb), sequence_dim=1)

    _ensure_on_device(attn.add_k_proj, encoder_hidden_states.device)
    _ensure_on_device(attn.norm_added_k, encoder_hidden_states.device)
    key = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
    key = attn.norm_added_k(key)
    if image_rotary_emb is not None:
        key = apply_rotary_emb(key, tuple(t[:text_seq_len] for t in image_rotary_emb), sequence_dim=1)

    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * scale  # q=image, k=text
    probs = scores.softmax(dim=-1)  # over text tokens -> each image position sums to 1
    probs = probs.mean(dim=(0, 1))  # (image_seq, text_seq)

    image_seq = probs.shape[0]
    assert image_seq == grid_size * grid_size, (
        f"grid mismatch: expected {grid_size * grid_size} image tokens, got {image_seq}."
    )
    return probs.transpose(0, 1).reshape(probs.shape[1], grid_size, grid_size)


def word_daam_map(words: list, daam_map: torch.Tensor, target: str) -> torch.Tensor | None:
    """Sum the DAAM token-maps belonging to the (surface) word `target` in a prompt's word grouping
    (from get_word_groups).
    """
    for surface, token_indices in words:
        if surface.strip().lower() == target.strip().lower():
            return daam_map[token_indices].sum(dim=0)
    return None


def word_dominance(words: list, dom_map: torch.Tensor, target: str) -> float | None:
    """Scalar image->text dominance of a word: its share map's mean over image positions (dom_map from
    run_daam_dominance_probe).
    """
    m = word_daam_map(words, dom_map, target)
    return None if m is None else float(m.mean())


def word_token_count(words: list, target: str) -> int | None:
    """Number of encoder tokens the (surface) word `target` occupies in a prompt's word grouping."""
    for surface, token_indices in words:
        if surface.strip().lower() == target.strip().lower():
            return len(token_indices)
    return None


def span_share(dom_map: torch.Tensor, idxs: list[int]) -> float:
    """Scalar image->text dominance of a token span: the span's summed share map, averaged over image
    positions (dom_map from run_daam_dominance_probe, whose shares sum to 1 per position).
    """
    return float(dom_map[idxs].sum(dim=0).mean())


def mention_estimands(
    dom_map: torch.Tensor, mention_idxs: list[list[int]], prompt_token_idxs: list[int],
    n_prompt_tokens: int, dep_idxs: dict[str, list[int]] | None = None,
) -> tuple[dict, dict]:
    """The actor-mention estimands for one image and one dual-stream layer, per the a-priori
    aggregation lock.
    """
    mention_shares = [span_share(dom_map, ix) for ix in mention_idxs]
    share_first = mention_shares[0]
    share_sum = sum(mention_shares)
    share_mean = share_sum / len(mention_shares)
    prompt_mass = span_share(dom_map, prompt_token_idxs)
    n_mention_tokens = sum(len(ix) for ix in mention_idxs)
    exp_frac = n_mention_tokens / n_prompt_tokens
    obs_frac = share_sum / prompt_mass if prompt_mass > 0 else None
    dep_shares = {d: span_share(dom_map, ix) for d, ix in (dep_idxs or {}).items()}
    dep_total = sum(dep_shares.values()) if dep_shares else None
    content_frac_first = (
        share_first / (share_first + dep_total)
        if dep_total is not None and share_first + dep_total > 0 else None
    )
    row = {
        "n_mentions": len(mention_idxs), "n_mention_tokens": n_mention_tokens,
        "n_mention_tokens_first": len(mention_idxs[0]), "n_prompt_tokens": n_prompt_tokens,
        "share_first": round(share_first, 6), "share_sum": round(share_sum, 6),
        "share_mean": round(share_mean, 6), "prompt_mass": round(prompt_mass, 6),
        "exp_frac": round(exp_frac, 6),
        "obs_frac": round(obs_frac, 6) if obs_frac is not None else None,
        "obs_over_exp": round(obs_frac / exp_frac, 4) if obs_frac is not None else None,
        "dep_share_total": round(dep_total, 6) if dep_total is not None else None,
        "content_frac_first": round(content_frac_first, 6) if content_frac_first is not None else None,
    }
    detail = {"mention_shares": [round(s, 6) for s in mention_shares],
              "dep_shares": {d: round(s, 6) for d, s in dep_shares.items()}}
    return row, detail


def token_dominance_shares(tokens: list[tuple], dom_map: torch.Tensor) -> list[dict]:
    """Per-token dominance shares for one layer, the unaggregated form of word_dominance_shares."""
    return [{"seq_index": seq_index, "token_index": token_index, "token_id": token_id,
             "token": surface, "word_index": word_index, "share": span_share(dom_map, [seq_index])}
            for seq_index, token_index, token_id, surface, word_index in tokens]


def word_dominance_shares(words: list, dom_map: torch.Tensor) -> list[dict]:
    """Per-word dominance share for one layer, with the word's token count alongside it."""
    return [
        {"word_index": i, "word": surface.strip(), "n_tokens": len(tidx),
         "share": span_share(dom_map, tidx)}
        for i, (surface, tidx) in enumerate(words)
    ]


def make_logger(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    def log(record: dict) -> None:
        print(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    return log


def load_pipeline_with_hooks(log, include_single_stream: bool = False) -> tuple:
    """Shared setup for both modes: env logging, pipeline load, hook registration."""
    import diffusers
    from diffusers import AutoPipelineForText2Image

    log({
        "event": "environment",
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, revision=FLUX_REVISION, torch_dtype=dtype)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    hooked_count = register_hooks(pipe)
    log({"event": "hooks_registered", "count": hooked_count, "expected": 5})
    single_count = 0
    if include_single_stream:
        single_count = register_single_stream_hooks(pipe)
        log({"event": "single_stream_hooks_registered", "count": single_count, "expected": 20})
    return pipe, device, hooked_count, single_count
