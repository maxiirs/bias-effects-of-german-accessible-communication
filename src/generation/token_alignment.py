"""Aligns prompt words and actor mentions with the token positions of the chat-templated sequence.
"""

from __future__ import annotations

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
FLUX_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"  # pinned for the pipeline and the tokenizer downloads


def templated_prompt_alignment(
    tokenizer, prompt: str, max_sequence_length: int = 256,
) -> tuple[list[tuple[int, int]], int, int]:
    """Char-offset table for the bare prompt plus its token offset inside the encoder's REAL text
    sequence, so char spans in the prompt can be mapped to positions in the attention maps.
    """
    bare = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    bare_ids = bare["input_ids"]
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    full_ids = tokenizer(
        chat_text, max_length=max_sequence_length, padding="max_length", truncation=True
    )["input_ids"]
    offset = next(
        (i for i in range(len(full_ids) - len(bare_ids) + 1) if full_ids[i:i + len(bare_ids)] == bare_ids), None
    )
    if offset is None:
        raise RuntimeError(
            f"Bare tokenization of {prompt!r} not found as a subsequence of its chat-templated "
            f"encoding - token/map alignment would be undefined. Check template/tokenizer versions."
        )
    return bare["offset_mapping"], offset, len(bare_ids)


def find_span_token_indices(
    offsets: list[tuple[int, int]], template_offset: int, prompt: str, needle: str, search_from: int = 0,
) -> tuple[list[int], int]:
    """Templated-sequence token indices of the first occurrence of `needle` at/after char position
    `search_from` in the prompt (a token counts if its char span overlaps the needle's - the leading
    space Qwen attaches to a word-initial token is covered by the overlap test).
    """
    char_start = prompt.find(needle, search_from)
    if char_start < 0:
        raise ValueError(f"{needle!r} not found in {prompt!r} at/after char {search_from}")
    char_end = char_start + len(needle)
    idxs = [template_offset + i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    assert idxs, (needle, prompt)
    return idxs, char_end


def get_word_groups(tokenizer, prompt: str, max_sequence_length: int = 256) -> list[tuple[str, list[int]]]:
    """Word -> token-index groups, with indices in the encoder's REAL (chat-templated) sequence - the
    coordinate system of every attention map computed here.
    """
    encoding = tokenizer(prompt, return_offsets_mapping=True)
    _, template_offset, _ = templated_prompt_alignment(tokenizer, prompt, max_sequence_length)
    word_ids = encoding.word_ids()
    groups: dict[int, list[int]] = {}
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        groups.setdefault(word_id, []).append(token_idx)
    words = []
    for word_id in sorted(groups):
        start, end = encoding.word_to_chars(word_id)
        words.append((prompt[start:end], [template_offset + t for t in groups[word_id]]))
    return words


def actor_mentions(corpus_spec, row: dict) -> list[str]:
    """Actor mention strings for one corpus row, taken from spec (the same code that built the prompts)
    rather than re-parsed out of the prompt text.
    """
    prof, scene, cond = row["profession"], row["scene_label"], row["condition"]
    actor, _num = corpus_spec.ACTORS[prof][cond]
    mentions = [actor]
    if row["variety"] == "ls" and "{ACTOR_DEF}" in corpus_spec.LS_FRAMES[prof][scene]:
        mentions.append(corpus_spec.definite_actor(actor))
    assert row["prompt_text"].startswith(mentions[0]), (row["image_id"], mentions[0], row["prompt_text"])
    return mentions


def mention_token_spans(
    offsets: list[tuple[int, int]], template_offset: int, prompt: str, mentions: list[str],
) -> list[list[int]]:
    """Templated-sequence token indices for each mention, searched left to right so a repeated mention
    resolves to its own occurrence and not again to the first one.
    """
    spans: list[list[int]] = []
    search_from = 0
    for mention in mentions:
        idxs, search_from = find_span_token_indices(offsets, template_offset, prompt, mention, search_from)
        spans.append(idxs)
    return spans


def get_token_groups(tokenizer, prompt: str, max_sequence_length: int = 256) -> list[tuple]:
    """(templated-sequence index, prompt-relative index, token id, token surface, word index) for every
    token of the prompt, i.e. the finest grain at which a dominance share exists.
    """
    offsets, template_offset, n_prompt_tokens = templated_prompt_alignment(
        tokenizer, prompt, max_sequence_length)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    surfaces = tokenizer.convert_ids_to_tokens(ids)
    word_of = {}
    for word_index, (_surface, token_indices) in enumerate(
            get_word_groups(tokenizer, prompt, max_sequence_length)):
        for t in token_indices:
            word_of[t] = word_index
    return [(template_offset + i, i, ids[i], surfaces[i], word_of.get(template_offset + i))
            for i in range(n_prompt_tokens)]
