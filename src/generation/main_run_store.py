#!/usr/bin/env python3
"""Manifest, result tables and per-image tensors of the main run, and the rule that resumes an
interrupted run.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))

import corpus_spec  # noqa: E402

MANIFEST_NAME = "image_manifest.csv"      # classify_depicted_gender.py --manifest reads this directly
RESULTS_NAME = "dominance_results.csv"
WORDS_NAME = "word_shares_long.csv"
TOKENS_NAME = "token_shares_long.csv"
MANIFEST_FIELDS = corpus_spec.FIELDS
RESULT_FIELDS = [
    "image_id", "profession", "scene_label", "condition", "variety", "seed", "layer_idx",
    "n_mentions", "n_mention_tokens", "n_mention_tokens_first", "n_prompt_tokens",
    "share_first", "share_sum", "share_mean", "prompt_mass", "exp_frac", "obs_frac",
    "obs_over_exp", "dep_share_total", "content_frac_first",
]
WORD_FIELDS = ["image_id", "layer_idx", "word_index", "word", "n_tokens", "share"]
TOKEN_FIELDS = ["image_id", "layer_idx", "seq_index", "token_index", "token_id", "token",
                "word_index", "share"]
TABLES = ((RESULTS_NAME, RESULT_FIELDS), (WORDS_NAME, WORD_FIELDS), (TOKENS_NAME, TOKEN_FIELDS))
ARTIFACT_PATTERNS = ("{image_id}.png", "reps_{image_id}.pt", "maps_{image_id}.pt")


def artifact_paths(out_dir: Path, image_id: str) -> list[Path]:
    return [out_dir / pattern.format(image_id=image_id) for pattern in ARTIFACT_PATTERNS]


def artifacts_complete(out_dir: Path, image_id: str) -> bool:
    return all(path.exists() for path in artifact_paths(out_dir, image_id))


def rewrite_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_state(out_dir: Path) -> set[str]:
    """Image ids whose artefacts are all present, with the tables truncated back to them."""
    manifest = out_dir / MANIFEST_NAME
    if not manifest.exists():
        return set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    done = {r["image_id"] for r in rows if artifacts_complete(out_dir, r["image_id"])}
    rewrite_csv(manifest, MANIFEST_FIELDS, [r for r in rows if r["image_id"] in done])
    for name, fields in TABLES:
        path = out_dir / name
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                keep = [r for r in csv.DictReader(handle) if r.get("image_id") in done]
            rewrite_csv(path, fields, keep)
    return done


class RunWriter:
    """Appends to the three tables and saves the tensors in the order that makes resumption safe:
    tensors first (each via a temporary file and an atomic rename), then the two long tables, and
    the manifest row only once everything else is durable.
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self._files = {}
        for name, fields in TABLES + ((MANIFEST_NAME, MANIFEST_FIELDS),):
            path = out_dir / name
            fresh = not path.exists() or path.stat().st_size == 0
            handle = path.open("a", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", restval="")
            if fresh:
                writer.writeheader()
            self._files[name] = (handle, writer)

    def _append(self, name: str, rows: list[dict]) -> None:
        handle, writer = self._files[name]
        writer.writerows(rows)
        handle.flush()

    def save_tensors(self, image_id: str, image, reps: dict, maps: dict) -> None:
        savers = (lambda p: image.save(p, format="PNG"),
                  lambda p: torch.save(reps, p),
                  lambda p: torch.save(maps, p))
        for path, save in zip(artifact_paths(self.out_dir, image_id), savers):
            tmp = path.with_suffix(path.suffix + ".tmp")
            save(tmp)
            os.replace(tmp, path)

    def commit(self, row: dict, tables: dict[str, list[dict]]) -> None:
        for name, _fields in TABLES:
            self._append(name, tables.get(name, []))
        self._append(MANIFEST_NAME, [row])

    def close(self) -> None:
        for handle, _ in self._files.values():
            handle.close()
