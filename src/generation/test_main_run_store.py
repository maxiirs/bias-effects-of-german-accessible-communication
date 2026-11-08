#!/usr/bin/env python3
"""Resume checks for main_run_store.py, no GPU needed.

Usage: python src/generation/test_main_run_store.py
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from main_run_store import (  # noqa: E402
    MANIFEST_NAME, RESULTS_NAME, TOKENS_NAME, WORDS_NAME, RunWriter, load_state,
)

OUT = Path(tempfile.mkdtemp(prefix="main_run_store_test_"))


class FakeImage:
    def save(self, path, format=None):
        Path(path).write_bytes(b"png")


def manifest_row(i: int) -> dict:
    return {"image_id": f"img{i}", "profession": "mechaniker", "scene_label": "s", "set": "test",
            "condition": "generic_masc", "variety": "sd", "seed": 0, "prompt_text": "Ein X."}


def tables(i: int) -> dict:
    return {
        RESULTS_NAME: [{"image_id": f"img{i}", "layer_idx": L, "share_first": 0.1 * L}
                       for L in range(5)],
        WORDS_NAME: [{"image_id": f"img{i}", "layer_idx": L, "word_index": 0, "word": "Ein",
                      "n_tokens": 1, "share": 0.02} for L in range(5)],
        TOKENS_NAME: [{"image_id": f"img{i}", "layer_idx": L, "seq_index": 3, "token_index": 0,
                       "token_id": 9, "token": "ĠEin", "word_index": 0, "share": 0.02}
                      for L in range(5)],
    }


def write(writer: RunWriter, i: int) -> None:
    writer.save_tensors(f"img{i}", FakeImage(), {"dual_0": torch.zeros(2)}, {"m": torch.zeros(2)})
    writer.commit(manifest_row(i), tables(i))


def read(name: str) -> list[dict]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_counts() -> set[int]:
    return {len(read(n)) for n in (RESULTS_NAME, WORDS_NAME, TOKENS_NAME)}


try:
    assert load_state(OUT) == set(), "a fresh directory must report nothing done"

    writer = RunWriter(OUT)
    for i in (1, 2, 3):
        write(writer, i)
    writer.close()
    assert len(read(MANIFEST_NAME)) == 3
    assert row_counts() == {15}, row_counts()
    assert load_state(OUT) == {"img1", "img2", "img3"}, "three clean images must resume as done"

    writer = RunWriter(OUT)                                    # interruption A
    writer.save_tensors("img4", FakeImage(), {"dual_0": torch.zeros(2)}, {"m": torch.zeros(2)})
    for name, rows in tables(4).items():
        writer._append(name, rows)
    writer.close()

    writer = RunWriter(OUT)                                    # interruption B
    write(writer, 5)
    writer.close()
    (OUT / "maps_img5.pt").unlink()

    done = load_state(OUT)
    assert done == {"img1", "img2", "img3"}, f"expected three complete, got {sorted(done)}"
    assert len(read(MANIFEST_NAME)) == 3, "the manifest row of the incomplete image must go"
    assert row_counts() == {15}, f"orphan table rows survived: {row_counts()}"
    assert {r["image_id"] for r in read(TOKENS_NAME)} == {"img1", "img2", "img3"}

    writer = RunWriter(OUT)                                    # the retry
    for i in (4, 5):
        write(writer, i)
    writer.close()
    manifest = read(MANIFEST_NAME)
    assert len(manifest) == 5 and len({r["image_id"] for r in manifest}) == 5, "duplicate rows"
    assert row_counts() == {25}, row_counts()
    assert (OUT / MANIFEST_NAME).read_text(encoding="utf-8").count("image_id,profession") == 1
    assert not list(OUT.glob("*.tmp")), "temporary files left behind"
    assert load_state(OUT) == {f"img{i}" for i in range(1, 6)}
    print("main_run_store: all resume/commit assertions passed")
finally:
    shutil.rmtree(OUT, ignore_errors=True)
