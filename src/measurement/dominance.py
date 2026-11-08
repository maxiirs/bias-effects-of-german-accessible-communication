"""Dominance D of the actor phrase against the verb, per image and dual-stream block, and the
test-grid readout it is computed on. Shared by notebook 09 and the dataset export.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))

import corpus_spec  # noqa: E402

GM, NT = "generic_masc", "neutral_term"
ENT_BLOCKS = ("single_4", "single_5")         # primary, replication
DOM_BLOCKS = (2, 4)                           # primary, replication
PARTICLES = {"ab", "an", "auf", "aus", "ein", "zu", "mit", "vor", "nach", "weg", "zurück",
             "fest", "her", "hin", "los", "um", "bei"}


def load_readout(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "mivolo_results.csv")
    df = df[(df["set"] == "test") & (df["scene_label"] != "low_context")].copy()
    df["face"] = df["face_available"].astype(str) == "True"
    return df


def verb_spans() -> dict[tuple, tuple[str, str | None]]:
    """(profession, scene) -> (finite verb, particle or None) for the scenes whose GSL finite verb
    heads the first clause of the GAL frame as well.
    """
    spans = {}
    for prof, scenes in corpus_spec.SCENES.items():
        for label, verb, _pl, rest, *_ in scenes:
            last = rest.rstrip(".").split()[-1]
            particle = last if last in PARTICLES else None
            ls_first = corpus_spec.ls_sentence(corpus_spec.LS_FRAMES[prof][label], prof, GM)
            first_words = ls_first.split(".")[0].split()
            if verb not in first_words:
                continue
            assert particle is None or first_words[-1] == particle, (prof, label, ls_first)
            spans[(prof, label)] = (verb, particle)
    return spans


def dominance(run_dir: Path, image_ids: set[str], readout: pd.DataFrame) -> pd.DataFrame:
    """D per image and dual-stream block for the matched scenes."""
    spans = verb_spans()
    words = pd.read_csv(run_dir / "word_shares_long.csv")
    words = words[words["image_id"].isin(image_ids) & words["layer_idx"].isin(DOM_BLOCKS)]
    est = pd.read_csv(run_dir / "dominance_results.csv").set_index(["image_id", "layer_idx"])
    meta = readout.set_index("image_id")
    rows = []
    for (image_id, layer), grp in words.sort_values("word_index").groupby(["image_id", "layer_idx"]):
        m = meta.loc[image_id]
        key = (m["profession"], m["scene_label"])
        if key not in spans:
            continue
        verb, particle = spans[key]
        w = grp["word"].tolist()
        i = w.index(verb)
        first_sentence_end = w.index(".")
        actor_share, actor_tokens = grp["share"].iloc[:i].sum(), grp["n_tokens"].iloc[:i].sum()
        run = est.loc[(image_id, layer)]
        assert actor_tokens == run["n_mention_tokens_first"], (image_id, layer)
        assert math.isclose(actor_share, run["share_first"], rel_tol=1e-3, abs_tol=2e-6), \
            (image_id, layer, actor_share, run["share_first"])
        verb_idx = [i] + ([first_sentence_end - 1] if particle else [])
        if particle:
            assert w[first_sentence_end - 1] == particle, (image_id, w)
        verb_share = grp["share"].iloc[verb_idx].sum()
        verb_tokens = grp["n_tokens"].iloc[verb_idx].sum()
        rows.append({"image_id": image_id, "layer_idx": layer,
                     "actor_per_token": actor_share / actor_tokens,
                     "verb_per_token": verb_share / verb_tokens,
                     "D": math.log((actor_share / actor_tokens) / (verb_share / verb_tokens))})
    return pd.DataFrame(rows)
