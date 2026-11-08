#!/usr/bin/env python3
"""German4All versions of the 36 base scenes at Level 1 (Leichte Sprache).

They are the automatic alternative to writing the GAL frames by hand. Notebook 02 checks them against
the same criteria as the GAL frames; none of them enters the corpus. auto_flags is a quick screen for
two failures that break a minimal pair: the actor noun is lost, or a pronoun appears.

Usage (GPU): python src/corpus/draft_ls_with_german4all.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import re
from pathlib import Path

import corpus_spec as spec

OUT_PATH = Path("data/corpus/ls_drafts.csv")

ADAPTER_ID = "tum-nlp/single_model_xl"
ADAPTER_REVISION = "81326af1187b7dca8c75cb68d7ecf5f0150e827e"
BASE_MODEL_ID = "google/flan-t5-xl"
LEVEL = 1                                                        # Leichte Sprache
GEN_PARAMS = dict(do_sample=True, top_p=0.9, temperature=0.1, max_new_tokens=256)  # model card
BATCH_SIZE = 6

# The prompt of the model card, verbatim with its doubled "Zielgruppe:", so that the input matches the
# documented use. It describes all five levels; generate_drafts then asks for Level 1.
DESCRIPTION = """Verwandle den folgenden Text in die Zielsprachebene gemäß den untenstehenden Beschreibungen.
KOMPLEXITÄTSBESCHREIBUNGEN
Level 1 : Leichte Sprache:
Zielgruppe: Zielgruppe: Personen mit Leseschwierigkeiten, inklusive Menschen mit Lernbehinderungen und solche, die erst kürzlich begonnen haben, Deutsch zu lernen.
Merkmale: Sehr kurze Sätze, nur kurze und häufig verwendete Wörter, direkte Ansprache. Vermeidung von Abkürzungen, Metaphern oder Ironie.
Beispiele: Einfache Anleitungen, barrierefreie Webseiten.

Level 2 : Einfaches Deutsch für Anfänger:
Zielgruppe: Nicht-Muttersprachler mit grundlegenden Deutschkenntnissen.
Merkmale: Einfache Satzstrukturen, Grundwortschatz, klarer Fokus auf wichtige Informationen, Vermeidung kulturspezifischer Ausdrücke.
Beispiele:  Sprachlernmaterialien, einführende Webtexte.

Level 3 : Gebräuchliche Standardsprache:
Zielgruppe: Öffentlichkeit mit unterschiedlichem Bildungsniveau.
Merkmale: Klare, strukturierte Sätze, Verständlichkeit steht im Vordergrund, Vermeidung von Fachjargon.
Beispiele: Breit gefächerte Nachrichtenportale, Blogs.

Level 4 : Gehobene Alltagssprache:
Zielgruppe: Regelmäßige Leser mit gutem Sprachverständnis.
Merkmale: Vielfältigeres Vokabular, gelegentlicher Fachjargon mit Erklärungen, komplexe Satzstrukturen.
Beispiele: Fachblogs, Qualitätszeitungen.

Level 5 : Akademische Sprache:
Zielgruppe: Akademiker und Experten.
Merkmale: Komplexe Satzstrukturen, Fachterminologie, Verwendung von Fachbegriffen.
Beispiele: Fachzeitschriften, wissenschaftliche Publikationen.
"""

FIELDS = ["item_id", "profession", "scene_label", "scene_source", "sd_text", "draft_idx",
          "torch_seed", "draft_text", "auto_flags", "adapter_id", "adapter_revision",
          "base_model", "level", "gen_params", "run_date"]

PRONOUN_RE = re.compile(r"\b([Ee]r|[Ss]ie|[Ee]s)\b")


def base_items() -> list[dict]:
    """The 36 base scenes as generic-masculine sentences. The low-context baseline is left out, since a
    bare noun has no structure to simplify.
    """
    items = []
    for prof in spec.PROFESSIONS:
        actor, num = spec.ACTORS[prof]["generic_masc"]
        for label, vsg, vpl, rest, src in spec.SCENES[prof]:
            items.append({
                "item_id": f"{prof}_{label}",
                "profession": prof,
                "scene_label": label,
                "scene_source": src,
                "sd_text": spec.sentence(actor, num, vsg, vpl, rest),
                "head_noun": spec.strip_article(actor).split("-")[-1],  # "Kfz-Mechaniker" -> "Mechaniker"
            })
    return items


def auto_flags(draft: str, head_noun: str) -> str:
    flags = []
    if head_noun.lower() not in draft.lower():
        flags.append("actor_lost")
    if PRONOUN_RE.search(draft):
        flags.append("pronoun_injected")
    return ";".join(flags)


def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp32, since T5 models overflow in fp16; flan-t5-xl in fp32 fits a 16 GB GPU.
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, ADAPTER_ID, revision=ADAPTER_REVISION).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, revision=ADAPTER_REVISION)
    return model, tokenizer, device


def generate_drafts(items: list[dict], num_drafts: int) -> list[dict]:
    import torch

    model, tokenizer, device = load_model()
    prefix = f"\n Paraphrasiere den folgenden Text auf Level {LEVEL}.\n Text: "
    run_date = datetime.date.today().isoformat()
    rows = []
    for draft_idx in range(num_drafts):
        torch.manual_seed(draft_idx)
        for start in range(0, len(items), BATCH_SIZE):
            batch = items[start:start + BATCH_SIZE]
            inputs = tokenizer([DESCRIPTION + prefix + it["sd_text"] for it in batch],
                               return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, **GEN_PARAMS)
            for it, ids in zip(batch, out):
                draft = tokenizer.decode(ids, skip_special_tokens=True).strip()
                rows.append({
                    "item_id": it["item_id"], "profession": it["profession"],
                    "scene_label": it["scene_label"], "scene_source": it["scene_source"],
                    "sd_text": it["sd_text"], "draft_idx": draft_idx, "torch_seed": draft_idx,
                    "draft_text": draft, "auto_flags": auto_flags(draft, it["head_noun"]),
                    "adapter_id": ADAPTER_ID, "adapter_revision": ADAPTER_REVISION,
                    "base_model": BASE_MODEL_ID, "level": LEVEL,
                    "gen_params": ";".join(f"{k}={v}" for k, v in GEN_PARAMS.items()),
                    "run_date": run_date,
                })
            done = min(start + BATCH_SIZE, len(items))
            print(f"  draft {draft_idx}: {done}/{len(items)} scenes")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-drafts", type=int, default=1,
                    help="drafts per scene (distinct torch seeds; default 1)")
    args = ap.parse_args()

    items = base_items()
    print(f"Drafting {len(items)} base scenes x {args.num_drafts} draft(s) "
          f"with {ADAPTER_ID}@{ADAPTER_REVISION[:8]} (Level {LEVEL}) ...")
    rows = generate_drafts(items, args.num_drafts)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    flagged = [r for r in rows if r["auto_flags"]]
    print(f"\nWrote {len(rows)} drafts -> {OUT_PATH}")
    print(f"Auto-flagged drafts (heuristic):"
          f" {len(flagged)}/{len(rows)}")
    for r in rows:
        mark = f"  [{r['auto_flags']}]" if r["auto_flags"] else ""
        print(f"\n{r['item_id']} (draft {r['draft_idx']}){mark}\n  GSL: {r['sd_text']}\n  GAL draft: {r['draft_text']}")


if __name__ == "__main__":
    main()
