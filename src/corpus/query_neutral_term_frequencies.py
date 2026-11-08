#!/usr/bin/env python3
"""Corpus frequency of the neutral terms: DWDS frequency API, DWDS full-text search for phrases, Duden
entry check.

Usage (network required): python src/corpus/query_neutral_term_frequencies.py [--candidates]
"""

from __future__ import annotations

import csv
import datetime
import json
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path

import corpus_spec as spec

OUT_PATH = Path("data/corpus/neutral_term_attestation.csv")
FREQ_API = "https://www.dwds.de/api/frequency/"
DSTAR_API = "https://ddc.dwds.de/dstar/dwdsxl/dstar.perl"
DUDEN_URL = "https://www.duden.de/rechtschreibung/{}"
USER_AGENT = "HAW-Hamburg-thesis-corpus-check/1.0"
PAUSE_S = 0.7                      # politeness delay between requests

ATTESTED_PER_M = 0.01
MARGINAL_PER_M = 0.001
DWDS_TOTAL_TOKENS = 53_142_845_055    # DWDS-Gesamtkorpus size as returned by the frequency API;
                                      # used for dstar-rescaled rows (freq-API rows carry their own total)


def _get(url: str, params: dict | None = None, timeout: int = 20) -> tuple[int, bytes]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(PAUSE_S)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def dwds_frequency(term: str) -> dict | None:
    """Lemma frequency in the DWDS-Gesamtkorpus; None if the API has no answer."""
    status, body = _get(FREQ_API, {"q": term})
    if status != 200:
        return None
    d = json.loads(body)
    if d.get("hits") is None:
        return None
    return {"hits": int(d["hits"]), "total": int(d["total"]), "freq_class": d["frequency"]}


def dstar_hits(term: str) -> int | None:
    """Full-text hit count in dstar/dwdsxl; quotes make multi-word terms a (lemma-based) phrase."""
    q = f'"{term}"' if " " in term else term
    status, body = _get(DSTAR_API, {"q": q, "fmt": "json", "limit": 1})
    if status != 200:
        return None
    return json.loads(body).get("nhits_")


def duden_status(term: str) -> int | None:
    """HTTP status of the Duden entry page (200 = lemma exists); None for multi-word terms."""
    if " " in term:
        return None
    status, _ = _get(DUDEN_URL.format(urllib.parse.quote(term)))
    return status


ALT_OUT_PATH = Path("data/corpus/neutral_term_alternatives.csv")
CANDIDATES = [
    # profession, candidate, source, rejection reason
    ("fotograf", "fotografierende Person", "gg",
     "activity- not profession-denoting; participle implies concurrent photographing, clashes with non-shooting scenes (foto_bearbeiten, bilder_sichten); breaks Fachkraft-type parallelism"),
    ("fotograf", "Fotografie betreibende Person", "gg",
     "unattested; +2 words vs. frame"),
    ("fotograf", "Fotografierende", "gg",
     "gender-neutral only in plural; singular 'ein Fotografierender/eine Fotografierende' re-genders"),
    ("fotograf", "Aufnehmende", "gg",
     "plural-only like Fotografierende; ambiguous (aufnehmen != photographing specifically)"),
    ("fotograf", "Fotoschaffende", "dwds",
     "plural-only/collective flavour"),
    ("fotograf", "Lichtbildner", "dwds",
     "is itself a generic masculine (Lichtbildnerin exists); archaic"),
    ("fotograf", "Fachkraft für Fotografie", "dwds",
     "unattested"),
    ("fotograf", "fotografische Fachkraft", "dwds",
     "unattested"),
    ("arzt", "ärztliches Fachpersonal", "gg",
     "collective noun — violates single-actor ('Ein ärztliches Fachpersonal' impossible)"),
    ("arzt", "Person im ärztlichen Dienst", "gg",
     "+3 words, institution-bound; length matching broken"),
    ("arzt", "behandelnde Person", "gg",
     "activity- not profession-denoting; clashes with deliberately patient-free scenes (nobody is treated while reading an X-ray)"),
    ("arzt", "Arztperson", "dwds",
     "equally marginal as ärztliche Fachkraft; Swiss-skewed"),
    ("arzt", "ärztliche Fachperson", "dwds",
     "unattested as phrase; Fachperson alone is Swiss-skewed and underspecified"),
    ("mechaniker", "Kfz-Mechanik-Fachkraft", "gg",
     "coined compound, unattested; no gain over Kfz-Fachkraft"),
    ("mechaniker", "Werkstattfachkraft", "dwds",
     "near-unattested; workshop- not automotive-specific"),
    ("mechaniker", "Fachkraft für Kfz-Technik", "dwds",
     "unattested"),
    ("mechaniker", "Kfz-Mechatroniker", "dwds",
     "official trade title but itself a generic masculine — not a neutralization"),
]


def verdict(per_million: float) -> str:
    if per_million >= ATTESTED_PER_M:
        return "attested"
    if per_million >= MARGINAL_PER_M:
        return "marginal"
    return "unattested"


def main() -> None:
    terms = [(prof, "neutral_term", t) for prof, t in spec.neutral_terms().items()] + \
            [(prof, "generic_masc", t) for prof, t in spec.generic_masc_terms().items()]
    today = datetime.date.today().isoformat()

    raw = {}
    for prof, role, term in terms:
        print(f"querying {term} …")
        raw[term] = {"freq": dwds_frequency(term) if " " not in term else None,
                     "dstar": dstar_hits(term),
                     "duden": duden_status(term)}

    # Scale factor dstar -> frequency-API equivalents, from terms present in both sources.
    ratios = [r["freq"]["hits"] / r["dstar"] for r in raw.values()
              if r["freq"] and r["freq"]["hits"] > 0 and r["dstar"]]
    scale = statistics.median(ratios) if ratios else None
    print(f"\ndstar->DWDS scale factor: median of {len(ratios)} reference ratios = "
          f"{scale:.1f}" if scale else "\nWARNING: no scale references, so no verdicts for multi-word terms")

    rows = []
    for prof, role, term in terms:
        r = raw[term]
        if r["freq"]:                                        # single word: frequency API is primary
            source, hits = "dwds_frequency_api", r["freq"]["hits"]
            total, freq_class = r["freq"]["total"], r["freq"]["freq_class"]
            hits_dwds_equiv = hits
        elif r["dstar"] is not None and scale:               # multi-word: rescaled dstar
            source, hits = "dwds_dstar_dwdsxl", r["dstar"]
            total, freq_class = "", ""
            hits_dwds_equiv = round(r["dstar"] * scale)
        else:
            source, hits, total, freq_class, hits_dwds_equiv = "unavailable", "", "", "", None
        ref_total = total if isinstance(total, int) else DWDS_TOTAL_TOKENS
        per_m = hits_dwds_equiv / ref_total * 1e6 if hits_dwds_equiv is not None else None
        rows.append({
            "profession": prof, "role": role, "term": term, "source": source,
            "hits_raw": hits, "corpus_total": total, "dwds_freq_class": freq_class,
            "hits_dwds_equiv": hits_dwds_equiv,
            "per_million": round(per_m, 5) if per_m is not None else "",
            "duden_http_status": r["duden"] if r["duden"] is not None else "",
            "verdict": verdict(per_m) if per_m is not None else "no_data",
            "retrieved": today,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT_PATH}\n")

    print(f"{'profession':<13}{'term':<26}{'role':<14}{'hits(eq)':>12}{'per_M':>11}"
          f"{'duden':>7}  verdict")
    for row in sorted(rows, key=lambda r: (r["role"], r["profession"])):
        print(f"{row['profession']:<13}{row['term']:<26}{row['role']:<14}"
              f"{row['hits_dwds_equiv'] if row['hits_dwds_equiv'] is not None else '-':>12}"
              f"{row['per_million'] if row['per_million'] != '' else '-':>11}"
              f"{row['duden_http_status'] if row['duden_http_status'] != '' else '-':>7}"
              f"  {row['verdict']}")

    bad = [r for r in rows if r["role"] == "neutral_term" and r["verdict"] != "attested"]
    print("\nNeutral terms:", "all attested." if not bad else
          "not attested: " + ", ".join(f"{r['term']} ({r['verdict']})" for r in bad) + ".")
    return scale


def check_candidates(scale: float | None) -> None:
    today = datetime.date.today().isoformat()
    rows = []
    for prof, term, src, reason in CANDIDATES:
        print(f"querying candidate {term} …")
        if " " not in term:
            f = dwds_frequency(term)
            hits_equiv = f["hits"] if f else None
        else:
            n = dstar_hits(term)
            hits_equiv = round(n * scale) if n is not None and scale else None
        per_m = hits_equiv / DWDS_TOTAL_TOKENS * 1e6 if hits_equiv is not None else None
        rows.append({
            "profession": prof, "candidate": term, "source": src,
            "hits_dwds_equiv": hits_equiv,
            "per_million": round(per_m, 5) if per_m is not None else "",
            "attestation": verdict(per_m) if per_m is not None else "no_data",
            "rejection_reason": reason, "retrieved": today,
        })

    with ALT_OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} considered-and-rejected candidates -> {ALT_OUT_PATH}")
    for r in rows:
        print(f"  {r['profession']:<11} {r['candidate']:<30} [{r['source']}] "
              f"{r['hits_dwds_equiv'] if r['hits_dwds_equiv'] is not None else '-':>8}  {r['attestation']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true",
                    help="also measure the considered-and-rejected alternative terms")
    args = ap.parse_args()
    scale_factor = main()
    if args.candidates:
        check_candidates(scale_factor)
