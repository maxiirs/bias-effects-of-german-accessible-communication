"""Definition of the constructed corpus. Professions and scenes are the tables in data/corpus/
(stereotype_value_de is the share of women in the German norms of Misersky et al. 2014); this module
holds the design constants and turns the tables into prompts.
"""

from __future__ import annotations

import csv
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"

SEEDS = [0, 1000, 2000, 3000, 4000]
CALIB_SCENES_PER_PROF = 6
LOW_CONTEXT = True                 # add a bare "{Actor}." baseline scene per profession

GENDER_FORMS = ["generic_masc", "neutral_term", "pair_form", "typographical"]
CALIB_ROLES = ["male_explicit", "female_explicit"]
CALIB_ACTORS = {"male_explicit": "Ein Mann", "female_explicit": "Eine Frau"}
PROVENANCE = {"sd": "authored_sd", "ls": "authored_ls_din_spec"}

FIELDS = ["image_id", "profession", "stereotype_direction", "stereotype_value_de", "scene_label",
          "scene_source", "set", "condition", "variety", "seed", "prompt_text", "n_words", "n_chars",
          "ls_provenance", "qa_flag"]


def _read(name: str) -> list[dict]:
    with (CORPUS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_PROFS = _read("professions.csv")
_SCENES = _read("scenes.csv")

PROFESSIONS = {p["profession"]: p["stereotype_direction"] for p in _PROFS}
STEREO_VALUE_DE = {p["profession"]: float(p["stereotype_value_de"]) for p in _PROFS}
NEUTRAL_TERM_QA = {p["profession"]: p["qa_flag"] for p in _PROFS}
ACTORS = {p["profession"]: {**{f: (p[f], "sg") for f in GENDER_FORMS},
                            **{r: (a, "sg") for r, a in CALIB_ACTORS.items()}}
          for p in _PROFS}
_SOC = {p["profession"]: p["onet_soc"] for p in _PROFS}
SCENES = {prof: [(s["scene_label"], s["verb_sg"], s["verb_pl"], s["rest"],
                  f"O*NET {_SOC[prof]} {s['onet_rank']}")
                 for s in _SCENES if s["profession"] == prof]
          for prof in PROFESSIONS}
LS_FRAMES = {prof: {s["scene_label"]: s["gal_frame"] for s in _SCENES if s["profession"] == prof}
             for prof in PROFESSIONS}


def sentence(actor: str, num: str, verb_sg: str, verb_pl: str, rest: str) -> str:
    return f"{actor} {verb_sg if num == 'sg' else verb_pl} {rest}"


def strip_article(noun_phrase: str) -> str:
    """'Eine Bäckereifachkraft' -> 'Bäckereifachkraft'; 'Ein Bäcker' -> 'Bäcker'."""
    article, _, term = noun_phrase.partition(" ")
    assert article in ("Ein", "Eine", "Ein*e"), noun_phrase
    return term


def definite_actor(noun_phrase: str) -> str:
    """Indefinite first-mention NP -> definite subsequent-mention NP: 'Ein Kfz-Mechaniker' -> 'Der Kfz-
    Mechaniker'; 'Eine Kfz-Mechanikerin oder ein Kfz-Mechaniker' -> 'Die Kfz-Mechanikerin oder der
    Kfz-Mechaniker'; 'Ein*e X*in' -> 'Der*die X*in'.
    """
    articles = {"Ein": "Der", "Eine": "Die", "Ein*e": "Der*die"}
    article, _, rest = noun_phrase.partition(" ")
    assert article in articles, noun_phrase
    return (f"{articles[article]} {rest}"
            .replace(" oder ein ", " oder der ").replace(" oder eine ", " oder die "))


def ls_sentence(template: str, prof: str, form: str) -> str:
    """Instantiate an LS frame template for one gender form / calibration role."""
    actor, _num = ACTORS[prof][form]
    return template.replace("{ACTOR}", actor).replace("{ACTOR_DEF}", definite_actor(actor))


def neutral_terms() -> dict[str, str]:
    """profession -> neutral term without article (attestation query)."""
    return {p: strip_article(ACTORS[p]["neutral_term"][0]) for p in PROFESSIONS}


def generic_masc_terms() -> dict[str, str]:
    """profession -> generic-masculine term without article (attestation baseline)."""
    return {p: strip_article(ACTORS[p]["generic_masc"][0]) for p in PROFESSIONS}


def _seed_rows(prof: str, scene_label: str, src: str, cond: str, variety: str,
               prompt: str) -> list[dict]:
    return [{
        "image_id": f"{prof}_{scene_label}_{cond}_{variety}_seed{seed}",
        "profession": prof, "stereotype_direction": PROFESSIONS[prof],
        "stereotype_value_de": STEREO_VALUE_DE[prof],
        "scene_label": scene_label, "scene_source": src,
        "set": "test" if cond in GENDER_FORMS else "calibration", "condition": cond,
        "variety": variety, "seed": seed, "prompt_text": prompt,
        "n_words": len(prompt.split()), "n_chars": len(prompt),
        "ls_provenance": PROVENANCE[variety],
        "qa_flag": NEUTRAL_TERM_QA[prof] if cond == "neutral_term" else "",
    } for seed in SEEDS]


def _conditions(scene_index: int) -> list[str]:
    """The four test forms, plus the calibration roles in the first CALIB_SCENES_PER_PROF scenes."""
    return GENDER_FORMS + (CALIB_ROLES if scene_index < CALIB_SCENES_PER_PROF else [])


def make_rows() -> list[dict]:
    rows = []
    for prof in PROFESSIONS:
        for si, (label, vsg, vpl, rest, src) in enumerate(SCENES[prof]):
            for cond in _conditions(si):
                actor, num = ACTORS[prof][cond]
                rows += _seed_rows(prof, label, src, cond, "sd", sentence(actor, num, vsg, vpl, rest))
        if LOW_CONTEXT:                                        # bare baseline (RQ1/RQ3 + output only)
            for form in GENDER_FORMS:
                rows += _seed_rows(prof, "low_context", "", form, "sd", f"{ACTORS[prof][form][0]}.")
    return rows


def make_ls_rows() -> list[dict]:
    rows = []
    for prof in PROFESSIONS:
        for si, (label, *_, src) in enumerate(SCENES[prof]):
            for cond in _conditions(si):
                rows += _seed_rows(prof, label, src, cond, "ls",
                                   ls_sentence(LS_FRAMES[prof][label], prof, cond))
    return rows


def unique_test_prompts(include_low_context: bool = False) -> list[dict]:
    """Test-set rows deduplicated across seeds (one row per profession x scene x form), the unit for
    item-control statistics; seed repetitions would only inflate n.
    """
    return [r for r in make_rows()
            if r["set"] == "test" and r["seed"] == SEEDS[0]
            and (include_low_context or r["scene_label"] != "low_context")]
