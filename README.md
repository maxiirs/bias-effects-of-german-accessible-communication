# Bias Effects of German Accessible Communication

Code and results of the master's thesis *"Behind the Scenes of Text-to-Image Models: Bias effects of German Accessible Communication"* (HAW Hamburg, 2026).

The thesis builds a controlled corpus of minimal-pair prompts (language form × gender form × profession) and measures inside FLUX.2-klein-4B how strongly the actor concept is entangled with gender and how much attention the actor phrase receives.

It builds on an earlier output-level study, [`easy-language-gender-bias`](https://github.com/maxiirs/easy-language-gender-bias), and reuses its generation setup and its MiVOLO readout.

## From thesis to code

Every section of the thesis that rests on data, with the notebook that produces its figures and tables and the code and data behind it.

| Section | Notebook | Code and data |
|---|---|---|
| **Corpus Construction** | | |
| Requirements | | `data/corpus/misersky_de_stereotypes.csv` |
| Constructed Stimuli rather than Found Text | `01_constructed_stimuli` | `src/corpus/mine_elgepa_candidates.py`, `src/corpus/readability.py` |
| Operationalisation of the Scenes | | `data/corpus/scenes.csv` |
| Operationalisation of the Gender Forms | | `src/corpus/query_neutral_term_frequencies.py`, `data/corpus/professions.csv` |
| Operationalisation of GAL | `02_gal_operationalisation` | `src/corpus/draft_ls_with_german4all.py`, `src/corpus/readability.py` |
| The Resulting Corpus | `03_resulting_corpus` | `src/corpus/corpus_spec.py`, `src/corpus/build_prompts.py` |
| **Experimental Setup** | | |
| Image Generation Setup | `04_image_generation_setup` | `src/generation/run_full_corpus.py`, `src/generation/token_alignment.py` |
| Output-Level Classification | | `src/measurement/classify_depicted_gender.py`, `src/measurement/validate_gender_classifier.py` |
| Attention Extraction via Forward Hooks | | `src/generation/forward_hooks.py`, `src/generation/token_alignment.py`, `src/generation/verify_attention_reconstruction.py` |
| Choice of Attention Layers | `05_attention_layers` | `src/generation/run_attention_controls.py`, `src/generation/control_stimuli.py` |
| The Dominance Metric | `05_attention_layers` | `src/measurement/dominance.py` |
| Representation Extraction via Forward Hooks | | `src/generation/forward_hooks.py` |
| Choice of the Measurement Space | `06_representation_layers` | `src/generation/run_concept_word_probe.py`, `data/concept_word_probe/` |
| The Entanglement Metric, Choice of Representation Layers | `06_representation_layers` | `src/measurement/gender_direction.py`, `src/measurement/score_entanglement.py` |
| Analysis of the Research Questions | `09_research_question_tests` | |
| **Results** | | |
| The Output-Level Classification | `07_output_level_classification`, `09_research_question_tests` | `src/measurement/classify_depicted_gender.py` |
| Conjunct Order of the Pair Form | `07_output_level_classification` | `src/generation/run_order_swap.py` |
| Entanglement of the Actor Concept and the Gender Concept | `08_entanglement`, `09_research_question_tests` | `src/measurement/score_entanglement.py` |
| Generic Masculine against Neutral Term | `09_research_question_tests` | `src/measurement/score_entanglement.py` |
| Dominance across Language Forms | `09_research_question_tests` | `src/measurement/dominance.py` |
| Interaction of Language Form and Gender Form | `09_research_question_tests` | `src/measurement/score_entanglement.py` |
| **Appendix:** Scene Inventory | | `data/corpus/scenes.csv` |

Every notebook opens with a cell that says what it needs to run. The code in `src/` follows the path of the data: `corpus/`, `generation/`, `measurement/`, and `publish/` for the export to Hugging Face. `data/` holds the corpus and the run tables, `exports/` and `figures/` everything the notebooks write.

## Data

The images and the per-image representations of the main run are on Hugging Face: [`maxrs/flux2-gal-gsl-images`](https://huggingface.co/datasets/maxrs/flux2-gal-gsl-images) and [`maxrs/flux2-gal-gsl-tensors`](https://huggingface.co/datasets/maxrs/flux2-gal-gsl-tensors). The notebooks load what they need at a pinned revision; see the dataset cards for their licence.

Three external corpora enter only the ecological comparison and the ELGEPA figure (notebook 01), and their results are stored in `data/corpus/anchor_candidates_raw.csv` and `exports/ecological_register_comparison.csv`: ELGEPA (Schomacker et al., 2025), DEplain (Stodden et al., 2023) and LeiKo (Jablotschkin and Zinsmeister). A missing corpus is carried over from the stored table.

## Running

Create the environments from the repository root:

```bash
conda env create -n easy-language -f environment-easy-language.yml
conda env create -n mivolo -f environment-mivolo.yml
```

Notebooks: the `easy-language` environment, started from `notebooks/`.

GPU scripts (generation, attention probes, German4All drafts): the same environment on a CUDA GPU; the runs of the thesis used a Tesla V100 16 GB. The MiVOLO scripts need the `mivolo` environment (`environment-mivolo.yml`) and the MiVOLO repository and weights, which they look for in a checkout of `easy-language-gender-bias` next to this repository, set up as its README describes. `src/corpus/mine_elgepa_candidates.py` and notebook 01 read the ELGEPA pairs from the same checkout.

## Licence

The code is released under the MIT licence, see `LICENSE`.
