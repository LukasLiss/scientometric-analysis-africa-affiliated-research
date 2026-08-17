# Scientometric Analysis of African-Affiliated Research about AI in Healthcare

Data and analysis code for the scientometric study of machine-learning-for-health
research with at least one African-affiliated author, published 2023–2025.
Running the pipeline below regenerates every figure, table, and statistic
reported in the paper.

## Contents

```
data/
  paper_review_process.csv       All 4,985 screened records: Paper_ID, first/second
                                 reviewer, first/second review decision, followed by
                                 the full Dimensions metadata of each paper.
  included_papers.csv            The analysed corpus: the 1,158 papers with an
                                 include decision in both review rounds.
  Papers_step_05_dataset_landscape_llm_cache.csv
                                 Cached LLM-extracted dataset names, patient-origin
                                 countries, and data modalities per paper.
  openalex_author_cache.json     Cached OpenAlex author-enrichment responses
                                 (retrieved June–August 2026).
analyses/
  00_full_paper_analysis.py      The full analysis pipeline (single entry point).
  00_full_paper_analysis.ipynb   The same analysis as an executed, cell-by-cell
                                 notebook for step-by-step inspection.
  llm_dataset_extraction.py      Provenance: the exact LLM prompt, model, and
                                 temperature used to build the dataset-landscape
                                 cache (local Ollama, gemma3:12b, temperature 0.1).
  fig05_institutions_with_countries.py
                                 Rebuilds the manuscript's Figure 5 variant of the
                                 institution ranking (country names in the labels);
                                 run after the main pipeline.
requirements.txt                 Pinned Python dependencies (tested with Python 3.13).
```

## Reproduce the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python analyses/00_full_paper_analysis.py
```

This reads `data/included_papers.csv` plus the two caches and writes:

- `figures/fig01…fig23_*.png|.pdf` — every figure in the paper,
- `figures/tables/*.csv` — machine-readable versions of the paper's tables,
- `figures/stats.json` — every scalar statistic cited in the manuscript text.

Alternatively, open `analyses/00_full_paper_analysis.ipynb` to run the same
analysis section by section.

No network access is required: the OpenAlex author enrichment and the
LLM-extracted dataset fields are fully covered by the shipped caches. (The
OpenAlex step only issues requests for authors missing from the cache; set
`ENABLE_OPENALEX = False` in the script to skip it entirely.)

## Screening provenance

Records were retrieved from the Dimensions database (January 2026; the exact
query string is given in the paper) and screened in two review rounds.
`data/paper_review_process.csv` is the complete screening record: one row per
retrieved paper with both reviewers' decisions. 

The LLM extraction of dataset names, patient-origin countries, and data
modalities from abstracts (`llm_dataset_extraction.py`) requires a local
[Ollama](https://ollama.com) server with the `gemma3:12b` model and is only
needed to extend or rebuild the cache — the shipped cache already covers the
full corpus.