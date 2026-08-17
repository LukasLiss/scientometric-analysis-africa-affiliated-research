#!/usr/bin/env python3
"""LLM extraction of dataset / patient-origin / modality fields from abstracts.

Runs a local Ollama model (gemma3:12b, temperature 0.1) over every paper in
data/included_papers.csv that is not yet in the cache CSV, and appends one row
per paper to data/Papers_step_05_dataset_landscape_llm_cache.csv.

The shipped cache already covers the full corpus, so the analysis pipeline
never needs to call an LLM.  This script documents the exact prompt, model,
and temperature used, and lets you regenerate or extend the cache:
delete rows from the cache CSV (or add papers to the corpus) and re-run.

Requires Ollama running on localhost:11434 with the gemma3:12b model pulled.
"""

import csv
import json
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"
TEMPERATURE = 0.1
CORPUS_PATH = DATA / "included_papers.csv"
CACHE_PATH = DATA / "Papers_step_05_dataset_landscape_llm_cache.csv"
CACHE_COLS = ["Paper_ID", "datasets_llm", "patient_origin_countries_llm",
              "data_modality_llm"]

PROMPT_TEMPLATE = """You are extracting dataset information from a biomedical AI/ML paper.
Given the abstract below, return a JSON object with these keys:
  "datasets": list of named datasets used (e.g., "MIMIC-III", "ABIDE", "ChestX-ray14"). Empty list if none named.
  "patient_origin_countries": list of countries where the patients/data subjects originate. Empty list if not stated.
  "data_modality": list from this allowed set: ["Imaging", "EHR", "Genomics", "Wearables", "Audio", "Text", "Tabular", "Sensor", "Other"].
Return ONLY a JSON object.

ABSTRACT:
{abstract}
"""


def query_ollama(prompt, retries=3, backoff=2.0):
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": TEMPERATURE}}
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(backoff)


def parse_json_block(text):
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def extract_dataset_info(abstract):
    if not isinstance(abstract, str) or not abstract.strip():
        return {"datasets": [], "patient_origin_countries": [], "data_modality": []}
    raw = query_ollama(PROMPT_TEMPLATE.format(abstract=abstract[:6000]))
    parsed = parse_json_block(raw) or {}
    return {k: (parsed.get(k, []) if isinstance(parsed.get(k, []), list) else [])
            for k in ["datasets", "patient_origin_countries", "data_modality"]}


def main():
    corpus = pd.read_csv(CORPUS_PATH, low_memory=False)
    if CACHE_PATH.exists():
        cached_ids = set(pd.read_csv(CACHE_PATH, low_memory=False)["Paper_ID"])
    else:
        cached_ids = set()
    todo = corpus[~corpus["Paper_ID"].isin(cached_ids)]
    print(f"{len(cached_ids)} papers cached, {len(todo)} to extract")
    for _, paper in todo.iterrows():
        out = extract_dataset_info(paper["Abstract"])
        row = {
            "Paper_ID": paper["Paper_ID"],
            "datasets_llm": "; ".join(out["datasets"]),
            "patient_origin_countries_llm": "; ".join(out["patient_origin_countries"]),
            "data_modality_llm": "; ".join(out["data_modality"]),
        }
        new_file = not CACHE_PATH.exists()
        with open(CACHE_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CACHE_COLS)
            if new_file:
                w.writeheader()
            w.writerow(row)
            f.flush()
        print(f'{paper["Paper_ID"]}: {row["datasets_llm"] or "(no named dataset)"}')
    print("done")


if __name__ == "__main__":
    main()
