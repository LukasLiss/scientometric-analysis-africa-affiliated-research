#!/usr/bin/env python3
"""Rebuild the manuscript's Figure 5: top 20 African institutions, with the
institution's country appended to each label.

Run analyses/00_full_paper_analysis.py first — this script reads the
figures/tables/african_institutions.csv it produces (so the ranking is exactly
the one used everywhere else) and the corpus for the institution->country map.
Writes figures/fig10_top_institutions_countrylabels.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

top = pd.read_csv(ROOT / "figures/tables/african_institutions.csv", index_col=0).head(20)
df = pd.read_csv(ROOT / "data/included_papers.csv", low_memory=False)

split = lambda v: [x.strip() for x in str(v).split(";") if x.strip()] if pd.notna(v) else []
org_country = {}
for _, r in df.iterrows():
    orgs = split(r["Research Organizations - standardized"])
    ctrs = split(r["Country of standardized research organization"])
    for i, o in enumerate(orgs):
        org_country.setdefault(o, ctrs[i] if i < len(ctrs) else "?")

labels = [f"{o} ({org_country.get(o, '?')})" for o in top.index]
vals = top["papers"].tolist()

fig, ax = plt.subplots(figsize=(9.1, 5.0), dpi=300)
y = range(len(labels))
ax.barh(y, vals, color="#2e8b57", height=0.62)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=8.2)
ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v + 0.6, i, str(v), va="center", fontsize=8.2)
ax.set_xlabel("Number of papers", fontsize=9)
ax.set_title("Top 20 African institutions", fontsize=10)
ax.set_xlim(0, 60)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=8.5)
fig.tight_layout()
out = ROOT / "figures/fig10_top_institutions_countrylabels.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
