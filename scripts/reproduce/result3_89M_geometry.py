"""The published 89M model's embedding geometry on 20,000 LCFM test spectra: embedding statistics, UMAP quality, EVoC clustering, cosine-hyperscore correlation, ESM2 alignment, Glass Box attribution.

Rows none with an author number; Fig. 3A is the UMAP of docs/references/results_paper_or_rerun.md (89M table). Protocol: `GEOMETRY` in _common.py.
"""

from _common import GEOMETRY, run

run(script="result3_89M_geometry", model="89M", tier="lcfm", split="test", overrides=GEOMETRY)
