"""Our 40M retrained at a global batch of 2,048 (job 23151784), step 90,001 (model_best): embedding geometry on 20,000 LCFM test spectra: embedding statistics, UMAP, EVoC, cosine-hyperscore, ESM2 alignment, Glass Box.

Rows 15, 18-21 of docs/references/results_paper_or_rerun.md (40M retrained at batch 2,048). Protocol: `GEOMETRY` in _common.py.
"""

from _common import GEOMETRY, run

run(script="result3_40Mb2048s90k_geometry", model="40M-b2048-90k", tier="lcfm", split="test", overrides=GEOMETRY)
