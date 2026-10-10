"""Our 40M checkpoint's embedding geometry on 20,000 LCFM test spectra: embedding statistics, UMAP, EVoC, cosine-hyperscore, ESM2 alignment, Glass Box.

Rows 15, 18-21 of docs/references/results_paper_or_rerun.md (40M trained here). Protocol: `GEOMETRY` in _common.py.
"""

from _common import GEOMETRY, run

run(script="result3_40Mours_geometry", model="40M-ours", tier="lcfm", split="test", protocol=GEOMETRY)
