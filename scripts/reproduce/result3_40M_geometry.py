"""The 40M model's embedding geometry on 20,000 LCFM test spectra: embedding statistics (anisotropy, effective rank), UMAP quality, EVoC clustering, cosine-hyperscore correlation, ESM2 alignment, Glass Box attribution (no author numbers; UMAP is qualitative in the paper).

Rows 15 of docs/references/results_paper_or_rerun.md (40M table). Protocol: `GEOMETRY` in _common.py.
"""

from _common import GEOMETRY, run

run(script="result3_40M_geometry", model="40M", tier="lcfm", split="test", protocol=GEOMETRY)
