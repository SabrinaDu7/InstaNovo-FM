"""Our 40M checkpoint's cosine-versus-hyperscore correlation on 20,000 LCFM test spectra.

Rows 22 of docs/references/results_paper_or_rerun.md (40M trained here). Protocol: `COSINE_HYPERSCORE` in _common.py.
"""

from _common import COSINE_HYPERSCORE, run

run(script="result6_40Mours_cosine_hyperscore", model="40M-ours", tier="lcfm", split="test", overrides=COSINE_HYPERSCORE)
