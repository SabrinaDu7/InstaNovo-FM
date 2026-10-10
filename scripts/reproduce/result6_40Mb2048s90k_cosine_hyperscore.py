"""Our 40M retrained at a global batch of 2,048 (job 23151784), step 90,001 (model_best): cosine-versus-hyperscore correlation on 20,000 LCFM test spectra.

Rows 22 of docs/references/results_paper_or_rerun.md (40M retrained at batch 2,048). Protocol: `COSINE_HYPERSCORE` in _common.py.
"""

from _common import COSINE_HYPERSCORE, run

run(script="result6_40Mb2048s90k_cosine_hyperscore", model="40M-b2048-90k", tier="lcfm", split="test", protocol=COSINE_HYPERSCORE)
