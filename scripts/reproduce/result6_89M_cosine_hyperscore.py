"""The 89M model's cosine-versus-hyperscore correlation on 20,000 LCFM test spectra (no author number): does closeness in embedding space track database-match quality.

Row 46 of docs/references/results_paper_or_rerun.md (89M table). Protocol: `COSINE_HYPERSCORE` in _common.py.
"""

from _common import COSINE_HYPERSCORE, run

run(script="result6_89M_cosine_hyperscore", model="89M", tier="lcfm", split="test", overrides=COSINE_HYPERSCORE)
