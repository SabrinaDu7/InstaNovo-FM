"""Our 40M retrained at a global batch of 2,048 (job 23151784), step 90,001 (model_best) on its own tier's test split (MCFM): probes and retrieval.

Rows 16-17 of docs/references/results_paper_or_rerun.md (40M retrained at batch 2,048). Protocol: `PROBES_RETRIEVAL` in _common.py.
"""

from _common import PROBES_RETRIEVAL, run

run(script="result5_40Mb2048s90k_mcfm_test", model="40M-b2048-90k", tier="mcfm", split="test", overrides=PROBES_RETRIEVAL)
