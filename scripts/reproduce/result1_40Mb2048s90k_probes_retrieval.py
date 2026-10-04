"""Our 40M retrained at a global batch of 2,048 (job 23151784), step 90,001 (model_best) on the LCFM test split: frozen linear probes and duplicate retrieval, the documented reproduction command, beside the released 40M checkpoint's rerun.

Rows 1-10 of docs/references/results_paper_or_rerun.md (40M retrained at batch 2,048). Protocol: `PROBES_RETRIEVAL` in _common.py.
"""

from _common import PROBES_RETRIEVAL, run

run(script="result1_40Mb2048s90k_probes_retrieval", model="40M-b2048-90k", tier="lcfm", split="test", overrides=PROBES_RETRIEVAL)
