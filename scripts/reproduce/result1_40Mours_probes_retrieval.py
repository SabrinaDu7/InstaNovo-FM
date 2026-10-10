"""Our 40M checkpoint (job 22701776) on the LCFM test split: frozen linear probes and duplicate retrieval, the documented reproduction command, beside the released 40M checkpoint's rerun.

Rows 1-10 of docs/references/results_paper_or_rerun.md (40M trained here). Protocol: `PROBES_RETRIEVAL` in _common.py.
"""

from _common import PROBES_RETRIEVAL, run

run(script="result1_40Mours_probes_retrieval", model="40M-ours", tier="lcfm", split="test", protocol=PROBES_RETRIEVAL)
