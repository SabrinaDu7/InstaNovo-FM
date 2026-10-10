"""The published 89M model on the LCFM test split: frozen linear probes and duplicate retrieval, the repository's documented reproduction command (Tables S5 and S10).

Rows 1-11 of docs/references/results_paper_or_rerun.md (89M table). Protocol: `PROBES_RETRIEVAL` in _common.py.
"""

from _common import PROBES_RETRIEVAL, run

run(script="result1_89M_probes_retrieval", model="89M", tier="lcfm", split="test", protocol=PROBES_RETRIEVAL)
