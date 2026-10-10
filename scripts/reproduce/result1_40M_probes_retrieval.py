"""The 40M MCFM baseline on the LCFM test split: frozen linear probes and duplicate retrieval, the repository's documented reproduction command (Table S5).

Rows 1-10 of docs/references/results_paper_or_rerun.md (40M table). Protocol: `PROBES_RETRIEVAL` in _common.py.
"""

from _common import PROBES_RETRIEVAL, run

run(script="result1_40M_probes_retrieval", model="40M", tier="lcfm", split="test", protocol=PROBES_RETRIEVAL)
