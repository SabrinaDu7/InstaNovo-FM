"""Our 40M checkpoint on its own tier's test split (MCFM): probes and retrieval.

Rows 16-17 of docs/references/results_paper_or_rerun.md (40M trained here). Protocol: `PROBES_RETRIEVAL` in _common.py.
"""

from _common import PROBES_RETRIEVAL, run

run(script="result5_40Mours_mcfm_test", model="40M-ours", tier="mcfm", split="test", protocol=PROBES_RETRIEVAL)
