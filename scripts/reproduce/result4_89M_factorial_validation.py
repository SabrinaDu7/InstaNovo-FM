"""The published 89M model (the TS-noPA column of Table S4) on the LCFM validation split with the probe-and-retrieval protocol; the ablation checkpoints follow once this row matches.

Rows 26-36 of docs/references/results_paper_or_rerun.md (factorial ablation, LCFM validation). Protocol: `PROBES_RETRIEVAL` in _common.py.
"""

from _common import PROBES_RETRIEVAL, run

run(script="result4_89M_factorial_validation", model="89M", tier="lcfm", split="valid", protocol=PROBES_RETRIEVAL)
