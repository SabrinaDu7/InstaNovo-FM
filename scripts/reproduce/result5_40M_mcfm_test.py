"""The 40M model on its own tier's test split (MCFM): probes and retrieval, a number the paper does not report and the reproduction training run will be compared against.

Rows none (our own number) of docs/references/results_paper_or_rerun.md (40M on its training tier). Protocol: `PROBES_RETRIEVAL` in _common.py.
"""

from _common import PROBES_RETRIEVAL, run

run(script="result5_40M_mcfm_test", model="40M", tier="mcfm", split="test", overrides=PROBES_RETRIEVAL)
