"""The 40M model at peak level on 10,000 LCFM test spectra: peak-type classification, confidence-signal AUROC, attention-head structure, IG attribution with its reconstruction bin accuracy per ion type (no author numbers exist for this model).

Rows 11-14 of docs/references/results_paper_or_rerun.md (40M table). Protocol: `PEAK_LEVEL` in _common.py.
"""

from _common import PEAK_LEVEL, run

run(script="result2_40M_peak_level", model="40M", tier="lcfm", split="test", overrides=PEAK_LEVEL)
