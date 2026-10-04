"""Our 40M retrained at a global batch of 2,048 (job 23151784), step 80,001 at peak level on 10,000 LCFM test spectra: peak-type classification, confidence-signal AUROC, attention-head structure, IG attribution with reconstruction bin accuracy per ion type.

Rows 11-14 of docs/references/results_paper_or_rerun.md (40M retrained at batch 2,048). Protocol: `PEAK_LEVEL` in _common.py.
"""

from _common import PEAK_LEVEL, run

run(script="result2_40Mb2048_peak_level", model="40M-b2048", tier="lcfm", split="test", overrides=PEAK_LEVEL)
