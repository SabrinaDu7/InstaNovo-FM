"""Our 40M checkpoint at peak level on 10,000 LCFM test spectra: peak-type classification, confidence-signal AUROC, attention-head structure, IG attribution with reconstruction bin accuracy per ion type.

Rows 11-14 of docs/references/results_paper_or_rerun.md (40M trained here). Protocol: `PEAK_LEVEL` in _common.py.
"""

from _common import PEAK_LEVEL, run

run(script="result2_40Mours_peak_level", model="40M-ours", tier="lcfm", split="test", overrides=PEAK_LEVEL)
