"""The published 89M model at peak level on 10,000 LCFM test spectra: peak-type classification (77.5 % / macro-F1 0.575), cross-spectrum AUROC (0.837), confidence-signal AUROC (0.658), attention-head structure, IG attribution with bin accuracy per ion type (70.1 %, y 74.3 %, b 59.2 %; 89.8 / 196.4 ppm).

Rows 14-25 of docs/references/results_paper_or_rerun.md (89M table). Protocol: `PEAK_LEVEL` in _common.py.
"""

from _common import PEAK_LEVEL, run

run(script="result2_89M_peak_level", model="89M", tier="lcfm", split="test", overrides=PEAK_LEVEL)
