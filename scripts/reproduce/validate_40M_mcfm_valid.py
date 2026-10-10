"""Row 23 of docs/references/results_paper_or_rerun.md (40M table): the trainer's validation pass on 250 batches of the
MCFM validation split, the way the released checkpoint was validated in Step 2 of
docs/agent_logs/session-2026-09-25-train40M.md. Protocol: `VALIDATION` in _common.py."""

from _common import VALIDATION, run

run(script="validate_released_40M", model="40M", tier="mcfm", split="valid", protocol=VALIDATION)
