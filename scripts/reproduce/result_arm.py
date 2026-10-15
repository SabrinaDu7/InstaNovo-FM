"""One result script for every model tag: the paper protocols of `result1`, `result2`, `result3` and `result5` on any
checkpoint of `_common.MODELS`, writing the same `docs/references/rerun/result<N>_<tag>_<name>/` directories the
per-model scripts write, so `fill_results.py` and `arch_results.py` read them unchanged.

    sbatch --time=02:59:00 --job-name=r1_arch-seed102 scripts/reproduce/submit.sh scripts/reproduce/result_arm.py --model arch-seed102 --result 1
"""

from __future__ import annotations

import argparse

from _common import GEOMETRY, MODELS, PEAK_LEVEL, PROBES_RETRIEVAL, run

# result number -> (script suffix, tier, split, protocol); the same four runs every checkpoint so far has had.
RESULTS = {
    1: ("probes_retrieval", "lcfm", "test", PROBES_RETRIEVAL),
    2: ("peak_level", "lcfm", "test", PEAK_LEVEL),
    3: ("geometry", "lcfm", "test", GEOMETRY),
    5: ("mcfm_test", "mcfm", "test", PROBES_RETRIEVAL),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--result", required=True, type=int, choices=sorted(RESULTS))
    args = ap.parse_args()
    suffix, tier, split, protocol = RESULTS[args.result]
    run(
        script=f"result{args.result}_{args.model}_{suffix}",
        model=args.model,
        tier=tier,
        split=split,
        protocol=protocol,
    )


if __name__ == "__main__":
    main()
