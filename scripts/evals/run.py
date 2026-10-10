"""Run one protocol of the evaluation suite on one of our exported datasets with one checkpoint.

    source .envrc && python scripts/evals/run.py --dataset ms2bac --model 40M --protocol retrieval
    sbatch --job-name=ev_ms2bac_40M_retrieval scripts/evals/submit.sh --dataset ms2bac --model 40M --protocol retrieval

The suite is `instanovofm_evals` (a dependency of this package, see `pyproject.toml`); this script is its
`run("instanovo-fm", ...)` with the checkpoint tags of `scripts/reproduce/_common.py::MODELS` and the results written
into this repository, `docs/results/rerun/<dataset>/<model>/<protocol>/`, where `render_results.py` reads them. The
datasets are the exports of `prepare_dataset.py` under `$EXTERNAL/<dataset>/`; the protocols (probes, retrieval,
geometry, peak_level, unlabelled, validation) and what each evaluates are defined in the package's `protocols.py` and
described in its README. `$EXTERNAL`, `$EVAL_WORKDIR` and `$CHECKPOINTS` come from `.envrc`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from instanovofm_evals import run as suite_run
from instanovofm_evals.protocols import DATASETS, PROTOCOLS as SUITE_PROTOCOLS, is_paper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproduce"))
from _common import MODELS, REPO, env_path  # noqa: E402

PROTOCOLS = tuple(p for p in SUITE_PROTOCOLS if not is_paper(p)) + ("validation",)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--protocol", choices=PROTOCOLS, required=True)
    args = ap.parse_args()
    os.environ.setdefault("INSTANOVO_FM_DATA_DIR", str(REPO / "data"))
    suite_run(
        "instanovo-fm",
        checkpoint=env_path("CHECKPOINTS") / f"{MODELS[args.model]}.ckpt",
        name=args.model,
        dataset=args.dataset,
        protocol=args.protocol,
        results=REPO / "docs" / "results" / "rerun",
    )


if __name__ == "__main__":
    main()
