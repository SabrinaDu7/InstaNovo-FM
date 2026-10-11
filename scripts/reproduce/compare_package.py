"""Does the suite, run as a package, give the numbers the embedded evaluator gave? Metric by metric, from the committed runs.

    python scripts/reproduce/compare_package.py [--reference docs/references/rerun_embedded] [--out docs/references/package_vs_embedded.md]

Both sides are script-layout trees of the repository: the reference is `docs/references/rerun_embedded/<script>/` (the
embedded evaluator's runs, as `docs/references/rerun/` was before the switch) and the results are the suite's runs that
`_common.run` copied into `docs/references/rerun/<script>/` (recognised by the `protocol` key the suite writes in
`run.json`; a script whose directory still holds an embedded run counts as not run). Each tree is laid out the way the
package lays out its results (`<dataset>/<model>/<protocol>/<task>.json`) through symlinks named by RESULT_SCRIPTS
below, and `instanovofm_evals.compare` lists every summary metric as identical, within float noise (relative 1e-4) or
different, and writes the markdown table. Needs no cluster, no data and no environment variable: any clone can check.
`--results <dir>` compares a live package results tree (`$RESULTS/package` on nibi) instead.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from instanovofm_evals.compare import compare

REPO = Path(__file__).resolve().parents[2]
# result script -> (dataset, model, protocol) in the package's layout; the one home of this correspondence.
RESULT_SCRIPTS: dict[str, tuple[str, str, str]] = {
    "result1_40M_probes_retrieval": ("lcfm-test", "40M", "paper-probes-retrieval"),
    "result1_89M_probes_retrieval": ("lcfm-test", "89M", "paper-probes-retrieval"),
    "result2_40M_peak_level": ("lcfm-test", "40M", "paper-peak-level"),
    "result2_89M_peak_level": ("lcfm-test", "89M", "paper-peak-level"),
    "result3_40M_geometry": ("lcfm-test", "40M", "paper-geometry"),
    "result3_89M_geometry": ("lcfm-test", "89M", "paper-geometry"),
    "result4_89M_factorial_validation": ("lcfm-valid", "89M", "paper-probes-retrieval"),
    "result5_40M_mcfm_test": ("mcfm-test", "40M", "paper-probes-retrieval"),
    "validate_released_40M": ("mcfm-valid", "40M", "paper-validation"),
}
# result6 (cosine-hyperscore) is the geometry run's cosine task in the package; its reference files are compared through result3.


def is_suite_run(run_dir: Path) -> bool:
    """A `run.json` written by the suite (`protocol` key); the embedded evaluator's carries `overrides` instead."""
    return (run_dir / "run.json").exists() and "protocol" in json.loads((run_dir / "run.json").read_text())


def script_tree(scripts: Path, out: Path, *, suite_runs_only: bool) -> Path:
    """`<out>/<dataset>/<model>/<protocol>` -> `<scripts>/<script>` symlinks, for every script that has a run there."""
    for script, (dataset, model, protocol) in RESULT_SCRIPTS.items():
        src = scripts / script
        if not src.exists() or (suite_runs_only and not is_suite_run(src)):
            continue
        link = out / dataset / model / protocol
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(src, link.parent))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=Path("docs/references/rerun_embedded"))
    ap.add_argument("--results", type=Path, default=None, help="a live package results tree instead of the committed runs")
    ap.add_argument("--out", type=Path, default=Path("docs/references/package_vs_embedded.md"))
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()
    os.chdir(REPO)  # the written document names its inputs by these paths: repository-relative, the same on every machine
    rerun = Path("docs/references/rerun")
    results = args.results or script_tree(rerun, rerun.parent / "rerun_as_package", suite_runs_only=True)
    reference = script_tree(args.reference, args.reference.parent / "rerun_embedded_as_package", suite_runs_only=False)
    compare(results, reference, ["40M", "89M"], args.out, args.rtol)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
