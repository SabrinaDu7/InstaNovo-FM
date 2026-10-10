"""Does the suite, run as a package, give the numbers the embedded evaluator gave? Metric by metric.

    source .envrc && python scripts/reproduce/compare_package.py [--reference docs/references/rerun_embedded] [--out docs/references/package_vs_embedded.md]

The reference is the tree `docs/references/rerun/` as it was before the switch (the embedded evaluator's runs, kept
under `docs/references/rerun_embedded/`), laid out the way the package lays out its results
(`<dataset>/<model>/<protocol>/<task>.json`) through symlinks named by RESULT_SCRIPTS below; the results are the
package's runs under `$RESULTS/package/`. `instanovofm_evals.compare` then lists every summary metric as identical,
within float noise (relative 1e-4) or different, and writes the markdown table.
"""

from __future__ import annotations

import argparse
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


def reference_tree(reference: Path, out: Path) -> Path:
    """`<out>/<dataset>/<model>/<protocol>` -> `<reference>/<script>` symlinks, for every script with a reference."""
    for script, (dataset, model, protocol) in RESULT_SCRIPTS.items():
        src = reference / script
        if not src.exists():
            continue
        link = out / dataset / model / protocol
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(src, link.parent))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=REPO / "docs" / "references" / "rerun_embedded")
    ap.add_argument("--results", type=Path, default=None, help="defaults to $RESULTS/package")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "references" / "package_vs_embedded.md")
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()
    results = args.results or Path(os.path.expandvars(os.path.expanduser(os.environ["RESULTS"]))) / "package"
    tree = reference_tree(args.reference, args.reference.parent / "rerun_embedded_as_package")
    compare(results, tree, ["40M", "89M"], args.out, args.rtol)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
