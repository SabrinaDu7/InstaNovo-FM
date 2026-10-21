"""Shared plumbing for `scripts/reproduce/result*.py`.

Every result script names one checkpoint, one corpus split and one of the paper's protocols. The evaluation itself is
`instanovofm_evals`, the suite extracted from this package into its own repository (a dependency, see
`pyproject.toml`): `instanovofm_evals.run("instanovo-fm", ...)` selects the spectra, builds the labels, runs the tasks
and writes `<task>.json`, `<task>.results.json` and `run.json` under `$RESULTS/package/<dataset>/<model>/<protocol>/`.
This module copies those files into `docs/references/rerun/<script>/`, so the numbers quoted in
`docs/references/results_paper_or_rerun.md` travel with the repository and `fill_results.py` reads them as before.

    source .envrc && python scripts/reproduce/result1_40M_probes_retrieval.py      # on a GPU node
    sbatch --job-name=result1_40M scripts/reproduce/submit.sh scripts/reproduce/result1_40M_probes_retrieval.py

The protocols (which spectra, caps, seed, batch size, tasks and their settings) are defined once, in the package's
`protocols.py`; the names below are theirs. `$CORPUS`, `$EVAL_WORKDIR`, `$EXTERNAL` and the `$CKPT_<TAG>` checkpoint paths
come from `.envrc`.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from instanovofm_evals import run as suite_run

REPO = Path(__file__).resolve().parents[2]
ARMS: tuple[str, ...] = ("seed102", "meta-precursor", "pa-mask2mask", "enc-fourier", "ionladder", "ffn3072", "enc-dual", "peaks300")
"""The architecture study's arms (docs/arch_study/README.md); `arch-<arm>` below is step 90,001 of run `arch-<arm>`."""
# `--model` tags. Tag T's checkpoint is `$CKPT_<T>` (T upper-cased, '-' as '_'); .envrc.example lists each with the run behind it.
MODELS: tuple[str, ...] = (
    "40M", "89M", "89M-ts-pa", "89M-sa-nopa", "89M-sa-pa", "40M-ours", "40M-b2048", "40M-b2048-90k",
    *(f"arch-{arm}" for arm in ARMS),
)

# The paper's protocols, by the names `instanovofm_evals.protocols.PROTOCOLS` gives them.
PROBES_RETRIEVAL = "paper-probes-retrieval"  # 200,000 spectra, 20,000 duplicate groups; probes on the tier's three splits
PEAK_LEVEL = "paper-peak-level"  # 10,000 spectra, batch 128
GEOMETRY = "paper-geometry"  # 20,000 spectra, six tasks
COSINE_HYPERSCORE = "paper-geometry"  # row 22 is the cosine-hyperscore task of the geometry run (same draw, same seed)
VALIDATION = "paper-validation"  # the trainer's validation pass on 250 batches


def env_path(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise SystemExit(f"{name} is not set: `source .envrc` first (see .envrc.example)")
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def checkpoint_path(model: str) -> Path:
    """The checkpoint file of `model`, from the `.envrc` variable named after its tag."""
    return env_path("CKPT_" + model.upper().replace("-", "_"))


def run(*, script: str, model: str, tier: str, split: str, protocol: str) -> Path:
    """Evaluate `model` on the `split` of `tier` under `protocol`; returns the directory the summaries were copied to."""
    return run_checkpoint(checkpoint=checkpoint_path(model), name=model, tier=tier, split=split, protocol=protocol,
                          dest=REPO / "docs" / "references" / "rerun" / script, script=script)


def run_checkpoint(*, checkpoint: Path, name: str, tier: str, split: str, protocol: str, dest: Path, script: str) -> Path:
    """Evaluate one checkpoint file under `protocol` and copy the suite's summaries to `dest` (stamped with `script`)."""
    os.environ.setdefault("INSTANOVO_FM_DATA_DIR", str(REPO / "data"))
    out = suite_run(
        "instanovo-fm",
        checkpoint=checkpoint,
        name=name,
        dataset=f"{tier}-{split}",
        protocol=protocol,
        results=env_path("RESULTS") / "package",
    )
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(out.glob("*.json"))
    for path in files:
        shutil.copy(path, dest / path.name)
    meta = json.loads((dest / "run.json").read_text())
    meta["script"] = script
    (dest / "run.json").write_text(json.dumps(meta, indent=1))
    print(f"{script}: {len(files)} files -> {dest}", flush=True)
    return dest
