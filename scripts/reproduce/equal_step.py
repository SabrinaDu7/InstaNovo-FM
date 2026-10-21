"""Compare an arm with the baseline at the same training step through one of the suite's paper protocols.

Both runs keep a checkpoint every 10,000 steps (`keep_model_every_interval=True`), so a leading arm can be judged on
retrieval and the probes before it finishes, against the baseline's checkpoint of the same step rather than its final one.

    source .envrc && python scripts/reproduce/equal_step.py --arm ionladder --step 50001            # mcfm-test, paper-probes-retrieval
    sbatch --time=02:59:00 scripts/reproduce/submit.sh scripts/reproduce/equal_step.py --arm ionladder --step 50001 --side arm
    sbatch --time=02:59:00 scripts/reproduce/submit.sh scripts/reproduce/equal_step.py --arm ionladder --step 50001 --side baseline

The checkpoints are `$RUNS/arch-<arm>/checkpoints/*_step_<step>.ckpt` and `$RUNS/train-40M-mcfm-90k-b2048/checkpoints/*_step_<step>.ckpt`;
the summaries land in `docs/arch_study/equal_step/<arm>_step<step>/<side>/`. `--side` runs one side (one SLURM job each);
without it both sides run in sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ARMS, PROBES_RETRIEVAL, REPO, env_path, run_checkpoint

BASELINE_RUN = "train-40M-mcfm-90k-b2048"


def checkpoint_at(run: str, step: int) -> Path:
    """The single checkpoint file of training run `run` at `step`."""
    matches = sorted((env_path("RUNS") / run / "checkpoints").glob(f"*_step_{step}.ckpt"))
    if len(matches) != 1:
        raise SystemExit(f"{run}: expected one checkpoint at step {step}, found {matches}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--step", type=int, required=True, help="training step of the checkpoints, e.g. 50001")
    ap.add_argument("--tier", default="mcfm")
    ap.add_argument("--split", default="test")
    ap.add_argument("--protocol", default=PROBES_RETRIEVAL)
    ap.add_argument("--side", choices=("arm", "baseline"), help="run one side only (default: both, arm first)")
    args = ap.parse_args()
    sides = {"arm": (f"arch-{args.arm}", f"arch-{args.arm}-step{args.step}"), "baseline": (BASELINE_RUN, f"40M-b2048-step{args.step}")}
    for side in ([args.side] if args.side else ["arm", "baseline"]):
        run, name = sides[side]
        script = f"equal_step_{args.arm}_{args.step}_{side}"
        run_checkpoint(checkpoint=checkpoint_at(run, args.step), name=name, tier=args.tier, split=args.split, protocol=args.protocol,
                       dest=REPO / "docs" / "arch_study" / "equal_step" / f"{args.arm}_step{args.step}" / side, script=script)


if __name__ == "__main__":
    main()
