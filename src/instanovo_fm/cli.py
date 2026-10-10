from typing import List, Optional

import typer
from typing_extensions import Annotated

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

from instanovo_fm.utils.hydra_config import compose_fm_config

logger = ColorLog(console, __name__).logger

cli = typer.Typer(rich_markup_mode="rich", pretty_exceptions_enable=False)


@cli.command("train")
def foundational_train(
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config-path",
            "-cp",
            help="Relative path to config directory.",
        ),
    ] = None,
    config_name: Annotated[
        Optional[str],
        typer.Option(
            "--config-name",
            "-cn",
            help="The name of the config (usually the file name without the .yaml extension).",
        ),
    ] = None,
    overrides: Optional[List[str]] = typer.Argument(None, hidden=True),
) -> None:
    """Train the InstaNovo Foundation Model."""
    logger.info("Initializing InstaNovo Foundation Model training.")

    if config_name is None:
        config_name = "foundational"

    config = compose_fm_config(
        config_name=config_name,
        overrides=overrides,
        config_dir=config_path,
    )

    logger.info("Starting InstaNovo Foundation Model training.")
    from instanovo_fm.trainer.train import FoundationalTrainer

    trainer = FoundationalTrainer(config)
    trainer.train()

    # Save MLflow run ID alongside checkpoint for post-training eval to pick up
    import os

    if trainer.tracker is not None and hasattr(trainer.tracker, "run_id"):
        checkpoint_dir = config.model.get("model_save_folder_path", "./checkpoints")
        run_id_path = os.path.join(checkpoint_dir, "mlflow_run_id.txt")
        with open(run_id_path, "w") as f:
            f.write(trainer.tracker.run_id)
        logger.info(f"Saved MLflow run ID to {run_id_path}")

    trainer.run_post_training_evaluation()


@cli.command("evaluate")
def foundational_evaluate(
    checkpoint: Annotated[str, typer.Option("--checkpoint", help="Checkpoint to evaluate (.ckpt).")],
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="A corpus split (lcfm-test, mcfm-test, lcfm-valid, mcfm-valid) for the paper-* protocols, or an exported dataset."),
    ],
    protocol: Annotated[
        str,
        typer.Option(
            "--protocol",
            help="paper-probes-retrieval, paper-peak-level, paper-geometry, paper-validation on a corpus split; probes, retrieval, geometry, "
            "peak_level, unlabelled, validation on an exported dataset.",
        ),
    ],
    name: Annotated[str, typer.Option("--name", help="Results directory name: <results>/<dataset>/<name>/<protocol>.")] = "model",
    results: Annotated[Optional[str], typer.Option("--results", help="Results root; default: the suite's own results/ directory.")] = None,
) -> None:
    """Evaluate a Foundation Model checkpoint with the evaluation suite.

    The suite is the ``instanovofm_evals`` package (a dependency); this command is its
    ``run("instanovo-fm", ...)``, the same as ``instanovofm-evals run --adapter instanovo-fm``.
    The protocol fixes which spectra, caps, seed, batch size and tasks; see the suite's README.
    """
    from instanovofm_evals import run

    logger.info(f"Evaluating {checkpoint} on {dataset} under {protocol} with instanovofm_evals.")
    run("instanovo-fm", checkpoint=checkpoint, name=name, dataset=dataset, protocol=protocol, **({"results": results} if results else {}))


from instanovo_fm.downstream.de_novo_sequencing.cli import cli as _denovo_cli

cli.add_typer(
    _denovo_cli,
    name="denovo",
    help="Downstream de novo sequencing: the FM encoder plus an InstaNovo decoder.",
)


if __name__ == "__main__":
    cli()
