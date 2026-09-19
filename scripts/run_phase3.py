"""
CLI entry point for Phase 3: PI-GAN Training.

Usage:
    python scripts/run_phase3.py --config configs/base_config.yaml
"""

import sys
import click
from pathlib import Path

# Ensure project root is in sys.path so 'from src...' imports work
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logger
from src.utils.geo_utils import load_config


@click.command()
@click.option("--config", default="configs/base_config.yaml")
@click.option("--epochs", default=500, type=int, help="Total training epochs.")
@click.option("--batch-size", default=8, type=int)
@click.option("--resume", default=None, help="Checkpoint tag to resume from.")
def main(config: str, epochs: int, batch_size: int, resume: str | None):
    """Run Phase 3: PI-GAN Training."""
    logger = setup_logger("pigan.phase3")
    cfg = load_config(config)

    from src.phase3_pigan.dataset import create_dataloaders
    from src.phase3_pigan.trainer import PIGANTrainer

    logger.info("=" * 60)
    logger.info("Phase 3: PI-GAN Super-Resolution Training")
    logger.info("=" * 60)

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, batch_size=batch_size,
    )

    # Initialise trainer
    trainer = PIGANTrainer(cfg)

    if resume:
        trainer.load_checkpoint(resume)
        logger.info("Resumed from checkpoint: %s", resume)

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        total_epochs=epochs,
    )

    # Plot training curves
    from src.utils.visualization import plot_training_curves
    plot_training_curves(
        history,
        title="PI-GAN Training Loss",
        save_path=Path(cfg["paths"]["outputs"]) / "training_curves.png",
    )

    logger.info("=" * 60)
    logger.info("Phase 3 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
