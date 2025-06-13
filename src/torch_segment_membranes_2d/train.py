from pathlib import Path

import typer
import lightning as L

from torch_segment_membranes_2d._cli import OPTION_PROMPT_KWARGS

from torch_segment_membranes_2d.dataset import MembraneDataModule
from torch_segment_membranes_2d.model import ResidualUNet18


def train_membrane_segmentation_model(
    dataset_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    output_directory: Path = typer.Option("training", **OPTION_PROMPT_KWARGS),
    batch_size: int = 1,
    gradient_steps: int = 2000,
    learning_rate: float = 1e-5,
) -> None:
    """Train a semantic segmentation model for membrane detection."""
    model = ResidualUNet18(batch_size=batch_size, learning_rate=learning_rate)
    data_module = MembraneDataModule(dataset_directory)
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        default_root_dir=output_directory,
        max_steps=gradient_steps,
        log_every_n_steps=10,
        check_val_every_n_epoch=None,
        val_check_interval=200,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
    )
    trainer.fit(model, datamodule=data_module)
    return None