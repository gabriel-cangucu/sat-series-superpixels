import torch
import lightning as L
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf
from typing import Any
from pathlib import Path

from models.deep_superpixels_module import DeepSuperpixelsModule
from data.ibge_labeled import IBGE_Labeled_Module
from utils.train_utils import load_from_checkpoint


DATASETS_REGISTRY = {
    "ibge_labeled": IBGE_Labeled_Module
}


def train_and_eval(config: Any) -> None:
    """
    Train a deep model for superpixel segmentation.
    """
    # Setting up model
    model_name = DeepSuperpixelsModule
    model = model_name(config)
    model = load_from_checkpoint(config, model, model_name=model_name)
    
    # Setting up data
    assert config.dataset.name in DATASETS_REGISTRY.keys(), f"Unsupported dataset type: {config.dataset.name}. \
                                                              Choose from: {list(DATASETS_REGISTRY.keys())}."
    data_module = DATASETS_REGISTRY[config.dataset.name](config)

    logger = WandbLogger(
        save_dir=config.checkpoint.save_dir,
        name=config.checkpoint.run_name,
        id=config.checkpoint.run_name
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.checkpoint.save_dir) / "wandb" / f"ckpt_{config.checkpoint.run_name}",
        filename=f"{config.checkpoint.run_name}-{{epoch:02d}}",
        monitor="val/loss" if config.dataset.validate else "train/loss",
        mode="min",
        save_top_k=2,
    )
    
    trainer = L.Trainer(
        max_epochs=config.solver.max_epochs,
        callbacks=[checkpoint_callback] if config.checkpoint.save_checkpoint else None,
        fast_dev_run=config.solver.dev_run,
        overfit_batches=1 if config.solver.overfit_batches else 0.,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
    )
    
    ckpt_path = config.checkpoint.ckpt_path if config.checkpoint.ckpt_path else None
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    if not config.solver.dev_run:
        trainer.test(model, ckpt_path="best")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a deep model for superpixel segmentation of satellite series.")
    parser.add_argument(
        "--config",
        help="Configurarion (.yaml) file to use",
        type=str,
        required=True
    )
    
    args = parser.parse_args()    
    config = OmegaConf.load(args.config)
    
    train_and_eval(config)
    print("Training completed successfully.")
