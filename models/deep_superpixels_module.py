import torch
import lightning as L
import torchmetrics
from typing import Any
from einops import rearrange

from utils.metrics import AchievableSegmentationAccuracy, BoundaryPrecision, BoundaryRecall, UndersegmentationError
from models.ssn import SSN
from models.spixel_fcn import SpixelNet
from models.cdspixel import CDSpixelNet, CLUB
from models.unet import Unet


MODEL_REGISTRY = {
    "ssn": SSN, 
    "spixel_net": SpixelNet,
    "cdspixel_net": (CDSpixelNet, CLUB),
    "unet": Unet
}

SOLVER_REGISTRY = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW
}


class DeepSuperpixelsModule(L.LightningModule):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.automatic_optimization = False
        
        self.config = config
        self.save_hyperparameters()

        assert config.model.name in MODEL_REGISTRY.keys(), f"Invalid model name {config.model.name}. \
                                                             Choose from {list(MODEL_REGISTRY.keys())}"
        if isinstance(MODEL_REGISTRY[config.model.name], (list, tuple)):
            model, regularizer = MODEL_REGISTRY[config.model.name]
        else:
            model = MODEL_REGISTRY[config.model.name]
            regularizer = None

        self.model = model(
            img_size=config.model.crop_size,
            num_channels=config.model.num_channels,
            num_frames=config.model.num_timestamps
        )
        self.regularizer = regularizer(x_dim=32, y_dim=32) if regularizer else None

        if isinstance(self.model, Unet):
            metrics = torchmetrics.MetricCollection({
                "iou": torchmetrics.classification.JaccardIndex(task="multiclass", num_classes=3)
            })
        else:
            metrics = torchmetrics.MetricCollection({
                "asa": AchievableSegmentationAccuracy(),
                "bp": BoundaryPrecision(radius=5),
                "br": BoundaryRecall(radius=5),
                "ue": UndersegmentationError()
            })
        
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
    
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["data"]
        y = batch["target"]

        assert len(x.shape) == 5, "Images must be (b c t h w)"
        # Collapsing the channels and timestamps dims since models are 2D
        x = rearrange(x, "b c t h w -> b (c t) h w")

        if self.regularizer:
            return self.model(x, y, self.regularizer)
        else:
            return self.model(x, y)
    
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.regularizer:
            self.regularizer.eval()

        outputs = self(batch)
        optimizers = self.optimizers()

        if isinstance(optimizers, (list, tuple)):
            main_optimizer = optimizers[0]
            reg_optimizer = optimizers[1]
        else:
            main_optimizer = optimizers
            reg_optimizer = None

        # Main model update
        main_loss = outputs["loss"]

        self.toggle_optimizer(main_optimizer)
        main_optimizer.zero_grad()
        self.manual_backward(main_loss)
        main_optimizer.step()
        self.untoggle_optimizer(main_optimizer)

        # Regularizer update
        if self.regularizer and reg_optimizer:
            self.regularizer.train()

            assert "reg_loss" in outputs.keys()
            reg_loss = outputs["reg_loss"]

            self.toggle_optimizer(reg_optimizer)
            reg_optimizer.zero_grad()
            self.manual_backward(reg_loss)
            reg_optimizer.step()
            self.untoggle_optimizer(reg_optimizer)

        self.train_metrics.update(outputs["preds"], batch["target"])
        self.log("train/loss", main_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("lr", main_optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True, prog_bar=True)
        
        return main_loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch)
        
        preds = outputs["preds"]
        loss = outputs["loss"]

        self.val_metrics.update(preds, batch["target"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()

        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        
        val_loss = self.trainer.callback_metrics["val/loss"]

        for sch in schedulers:
            # ReduceLROnPlateau needs a metric
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_loss is not None:
                    sch.step(val_loss)
            else:
                # Standard schedulers
                sch.step()

        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()
    
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch)

        preds = outputs["preds"]
        loss = outputs["loss"]

        self.test_metrics.update(preds, batch["target"])
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()
    
    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self(batch)
    
    def configure_optimizers(self) -> dict[str, Any] | tuple:
        assert self.config.solver.name in SOLVER_REGISTRY.keys(), f"Solver {self.config.solver.name} is not supported. \
                                                                    Choose from {list(SOLVER_REGISTRY.keys())}."
        
        main_optimizer = SOLVER_REGISTRY[self.config.solver.name](
            params=self.model.parameters(),
            lr=self.config.solver.learning_rate,
            weight_decay=self.config.solver.weight_decay if self.config.solver.weight_decay else 0.0
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            main_optimizer, mode="min", factor=0.5, patience=10
        )

        if self.regularizer:
            reg_optimizer = torch.optim.Adam(params=self.regularizer.parameters(), lr=5e-4)
            
            return (
                [main_optimizer, reg_optimizer],
                [{"scheduler": scheduler, "interval": "epoch"}]
            )
        
        return {
            "optimizer": main_optimizer,
            "lr_scheduler": { "scheduler": scheduler, "interval": "epoch"}
        }

