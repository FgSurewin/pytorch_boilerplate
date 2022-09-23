import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy


class ModelDistributor(pl.LightningModule):
    def __init__(
        self, model, metrics=[Accuracy(num_classes=10)], lr=0.001, **kargs
    ) -> None:
        super().__init__()
        self.model = model
        self.metrics = metrics
        self.save_hyperparameters(ignore=["model", "metrics"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        y_hat = self(data)
        loss = F.cross_entropy(y_hat, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric in self.metrics:
            self.log(
                f"train/{metric._get_name()}",
                metric(y_hat, labels),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        y_hat = self(data)
        loss = F.cross_entropy(y_hat, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric in self.metrics:
            self.log(
                f"val/{metric._get_name()}",
                metric(y_hat, labels),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        y_hat = self(data)
        loss = F.cross_entropy(y_hat, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric in self.metrics:
            self.log(
                f"test/{metric._get_name()}",
                metric(y_hat, labels),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
