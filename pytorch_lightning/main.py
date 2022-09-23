import wandb
import torch
from datasets import DatasetDistributor
from pytorch_lightning import Trainer
from models.model_interface import ModelDistributor
from models.conv_model import ConvModel
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torchmetrics
import os


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)

        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(val_imgs, preds, self.val_labels)
                ],
                "global_step": trainer.global_step,
            }
        )


def main():
    #  Define dataset
    mnist = DatasetDistributor()
    # mnist.prepare_data()
    mnist.setup()

    # grab samples to log predictions on
    samples = next(iter(mnist.val_dataloader()))

    # print((samples[0].shape, samples[1].shape))

    # # define model
    conv_model = ConvModel()
    model = ModelDistributor(conv_model)

    # # train model
    wandb_logger = WandbLogger(project="pytorch-lightning-learning", name="mnist-conv2")
    trainer = trainer = Trainer(
        default_root_dir="./checkpoints/test2",
        logger=wandb_logger,
        max_epochs=3,
        deterministic=True,
        callbacks=[ImagePredictionLogger(samples)],
    )
    trainer.fit(model, mnist)

    # # test model
    trainer.test(model, mnist)
    wandb.finish()


if __name__ == "__main__":
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    main()
