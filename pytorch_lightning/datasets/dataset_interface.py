import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

data_path = os.path.join(os.path.abspath("."), "data")


class DatasetDistributor(pl.LightningDataModule):
    def __init__(self, data_dir=data_path, batch_size=128, num_workers=8) -> None:
        # Define required parameters
        super().__init__()
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self) -> None:
        # Define setps that should be done only once
        # On only one GPU
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Define steps that should be done on every GPU
        # like splitting the dataset, etc.
        if stage == "fit" or stage is None:
            mnist = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        # Define the train dataloader
        mnist_train = DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return mnist_train

    def val_dataloader(self) -> DataLoader:
        # Define the validation dataloader
        mnist_val = DataLoader(
            self.mnist_val,
            batch_size=10 * self.batch_size,
            num_workers=self.num_workers,
        )
        return mnist_val

    def test_dataloader(self) -> DataLoader:
        # Define the test dataloader
        mnist_test = DataLoader(
            self.mnist_test,
            batch_size=10 * self.batch_size,
            num_workers=self.num_workers,
        )
        return mnist_test


if __name__ == "__main__":
    mnist = DatasetDistributor()
    mnist.prepare_data()
    mnist.setup()
    # grab samples to log predictions on
    samples = next(iter(mnist.val_dataloader()))
    print((samples[0].shape, samples[1].shape))
