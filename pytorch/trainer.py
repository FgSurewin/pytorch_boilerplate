from utils import ModelUtils
from tqdm.auto import tqdm
import torch
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        model,
        epochs,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        lr_scheduler=None,
        metrics=None,
        device=device,
    ):
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics
        self.device = device
        self.history = []

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            # Complie the model
            self.model.compile(
                device=self.device,
                optimizer=self.optimizer,
                criterion=self.criterion,
                metrics=self.metrics,
            )
            # Training step
            train_epoch_history = self.train(epoch_idx=epoch)
            # Validation step
            val_epoch_history = self.validate(epoch_idx=epoch)
            # Adjust learning rate if learning rate scheduler is provided
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # Get learning rate for current epoch
            learning_rate = self.optimizer.state_dict()["param_groups"][0]["lr"]
            # Collect epoch history and save to history
            epoch_history = self.generate_epoch_history(
                epoch, learning_rate, train_epoch_history, val_epoch_history
            )
            self.history.append(epoch_history)
            # Display epoch history
            self.display_epoch_history(epoch_history)

        return self.history

    def train(self, epoch_idx):
        # Enable training mode
        self.model.train()
        outputs = []
        n_total_steps = len(self.train_loader)
        tqdm_train_loader = tqdm(self.train_loader, leave=False)
        for batch_idx, batch in enumerate(tqdm_train_loader):
            train_batch_history = self.model.training_step(batch, batch_idx=batch_idx)
            # Set tqdm
            tqdm_train_loader.set_description_str(
                f"Train - Epoch:{epoch_idx}/{self.epochs} | Batch: {batch_idx + 1}/{n_total_steps}"
            )
            tqdm_train_loader.set_postfix(
                ModelUtils.history_tensor2item(train_batch_history)
            )
            outputs.append(train_batch_history)
        return self.model.training_epoch_end(outputs)

    @torch.no_grad()
    def validate(self, epoch_idx):
        # Enable evaluation mode
        self.model.eval()
        outputs = []
        n_total_steps = len(self.val_loader)
        tqdm_val_loader = tqdm(self.val_loader, leave=False)
        for batch_idx, batch in enumerate(tqdm_val_loader):
            val_batch_history = self.model.validation_step(batch, batch_idx=batch_idx)
            # Set tqdm
            tqdm_val_loader.set_description_str(
                f"Validate - Epoch:{epoch_idx}/{self.epochs} | Batch: {batch_idx + 1}/{n_total_steps}"
            )
            tqdm_val_loader.set_postfix(
                ModelUtils.history_tensor2item(val_batch_history)
            )
            outputs.append(val_batch_history)
        return self.model.validation_epoch_end(outputs)

    def generate_epoch_history(self, epoch, lr, train_epoch_history, val_epoch_history):
        epoch_history = {
            "Epoch": epoch,
            "lr": lr,
        }
        epoch_history.update(train_epoch_history)
        epoch_history.update(val_epoch_history)
        return epoch_history

    def display_epoch_history(self, epoch_history):
        result = ""
        for key, value in epoch_history.items():
            result += f"{key}: {value:.4f} | "
        # Remove the last pipe and white space
        result = result[:-3]
        print(result)

    def get_model_summary(self, input_size):
        return summary(self.model, input_size=input_size)
