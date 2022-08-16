import torch
import torch.nn as nn
from utils import ModelUtils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelBase(nn.Module):
    def compile(self, device, optimizer, criterion, metrics=None):
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        data, labels = batch
        # Convert data and labels to torch.Tensor
        data = data.to(self.device)
        labels = labels.to(self.device)
        # Forward pass
        output = self(data)
        # _, predictions = torch.max(output, 1) # We will use torchmetrics to calculate accuracy
        loss = self.criterion(output, labels)
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logs
        train_history = {"loss": loss.detach()}
        # Metrics | e.g. metrics = [{'acc':accuracy}, {'precision':precision}, {'recall':recall}]
        if self.metrics is not None:
            for metric in self.metrics:
                for key, func in metric.items():
                    train_history[key] = func(output, labels)
        return train_history

    def training_epoch_end(self, outputs):
        result = {}
        metric_names = outputs[0].keys()
        for metric_name in metric_names:
            if metric_name == "batch":
                continue
            metric_values = [x[metric_name] for x in outputs]
            # Calculate the mean of each metric
            result[metric_name] = torch.stack(metric_values).mean().item()
        return result

    def validation_step(
        self,
        batch,
        batch_idx,
        is_eval=False,
    ):
        data, labels = batch
        # Convert data and labels to torch.Tensor
        data = data.to(self.device)
        labels = labels.to(self.device)
        # Forward pass
        output = self(data)
        # _, predictions = torch.max(output, 1) # We will use torchmetrics to calculate accuracy
        loss = self.criterion(output, labels)

        # Logs
        prefix = "" if is_eval else "val_"
        val_history = {f"{prefix}loss": loss.detach()}
        # Metrics | e.g. metrics = [{'acc':accuracy}, {'precision':precision}, {'recall':recall}]
        if self.metrics is not None:
            for metric in self.metrics:
                for key, func in metric.items():
                    val_history[str(prefix + key)] = func(output, labels)
        return val_history

    def validation_epoch_end(self, outputs):
        result = {}
        metric_names = outputs[0].keys()
        for metric_name in metric_names:
            if metric_name == "batch":
                continue
            metric_values = [x[metric_name] for x in outputs]
            # Calculate the mean of each metric
            result[metric_name] = torch.stack(metric_values).mean().item()
        return result

    @torch.no_grad()
    def predict(self, data: torch.Tensor):
        self.eval()
        # Convert data to torch.Tensor
        data = data.to(self.device)
        # Forward pass
        output = self(data)
        _, predictions = torch.max(output, 1)
        return predictions

    @torch.no_grad()
    def predict_proba(self, data: torch.Tensor):
        self.eval()
        # Convert data to torch.Tensor
        data = data.to(self.device)
        # Forward pass
        output = self(data)
        return output

    def evaluate_with_tensor(self, data: torch.Tensor, labels: torch.Tensor):
        result = {}
        self.eval()
        with torch.no_grad():
            # Convert data to torch.Tensor
            data = data.to(self.device)
            labels = labels.to(self.device)
            # Forward pass
            output = self(data)
            loss = self.criterion(output, labels)
            result["loss"] = loss.detach()
            # Metrics | e.g. metrics = [{'acc':accuracy}, {'precision':precision}, {'recall':recall}]
            if self.metrics is not None:
                for metric in self.metrics:
                    for key, func in metric.items():
                        result[key] = func(output, labels)
            return result

    @torch.no_grad()
    def evaluate_with_dataloader(self, dataloader: DataLoader):
        # Enable evaluation mode
        self.eval()
        outputs = []
        n_total_steps = len(dataloader)
        tqdm_val_loader = tqdm(dataloader)
        for batch_idx, batch in enumerate(tqdm_val_loader):
            val_batch_history = self.validation_step(
                batch, batch_idx=batch_idx, is_eval=True
            )
            # Set tqdm
            tqdm_val_loader.set_description_str(
                f"Batch: {batch_idx + 1}/{n_total_steps}"
            )
            tqdm_val_loader.set_postfix(
                ModelUtils.history_tensor2item(val_batch_history)
            )
            outputs.append(val_batch_history)
        return self.validation_epoch_end(outputs)

    def evaluate(
        self,
        data: torch.Tensor = None,
        labels: torch.Tensor = None,
        dataloader: DataLoader = None,
    ):
        if data is not None and labels is not None:
            return self.evaluate_with_tensor(data, labels)
        elif dataloader is not None:
            return self.evaluate_with_dataloader(dataloader)
