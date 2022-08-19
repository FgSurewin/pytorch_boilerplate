from dataset import test_dataset, train_dataset, val_dataset, get_loader, classes

# from utils import ImageUtils
import random
import numpy as np
import torch
from utils import ModelUtils
from cnn_model import ConvNet
from trainer import Trainer
from torchmetrics import Accuracy, Precision, Recall


# print(sample)
# print(sample.shape, classes[label_id])
# ImageUtils.show_image(sample, title=classes[label_id])


def main():
    # Set seed for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Global parameters
    num_epochs = 2
    batch_size = 50
    learning_rate = 0.001
    input_channels = 3  # RGB image - (3, 32, 32)
    num_classes = 10  # 10 classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data loaders
    train_loader = get_loader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_loader = get_loader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Define the model
    model = ConvNet(input_channels=input_channels, num_classes=num_classes).to(device)

    # Define the loss criterion, optimizer and metrics
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metrics = [
        {"accuracy": Accuracy().to(device)},
        # {"precision": Precision(num_classes=num_classes).to(device)},
        # {"recall": Recall(num_classes=num_classes).to(device)},
    ]

    # Train the model
    trainer = Trainer(
        model,
        epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        metrics=metrics,
        device=device,
    )
    print("Model acceleration:", ModelUtils.get_model_acceleration(model))
    print(trainer.get_model_summary((3, 32, 32)))
    # history = trainer.fit()
    # print(len(history))


if __name__ == "__main__":
    main()
    print("Done")
