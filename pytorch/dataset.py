import os
import torch
import torchvision
import torchvision.transforms as transforms

data_path = os.path.join(os.path.abspath("."), "data")

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(
    root=data_path, train=True, download=True, transform=transform
)

full_test_dataset = torchvision.datasets.CIFAR10(
    root=data_path, train=False, download=True, transform=transform
)

num_full_test_batches = len(full_test_dataset)
num_val_dataset = int(num_full_test_batches * 0.5)
num_test_dataset = num_full_test_batches - num_val_dataset
val_dataset, test_dataset = torch.utils.data.random_split(
    full_test_dataset, [num_val_dataset, num_test_dataset]
)


def get_loader(dataset, batch_size, shuffle=True, num_workers=2):
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return data_loader


classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

if __name__ == "__main__":
    print(len(train_dataset), len(val_dataset), len(test_dataset))
