import torch
import torch.nn as nn
# from common_net import ConvBlock

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        has_maxpool=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.has_maxpool = has_maxpool
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.has_maxpool:
            x = self.pool(x)
        return x


class ConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBlock(1, 32, has_maxpool=False)
        self.conv2 = ConvBlock(32, 64, has_maxpool=True)
        self.fc1 = nn.Linear(14 * 14 * 64, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 14 * 14 * 64)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    test = torch.rand(1, 1, 28, 28)
    model = ConvModel()
    print(model(test).shape)
