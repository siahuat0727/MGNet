import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['VanillaCNN']

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x
