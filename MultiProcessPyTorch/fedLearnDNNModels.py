'''
提供 DNN Models
'''

import torch
from torch import nn
import torch.nn.functional as func


class LinearModel(nn.Module):

    def __init__(self) -> None:
        super(LinearModel, self).__init__()
        self.flatten = nn.Flatten()  # torch == 1.0.0  no this function nn.Flatten()
        # self.flatten = torch
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class CNNModel(nn.Module):

    def __init__(self) -> None:
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.dense1 = nn.Linear(128 * 7 * 7, 4096)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 7 * 7)
        logits = self.dense3(
            self.drop2(
                func.relu(
                    self.dense2(
                        self.drop1(
                            func.relu(self.dense1(x)))))))
        return logits


if __name__ == "__main__":
    pass
