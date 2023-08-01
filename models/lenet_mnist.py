"""Defines the neural network, loss function and metrics"""

# import packages
import torch.nn as nn
from torchmetrics import Accuracy
import torch.nn.functional as F


class LeNet5(nn.Module): 

    """
    This class defines the LeNet5 model with MNIST dataset
    """

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2), # 28x28x1 -> 32*32 --> 28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 14*14

            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 5*5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(self.feature(x))
    

    """
# LeNet Model definition
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    """