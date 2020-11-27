import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn

class Baseline(nn.Module):
    """
    Baseline networkdata_loader
    """

    def __init__(self, input_channels=2, n_classes=2, dropout=True):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.2)

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #         self.layer1 = nn.Sequential(nn.Conv2d(input_channels, 64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU())

        #         self.layer2 = nn.Sequential(nn.Conv2d(64, 64,kernel_size=3,stride=2,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        #         self.layer3 = nn.Sequential(nn.Conv2d(64, 16,kernel_size=3,stride=2,padding=1),nn.BatchNorm2d(16),nn.ReLU())

        self.maxpooling = nn.MaxPool2d(2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, n_classes)

    #         self.mask4 = nn.Linear(64, n_classes)
    #         self.weight_init()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        #         x=self.maxpooling(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #         x=self.maxpooling(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        #         x=self.maxpooling(x)
        if self.use_dropout:
            x = self.dropout(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
