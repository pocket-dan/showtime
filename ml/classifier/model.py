from collections import OrderedDict
from typing import Dict

import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        n_vertex = 7
        n_hidden = 128
        n_hidden = 128
        n_classes = 8

        self.fc1 = nn.Linear(n_vertex * 2, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)

        self.classes = [
            "hands-on-head",
            "victory",
            "cheer-up",
            "go-next",
            "go-back",
            "ultraman",
            "others",
        ]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def infer(self, x):
        y = self.forward(x)
        return y.argmax(dim=1, keepdim=True)

    def decode_class(self, n):
        return self.classes[n]
