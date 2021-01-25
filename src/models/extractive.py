import numpy as np
import torch
from torch import nn


def extractive_predict_max(model: nn.Module, embeddings) -> int:
    results = model.predict(torch.from_numpy(np.array(embeddings, dtype=np.float32)))
    max_idx = results.argmax()
    return max_idx.item()


class Perceptron(nn.Module):
    def __init__(self, d_in, n):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, n),
            nn.Sigmoid(),
            nn.Linear(n, n),
            nn.Sigmoid(),
            nn.Linear(n, n),
            nn.Sigmoid(),
            nn.Linear(n, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.layers(x)
        return res

    def predict(self, x):
        ret = self.forward(x)
        return self.sigmoid(ret)