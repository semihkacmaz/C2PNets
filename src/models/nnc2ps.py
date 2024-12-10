import torch
import torch.nn as nn
from pathlib import Path
from .base_model import BaseEOSModel

class NNC2PS(BaseEOSModel):
    def __init__(self, config):
        super().__init__()

        hidden_dims = config["model"]["hidden_dims"]
        input_dim = config["model"]["input_dim"]
        output_dim = config["model"]["output_dim"]

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def save(self, path):
        scripted_model = torch.jit.script(self)
        torch.jit.save(scripted_model, path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
