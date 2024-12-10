from abc import ABC, abstractmethod
import torch.nn as nn

class BaseEOSModel(nn.Module, ABC):
    """Base class for EOS neural network models."""

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model."""
        pass

    @abstractmethod
    def save(self, path):
        """Save model weights."""
        pass

    @abstractmethod
    def load(self, path):
        """Load model weights."""
        pass
