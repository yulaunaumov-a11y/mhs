import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp

import segmentation_models_pytorch as smp
from utils.utils import read_config

config = read_config("./config.yml")

THRESHOLD = config.visualization.THRESHOLD

class ComplexLoss(torch.nn.Module):
    def __init__(self, losses_list: tp.List[torch.nn.Module], weights: tp.Optional[tp.List[float]]=None) -> torch.Tensor:
        """
        Initialize ComplexLoss class

        param:
            losses_list: tp.List[torch.nn.Module]
                List of loss functions
            weights: tp.Optional[tp.List[float]]
                List of weights
        """
        super(ComplexLoss, self).__init__()
        self.losses_list = losses_list
        self.weights = weights

    def __str__(self):
        return f"ComplexLoss(losses_list={self.losses_list},\n weights={self.weights})"

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor)-> torch.Tensor:
        """
        Forward pass for weighted loss calculations

        :param inputs: Input (predicted) tensor
        :param targets: Target tensor

        :return: Loss value
        """
        if self.weights is None:
            self.weights = [1] * len(self.losses_list)

        loss = sum(w * loss_fn(inputs, targets) for w, loss_fn in zip(self.weights, self.losses_list))
        return loss
    