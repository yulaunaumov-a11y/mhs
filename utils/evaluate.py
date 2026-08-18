import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from typing import List

def evaluate(model: torch.nn.Module,
            dataloader: DataLoader,
            criterion: torch.nn.Module,
            device: torch.device,
) -> float:
    """
    Evaluate the model.

    param:
        model: torch.nn.Module
            CNN model.
        dataloader: DataLoader
            Training dataloader.
        criterion: torch.nn.Module
            Loss functions.
        device: torch.device
            Device for learning.

    return: 
        Validation loss
    """
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            y_pred = model(images)

            loss = criterion(y_pred, labels)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def draw_loss_plot(losses: List, graph_name: str) -> None:
    """
    Function to draw a loss plot from a list of losses.
    
    param:
        losses (list): List of loss values (floats or integers) to plot.
    """
    if not losses:
        print("The list of losses is empty.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss', color='blue', linewidth=2)
    plt.title(graph_name, fontsize=16)
    plt.xlabel('Iteration/Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()