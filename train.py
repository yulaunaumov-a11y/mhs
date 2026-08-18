import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch
import datetime
import os

from clearml import Task, Logger
from utils.Image_processing import split_image_into_patch, get_rgb_image
from utils.augmentation import get_augmentations
from utils.dataset_class import DatasetClass
from utils.utils import read_config, get_image_in_folder
from utils.training import train
# from utils.metrics import denormalize, visualization_testing_data
from utils.evaluate import evaluate, draw_loss_plot



if __name__ == "__main__":

    config = read_config("./config.yml")
    
    model = smp.FPN(
    encoder_name=config.model.encoder_name,
    encoder_weights=config.model.encoder_weights,
    in_channels=config.model.in_channels,
    classes=config.model.classes,
    activation=config.model.activation,
)
    model, test_metric = train(config, model)