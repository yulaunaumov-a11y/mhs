import torch
import addict
import logging
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict

from utils.logger import Logger
from utils.augmentation import get_augmentations
from utils.utils import object_from_dict
from utils.evaluate import evaluate, draw_loss_plot
from utils.metrics import calculate_metrics
from utils.custom_loss import ComplexLoss
from utils.boundary_loss import BoundaryLoss
from utils.data import build_dataloaders

def train(config: Dict, model: torch.nn.Module) -> None:
    """
    Perform training procedure.

    param:
        config: addict.Dict
            Configuration for training the model.
        model: torch.nn.Module
            CNN model.
    return:
        None
    """
    train_loss_all = []
    val_loss_all = []
    metrics_dict_val={
                        "Jaccard": [],
                        "F1": [],
                        "Recall": [],
                        "Precision": [],
                        "Acc": [],
                    }
    print("Declare a device.")
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Device use: {device}")
    print("Create a configuration for augmentations.")
    augmentations = get_augmentations(config)
    print("Build a dataloaders.")
    train_loader, test_loader, val_loader, contour_counts = build_dataloaders(config, augmentations)
    print("Initializing the model.")
    model = model.to(device)
    print("Initializing criterion, optimizer.")
    config_criterion = object_from_dict(config.criterion)
    criterion = config_criterion
    # criterion = ComplexLoss([config_criterion,
    #                          smp.losses.JaccardLoss(mode='multiclass')], [0.5, 0.5])
    # criterion = ComplexLoss([config_criterion,
    #                          smp.losses.DiceLoss(mode='multiclass'),
    #                          smp.losses.FocalLoss(mode='multiclass')], [0.25, 0.25, 0.5])
    optimizer = object_from_dict(config.optimizer, params = model.parameters())
    # scheduler = object_from_dict(config.scheduler, optimizer=optimizer)
    metrics_dict_last = {
                "Jaccard": 0,
                "F1": 0,
                "Recall": 0,
                "Precision": 0,
                "Acc": 0,
            }
    print("Initializing Clearml logging.")
    logger = Logger(config)
    print("Start the learning process.")
    logger._distribution_classes(contour_counts)
    for epoch in tqdm(range(config.training.epochs), total=config.training.epochs, desc="Training"):

        if (config.training.epochs // 5) * 2 == epoch:
            criterion = ComplexLoss([config_criterion,
                                     smp.losses.JaccardLoss(mode='multiclass')], [0.5, 0.5])

        if (config.training.epochs // 5) * 4 == epoch:
            criterion = ComplexLoss([config_criterion,
                                     smp.losses.JaccardLoss(mode='multiclass'),
                                     smp.losses.FocalLoss(mode='multiclass')], [0.1, 0.4, 0.5])

        val_loss = evaluate(model, val_loader, criterion, device)
        if epoch == 0:
            train_loss = val_loss

        logger.log_loss(epoch=epoch, train_loss=train_loss, val_loss=val_loss)

        # logger.model_predict_for_log(model, train_loader, epoch, device, title = f"Train image {epoch}")
        # logger.log_learning_rate(optimizer.param_groups[0]['lr'], epoch)

        # metrics_dict, per_class_metrics, contour_statistic = calculate_metrics(model, val_loader, device)
        metrics_dict, per_class_metrics = calculate_metrics(model, val_loader, device)
        for metric in metrics_dict_val:
            metrics_dict_val[metric].append(metrics_dict[metric])

        if metrics_dict["F1"] > metrics_dict_last["F1"]:
            logger.model_predict_for_log(model, val_loader, epoch, device, title = f"Validation image {epoch}")
            logger.log_model_weights(model_weights=model, epoch=epoch, name = "best_model")
            # logger.log_confused_matrix(model, val_loader, device, epoch, config.task.num_classes)
            metrics_dict_last = metrics_dict

        logger.log_train_and_val_metrics(epoch=epoch, metrics_dict=metrics_dict, title="All metric")

        for class_name in per_class_metrics:
            logger.log_train_and_val_metrics(epoch=epoch, metrics_dict=per_class_metrics[class_name], title=f"{class_name} metric")
            # logger.log_contour_metrics(epoch=epoch, metrics_dict=contour_statistic[class_name], title=f"{class_name} contour metric")

        logger.log_model_weights(model_weights=model, epoch=epoch, name = "last")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # scheduler.step()
    test_metric, per_class_metrics = calculate_metrics(model, test_loader, device)
    logger.log_test_metrics(test_metric)
    logger.complete_logging()

    return model, test_metric


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Perform a single epoch training.
    
    param:
        model: torch.nn.Module
            CNN model.
        dataloader: DataLoader
            Training dataloader.
        optimizer: torch.optim.Optimizer
            Optimization algorithm.
        criterion: torch.nn.Module
            Loss functions.
        device: torch.device
            Device for learning.

    return:
        Loss value.
    """
    model.train()
    epoch_loss = 0
    accumulation_steps = 10
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Train one epoch", position=0, leave=True)):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        y_pred = model(images)
        loss = criterion(y_pred, labels) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)