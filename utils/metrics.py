import torch
import cv2
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import read_config
from typing import Dict
from collections import defaultdict
from skimage import measure
from skimage.draw import polygon
from scipy.spatial import ConvexHull, distance
from shapely.geometry import Polygon, Point

config = read_config("./config.yml")

THRESHOLD = config.visualization.THRESHOLD

def denormalize(tensor: torch.Tensor, mean: list =[0.485, 0.456, 0.406], std: list =[0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    Denormalizes the tensor transformed by albumentations.Normalize.
    
    param:
        tensor: torch.Tensor
            The normalized tensor is [B, C, H, W]
        mean: list
            Average values used for normalization
        std: list
            Standard deviations used for normalization
    
    return: 
        Denormalized tensor in the range [0, 1]
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    
    denorm_tensor = tensor * std + mean
    
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    
    return denorm_tensor

def calculate_metrics(
    model: torch.nn.Module, 
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Calculate the metrics of the model quality.
    
    param:
        model: torch.nn.Module
            The trained model.
        dataloader: DataLoader
            Training dataloader.
        device: torch.device
            Device for learning.
    return:
        dict[str, float]
            Model quality metrics.
    """
    model = model.to(device)
    model.eval()
    all_tp, all_fp, all_fn, all_tn = [], [], [], []

    contour_counts_labels = {config.task.labels[class_id]: class_id for class_id in range(len(config.task.labels))}
    contour_counts_y_pred = {config.task.labels[class_id]: class_id for class_id in range(len(config.task.labels))}

    per_contour_statistic = {
                            "Relation": 0,
                            "Relation positive": 0,
                            "Relation iou50": 0,
                            "Relation iou75": 0,
                            "Relation iou95": 0
                    }

    contour_statistic = {config.task.labels[class_id]: per_contour_statistic for class_id in range(len(config.task.labels))}

    with torch.no_grad():
        for images, labels in dataloader:

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            # print(f"labels.shape = {labels.shape}")
            # print(f"unique(labels) = {torch.unique(labels)}")
            y_pred = model(images)

            # ccl, ccp = count_contours_per_class(labels, y_pred)

            # for key, value in ccl.items():
            #     contour_counts_labels[key] = contour_counts_labels.get(key, 0) + value

            # for key, value in ccp.items():
            #     contour_counts_y_pred[key] = contour_counts_y_pred.get(key, 0) + value

            # contour_statistic_iter = get_per_class_count_stat(contour_counts_labels, contour_counts_y_pred, labels, y_pred)

            # for key, value in contour_statistic.items():
            #     for key_in, value_in in contour_statistic[key].items():
            #         contour_statistic[key][key_in] = contour_statistic_iter[key][key_in] + value_in

            y_pred_processed = torch.argmax(y_pred, dim=1)
            conf_mtx = smp.metrics.get_stats(y_pred_processed, labels.int(), mode=config.task.mode, num_classes=config.task.num_classes)

            tp, fp, fn, tn = conf_mtx

            all_tp.append(tp)
            all_fp.append(fp)
            all_fn.append(fn)
            all_tn.append(tn)

    tp_total = torch.cat(all_tp).sum(dim=0)
    fp_total = torch.cat(all_fp).sum(dim=0)
    fn_total = torch.cat(all_fn).sum(dim=0)
    tn_total = torch.cat(all_tn).sum(dim=0)
    
    iou_score = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total, reduction="macro")
    f1_score = smp.metrics.f1_score(tp_total, fp_total, fn_total, tn_total, reduction="macro")
    accuracy = smp.metrics.accuracy(tp_total, fp_total, fn_total, tn_total, reduction="macro")
    recall = smp.metrics.recall(tp_total, fp_total, fn_total, tn_total, reduction="macro")
    precision = smp.metrics.precision(tp_total, fp_total, fn_total, tn_total, reduction="macro")

    metrics_dict = {
        "Jaccard": iou_score.cpu().item(),
        "F1": f1_score.cpu().item(),
        "Recall": recall.cpu().item(),
        "Precision": precision.cpu().item(),
        "Acc": accuracy.cpu().item(),
    }

    per_class_metrics = _calculate_class_metric(tp_total, fp_total, fn_total, tn_total)
    # return metrics_dict, per_class_metrics, contour_statistic
    return metrics_dict, per_class_metrics

def _calculate_class_metric(tp_total, fp_total, fn_total, tn_total) -> Dict:


    per_class_metrics = {}

    for class_idx in range(config.task.num_classes):

        iou_score = smp.metrics.iou_score(
            tp_total[class_idx], fp_total[class_idx], fn_total[class_idx], tn_total[class_idx], 
            reduction="macro"
        )
        f1_score = smp.metrics.f1_score(
            tp_total[class_idx], fp_total[class_idx], fn_total[class_idx], tn_total[class_idx],
            reduction="macro"
        )
        accuracy = smp.metrics.accuracy(
            tp_total[class_idx], fp_total[class_idx], fn_total[class_idx], tn_total[class_idx],
            reduction="macro"
        )
        recall = smp.metrics.recall(
            tp_total[class_idx], fp_total[class_idx], fn_total[class_idx], tn_total[class_idx],
            reduction="macro"
        )
        precision = smp.metrics.precision(
            tp_total[class_idx], fp_total[class_idx], fn_total[class_idx], tn_total[class_idx],
            reduction="macro"
        )
        class_name = config.task.labels[class_idx]

        per_class_metrics[class_name] = {
            "Jaccard": iou_score.cpu().item(),
            "F1": f1_score.cpu().item(),
            "Recall": recall.cpu().item(),
            "Precision": precision.cpu().item(),
            "Acc": accuracy.cpu().item(),
        }

    return per_class_metrics

def get_masked_region(
    image: torch.Tensor, 
    label: torch.Tensor, 
    y_pred: torch.Tensor
) -> torch.Tensor:
    """
    Get transparent regions of the prediction mask and gt.
    
    param:
        image: torch.Tensor
            Original image.
        label: torch.Tensor
            GT mask.
        y_pred: torch.Tensor
            Predict mask.
    return:
        torch.Tensor
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    y_pred_np = (y_pred_np * 255).astype(np.uint8)

    colored_mask_pred = np.zeros_like(image_np)
    colored_mask_pred[:, :, 2] = 255 
    
    alpha = 0.1 
    masked_image_pred = cv2.addWeighted(image_np, 1, colored_mask_pred, alpha, 0)
    masked_region = np.where(y_pred_np[..., None] != 0, masked_image_pred, image_np)
    
    return masked_region.squeeze(0)

def count_contours_per_class(
    labels: torch.Tensor,
    y_pred: torch.Tensor,
    min_area: int = 10,
    verbose: bool = True
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Подсчитывает количество контуров для каждого класса в наборе масок.

    Args:
        labels: Tensor of shape [B, H, W] или [B, 1, H, W] - ground truth masks
        y_pred: Tensor of shape [B, C, H, W] - predicted masks (после softmax/sigmoid)
        min_area: минимальная площадь контура для учёта (в пикселях)
        verbose: показывать прогресс-бар

    Returns:
        Кортеж (contour_counts_labels, contour_counts_y_pred)
    """
    
    # Конвертируем в numpy и сжимаем лишние измерения
    labels_np = labels.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # Если labels имеет размерность [B, 1, H, W], меняем на [B, H, W]
    if len(labels_np.shape) == 4:
        labels_np = labels_np.squeeze(1)  # удаляем канальное измерение

    # Для y_pred берем argmax по каналам
    if len(y_pred_np.shape) == 4:
        y_pred_np = np.argmax(y_pred_np, axis=1)  # [B, H, W]

    num_classes = len(config.task.labels)

    # Инициализируем словари с нулевыми значениями
    contour_counts_labels = {config.task.labels[class_id]: 0 for class_id in range(num_classes)}
    contour_counts_y_pred = {config.task.labels[class_id]: 0 for class_id in range(num_classes)}

    batch_size = len(labels_np)

    for i in range(batch_size):
        label_mask = labels_np[i]  # [H, W]
        pred_mask = y_pred_np[i]   # [H, W]

        for class_id in range(num_classes):
            # Для ground truth
            binary_mask_labels = (label_mask == class_id).astype(np.uint8)
            if binary_mask_labels.max() > 0:
                binary_mask_labels_8bit = binary_mask_labels * 255
                contours_labels, _ = cv2.findContours(
                    binary_mask_labels_8bit, 
                    cv2.RETR_EXTERNAL,  
                    cv2.CHAIN_APPROX_SIMPLE
                )
                valid_contours_labels = [
                    cnt for cnt in contours_labels 
                    if cv2.contourArea(cnt) >= min_area
                ]
                contour_counts_labels[config.task.labels[class_id]] += len(valid_contours_labels)

            # Для предсказаний
            binary_mask_y_pred = (pred_mask == class_id).astype(np.uint8)
            if binary_mask_y_pred.max() > 0:
                binary_mask_y_pred_8bit = binary_mask_y_pred * 255
                contours_y_pred, _ = cv2.findContours(
                    binary_mask_y_pred_8bit, 
                    cv2.RETR_EXTERNAL,  
                    cv2.CHAIN_APPROX_SIMPLE
                )
                valid_contours_y_pred = [
                    cnt for cnt in contours_y_pred 
                    if cv2.contourArea(cnt) >= min_area
                ]
                contour_counts_y_pred[config.task.labels[class_id]] += len(valid_contours_y_pred)

    return contour_counts_labels, contour_counts_y_pred

def get_per_class_count_stat(
    contour_counts_labels: Dict[str, int], 
    contour_counts_y_pred: Dict[str, int],
    labels: torch.Tensor,
    y_pred: torch.Tensor
):
    """
    Args:
        contour_counts_labels: number of contours in the class gt
        contour_counts_y_pred: number of contours in the class predict
        labels: Tensor of shape [B, C, H, W] - ground truth masks
        y_pred: Tensor of shape [B, C, H, W] - predicted masks
    Returns:
        contour_statistic: metrics and statistics for predicted contours.
    """

    per_contour_statistic = {
                                "Relation": 0,
                                "Relation positive": 0,
                                "Relation iou50": 0,
                                "Relation iou75": 0,
                                "Relation iou95": 0
                        }

    contour_statistic = {config.task.labels[class_id]: per_contour_statistic for class_id in range(len(config.task.labels))}

    k = list(contour_statistic.keys())
    for batch in range(y_pred.shape[0]):
        for label_id in range(y_pred.shape[1]):
            # print(f"labels.shape = {labels.shape}")
            # print(f"y_pred.shape = {y_pred.shape}")
            # print(f"(labels[batch, :, :] == label_id).int().shape = {(labels[batch, :, :] == label_id).int().shape}")

            relation_positive, relatiou50, relatiou75, relatiou95 = compute_contour_iou((labels[batch, :, :] == label_id).int(), y_pred[batch, label_id, :, :])

            contour_statistic[k[label_id]]["Relation positive"] += relation_positive
            contour_statistic[k[label_id]]["Relation iou50"] += relatiou50
            contour_statistic[k[label_id]]["Relation iou75"] += relatiou75
            contour_statistic[k[label_id]]["Relation iou95"] += relatiou95

    # print(f"y_pred.shape[0] = {y_pred.shape[0]}")
    # print(f"contour_statistic[class_name][Relation positive] = {contour_statistic['single roof']['Relation positive']}")
    # print(f"contour_statistic[class_name][Relation positive] / y_pred.shape[0] = {contour_statistic['single roof']['Relation positive'] / y_pred.shape[0]}")
    # print(f"contour_statistic = {contour_statistic}")
    # print(f"contour_counts_y_pred = {contour_counts_y_pred}")
    # print(f"contour_counts_labels = {contour_counts_labels}")
    for class_name in contour_statistic.keys():
        contour_statistic[class_name]["Relation"] = contour_counts_y_pred[class_name] / contour_counts_labels[class_name]
        contour_statistic[class_name]["Relation positive"] = contour_statistic[class_name]["Relation positive"] / y_pred.shape[0]
        contour_statistic[class_name]["Relation iou50"] = contour_statistic[class_name]["Relation iou50"] / y_pred.shape[0]
        contour_statistic[class_name]["Relation iou75"] = contour_statistic[class_name]["Relation iou75"] / y_pred.shape[0]
        contour_statistic[class_name]["Relation iou95"] = contour_statistic[class_name]["Relation iou95"] / y_pred.shape[0]

    return contour_statistic

def compute_contour_iou(pred_mask: torch.Tensor, 
                        label_mask: torch.Tensor, 
                        threshold: float = 0.01,
                        min_contour_area: int = 10):
    """
    Вычисляет IoU для контуров между предсказанием и разметкой.
    
    Args:
        pred_mask: torch.Tensor [H, W] - предсказанная маска одного класса
        label_mask: torch.Tensor [H, W] - истинная маска одного класса
        pred_threshold: float - порог для бинаризации предсказания
        min_contour_area: int - минимальная площадь контура для учета
    
    Returns:
        dict: словарь с метриками для каждого контура
    """
    per_contour_statistic = {
                                "Relation positive": 0,
                                "Relation iou50": 0,
                                "Relation iou75": 0,
                                "Relation iou95": 0
                        }
    predict = pred_mask.cpu().detach().numpy()
    binary_pred = (predict > threshold).astype(np.uint8)
    
    # print(f"binary_pred.shape = {binary_pred.shape}")
    # print(f"type(binary_pred) = {type(binary_pred)}")
    
    label_np = label_mask.cpu().detach().numpy()
    binary_label = (label_np > threshold).astype(np.uint8)

    pred_contours = measure.find_contours(binary_pred, 0.5)
    label_contours = measure.find_contours(binary_label, 0.5)
    
    height, width = binary_pred.shape
    
    for pred_contour in pred_contours:

        pred_centroid = np.mean(pred_contour, axis=0).astype(np.int64)

        if bool(label_mask[pred_centroid[0], pred_centroid[1]] != 0):

            number_contour = 100000
            find_countour = False

            for i, label_contour in enumerate(label_contours):

                label_centroid = np.mean(label_contour, axis=0).astype(np.int64)

                if np.sqrt((pred_centroid[0] - label_centroid[0])**2 + (pred_centroid[1] - label_centroid[1])**2) < number_contour:
                    number_contour = i
                    find_countour = True

            if find_countour:
                per_contour_statistic['Relation positive'] += 1
                label_contour = label_contours[number_contour]
            
                pred_mask_single = np.zeros((height, width), dtype=np.uint8)
                label_mask_single = np.zeros((height, width), dtype=np.uint8)
                
                pred_contour_int = np.round(pred_contour).astype(int)
                label_contour_int = np.round(label_contour).astype(int)
                
                rr, cc = polygon(pred_contour_int[:, 0], pred_contour_int[:, 1], 
                                         pred_mask_single.shape)
                pred_mask_single[rr, cc] = 1
                
                rr, cc = polygon(label_contour_int[:, 0], label_contour_int[:, 1],
                                         label_mask_single.shape)
                label_mask_single[rr, cc] = 1
                
                intersection = np.logical_and(pred_mask_single, label_mask_single).sum()
                union = np.logical_or(pred_mask_single, label_mask_single).sum()
                
                iou = intersection / union if union > 0 else 0.0

                if iou >= 0.5:
                    per_contour_statistic["Relation iou50"] += 1
                if iou >= 0.75:
                    per_contour_statistic["Relation iou75"] += 1
                if iou >= 0.95:
                    per_contour_statistic["Relation iou95"] += 1

    return  list(per_contour_statistic.values())