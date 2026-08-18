"""This module contains implementation of logger for ClearML."""
from pathlib import Path
import datetime
import cv2
import addict
from clearml import Task
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Dict
from torch.utils.data import DataLoader
from utils_for_inference.image_proccesing import get_binary_mask
from utils.utils import read_config
from sklearn.metrics import confusion_matrix

import io
from PIL import Image

class Logger:
    """Implements logging to ClearML."""

    def __init__(self, config: Dict) -> None:
        """
        Initialize logger.

        Args:
            config: Project configuration object.
        """
        self.config = config
        self.job_time = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")
        Task.set_credentials(
            web_host=config.project.web_host,
            api_host=config.project.api_host,
            files_host=config.project.files_host,
            key=config.project.key,
            secret=config.project.secret
        )
        self.task = Task.init(project_name=config.project.name,
                              task_name=f"{config.project.name}_{self.job_time}",
                              tags=[config.model.encoder_name], 
                              auto_connect_frameworks=False)
        self.task.connect(config)
        self.logger = self.task.get_logger()
        self.task.upload_artifact("train_config.yml", artifact_object="config.yml")
        self.folder_name_to_save = Path(self.config.project.path_weights, f"weights_{datetime.datetime.now().isoformat()}")
        self.folder_name_to_save.mkdir(exist_ok=True)

    def log_loss(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """
        Log train and validation loss.

        Args:
            epoch: Epoch number;
            train_loss: Train loss;
            val_loss: Val loss;
        """
        self.logger.report_scalar(title="Losses", series="Train loss", iteration=epoch, value=train_loss)
        self.logger.report_scalar(title="Losses", series="Val loss", iteration=epoch, value=val_loss)

    def log_train_and_val_metrics(self, epoch: int, metrics_dict: Dict, title:str = "Metrics") -> None:
        """
        Log train and validation metrics.

        Args:
            epoch: Epoch number;
            metrics_dict_val: Results of evaluation encapsulated in separate object.
        """

        self.logger.report_scalar(
            title=title,
            series="Jaccard",
            iteration=epoch,
            value=metrics_dict["Jaccard"],
        )
        self.logger.report_scalar(
            title=title,
            series="F1",
            iteration=epoch,
            value=metrics_dict["F1"],
        )
        self.logger.report_scalar(
            title=title,
            series="Recall",
            iteration=epoch,
            value=metrics_dict["Recall"],
        )
        self.logger.report_scalar(
            title=title,
            series="Precision",
            iteration=epoch,
            value=metrics_dict["Precision"],
        )
        self.logger.report_scalar(
            title=title,
            series="Acc",
            iteration=epoch,
            value=metrics_dict["Acc"],
        )

    def log_test_metrics(self, test_results: Dict) -> None:
        """
        Log test metrics.

        Args:
            epoch: Number of epoch with best validation metric;
            test_results: Results of evaluation encapsulated in separate object.
        """
        self.logger.report_scalar(
            title="Metrics test",
            series="Jaccard",
            iteration=1,
            value=test_results["Jaccard"],
        )
        self.logger.report_scalar(
            title="Metrics test",
            series="F1",
            iteration=1,
            value=test_results["F1"],
        )
        self.logger.report_scalar(
            title="Metrics test",
            series="Recall",
            iteration=1,
            value=test_results["Recall"],
        )
        self.logger.report_scalar(
            title="Metrics test",
            series="Precision",
            iteration=1,
            value=test_results["Precision"],
        )
        self.logger.report_scalar(
            title="Metrics test",
            series="Acc",
            iteration=1,
            value=test_results["Acc"],
        )

    def log_learning_rate(self, lr: float, epoch: int) -> None:
        """
        Log learning rate to ClearML
        
        Args:
            lr: Learning rate value
            epoch: Current epoch
        """
        self.logger.report_scalar(
            title="Learning Rate",
            series="LR",
            value=lr,
            iteration=epoch
        )

    def log_model_weights(self, model_weights: Dict[str, torch.Tensor], epoch: int, name: str) -> None:
        """
        Save model's weights on the best epoch to the ClearML.

        Args:
            epoch: Number of epoch with best validation metric;
            model_weights: Model's state dictionary.
        """
        time = datetime.datetime.now()

        matching_files = list(self.folder_name_to_save.glob(f"*{name}*"))

        for file_path in matching_files:
            if file_path.is_file():
                file_path.unlink()

        file_name = f"{name}_{self.config.project.name}_{time}_epoch:{epoch}.pth"
        weights_path = Path(self.folder_name_to_save.as_posix(), file_name)
        torch.save(model_weights.state_dict(), weights_path.as_posix())
        self.task.upload_artifact("model_weights.pth", artifact_object=weights_path.as_posix())

    def model_predict_for_log(self,
                            model: torch.nn.Module,
                            dataloader: DataLoader,
                            epoch: int,
                            device: str,
                            title: str = "Image Predictions") -> None:
        model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)

                y_pred = model(images)

                break
        self._log_images_with_masks(images, labels, y_pred, epoch, title)

    def _log_images_with_masks(self,
                            images: torch.Tensor,
                            gt_masks: torch.Tensor,
                            pred_masks: torch.Tensor,
                            epoch: int,
                            title: str = "Image Predictions") -> None:
        """
        Log images with ground truth and predicted masks to ClearML.
        
        Args:
            images: Tensor of shape [B, C, H, W] - original images
            gt_masks: Tensor of shape [B, C, H, W] - ground truth masks
            pred_masks: Tensor of shape [B, C, H, W] - predicted masks
            epoch: Current epoch number
            title: Title for the plot in ClearML
        """

        images_np = images.detach().cpu().numpy()

        for idx in range(len(images_np)):
            img = images_np[idx].transpose(1, 2, 0)

            if img.shape[2] == 3:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img, 0, 1)

            binary_mask, masked_region, segmented = get_binary_mask(img, pred_masks[idx])
            gt_masks_np = gt_masks.cpu().squeeze().detach().numpy()
            gt_mask = gt_masks_np[idx]
            # gt_mask = gt_masks_np

            fig, axes = plt.subplots(1, 5, figsize=(16, 6))
    
            axes[0].imshow(img)
            axes[0].set_title(f'Image {idx}')
            axes[0].axis('off')

            axes[1].imshow(gt_mask, cmap='gray')
            axes[1].set_title(f'GT Mask {idx}')
            axes[1].axis('off')

            axes[2].imshow(binary_mask)
            axes[2].set_title(f'Pred Mask {idx}')
            axes[2].axis('off')

            axes[3].imshow(masked_region)
            axes[3].set_title(f'Masked Region {idx}')
            axes[3].axis('off')

            axes[4].imshow(segmented)
            axes[4].set_title(f'Segmented {idx}')
            axes[4].axis('off')

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg', dpi=150)
            plt.close(fig)

            buf.seek(0)
            pil_image = Image.open(buf)

            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            self.logger.report_image(
                title=title,
                series=f"title {idx}",
                iteration=epoch,
                image=pil_image,
                max_image_history=1000
            )

            # self.logger.report_matplotlib_figure(
            #     title=title,
            #     series=f"title {idx}",
            #     iteration=epoch,
            #     figure=fig,
            #     report_image=pil_image
            # )
            buf.close()

            if idx == 7:
                break

    def _distribution_classes(self, contour_counts: Dict) -> None:
        """
        Log class distribution statistics to ClearML as a table.
        
        Args:
            contour_counts: Dictionary containing contour counts per class for different dataset splits.
        """
        df = pd.DataFrame(contour_counts,
                        index=["train", "test", "val"],
                        )
        df.index.name = "id"
        self.logger.current_logger().report_table(
            title="Расспредление классов",
            series="Классы",
            iteration=0,
            table_plot=df
        )

    def log_confused_matrix(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        epoch: int,
        num_classes: int
        ) -> None:
        """
        Calculate and visualize confusion matrix for segmentation model.
        """
        model.eval()
        model.to(device)

        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                y_pred = model(images)
                y_pred_processed = torch.argmax(y_pred, dim=1)

                all_predictions.append(y_pred_processed.cpu().numpy().flatten())
                all_labels.append(labels.cpu().numpy().flatten())

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        cm = confusion_matrix(all_labels, all_predictions, 
                              labels=np.arange(num_classes))
        config = read_config("./config.yml")
        class_names = config.task.labels
        df_cm = pd.DataFrame(cm, 
                            index=[f"True_{name}" for name in class_names],
                            columns=[f"Pred_{name}" for name in class_names])

        df_cm['Total_True'] = df_cm.sum(axis=1)
        df_cm.loc['Total_Pred'] = df_cm.sum(axis=0)

        self.logger.current_logger().report_table(
            title=f"Матрица ошибок {epoch}", 
            series=f"Epoch {epoch}", 
            iteration=epoch, 
            table_plot=df_cm
        )
    def log_contour_metrics(self, epoch: int, metrics_dict: Dict, title:str = "Metrics") -> None:
        """
        Log train and validation metrics.

        Args:
            epoch: Epoch number;
            metrics_dict_val: Results of evaluation encapsulated in separate object.
        """

        # Log metrics.
        self.logger.report_scalar(
            title=title,
            series="Relation",
            iteration=epoch,
            value=metrics_dict["Relation"],
        )
        self.logger.report_scalar(
            title=title,
            series="Relation positive",
            iteration=epoch,
            value=metrics_dict["Relation positive"],
        )
        self.logger.report_scalar(
            title=title,
            series="Relation iou50",
            iteration=epoch,
            value=metrics_dict["Relation iou50"],
        )
        self.logger.report_scalar(
            title=title,
            series="Relation iou75",
            iteration=epoch,
            value=metrics_dict["Relation iou75"],
        )
        self.logger.report_scalar(
            title=title,
            series="Relation iou95",
            iteration=epoch,
            value=metrics_dict["Relation iou95"],
        )

    def complete_logging(self) -> None:
        """Complete logging."""
        self.task.close()