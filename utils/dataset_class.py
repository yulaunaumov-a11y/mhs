import cv2
import typing as tp
import torch
import albumentations as A
from torch.utils.data import Dataset

class DatasetClass(Dataset):
    def __init__(self, images_path:tp.Iterable[str],
                       masks_path:tp.Iterable[str],
                       transforms:A.Compose,
                       num_classes: int = 4):
        """
        Initialize dataset.

        param:
            images_path: tp.Iterable[str]
                Paths to images
            masks_path: tp.Iterable[str]
                Paths to masks
            transforms: tp.Optional[A.Compose]
                Augmentations
            num_classes: int
                Count classes
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.transforms = transforms
        self.num_classes = num_classes

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index:int)->tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.

        param:
            index: int
                Index of item.

        return: 
            tp.Tuple[torch.Tensor, torch.Tensor]
                Tuple of image and mask.
        """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            aug= self.transforms(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']


        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask