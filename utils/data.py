import albumentations
import pandas as pd
import torch
import numpy as np 
import random
import cv2

from tqdm import tqdm
from pathlib import Path, PosixPath
from torch.utils.data import DataLoader
from utils.dataset_class import DatasetClass
from utils.utils import read_config
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedShuffleSplit
from utils.utils import get_image_in_folder

config = read_config("./config.yml")

def worker_init_fn(worker_id: int) -> None:
    """
    Fix seed for current worker. This should fix the bug with identical augs.

    Args:
        worker_id: ID of current dataloader worker.
    """
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(str(seed))
    torch.manual_seed(seed)


def split_train_test_eval(
    images: List[PosixPath], 
    masks: List[PosixPath]
) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
    """
    The function divides the dataset into split training, test and eval.
    
    param:
        images: list[str]
            Paths to images in all directories.
        masks: list[str]
            Paths to masks in all directories.
    reutrn:
        Tuple[list, list, list]
            The image lists are divided into selections.
    """
    data = []

    for image in images:
        mask_path = image.parent.parent / 'masks' / image.name

        if mask_path.exists():
            data.append({
                'file_name': image.name,
                'image_path': image,
                'mask_path': mask_path,
                'city': image.parent.parent.name
            })
    df = pd.DataFrame(data)
    city_to_num = {city: idx for idx, city in enumerate(df['city'].unique())}
    df['split'] = df['city'].map(city_to_num)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25)

    for train_index, test_index in sss.split(df, df.split):
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]

    for test_index, val_index in sss.split(df_test, df_test.split):
        new_df_test, df_val = df_test.iloc[test_index], df_test.iloc[val_index]
        
    return df_train, new_df_test, df_val

def build_dataloaders(
    config: Dict, 
    augmentations: Dict[str, albumentations.Compose]
) -> Tuple[DataLoader, DataLoader]:
    """
    Build a dataloader.
    
    param:
        config: addict.Dict
            Configuration for training the model.
        augmentations: dict[str, albumentations.Compose]
            Aug config.
    return:
        tuple[DataLoader, DataLoader]
    """
    image_all_path = []
    for dataset in config.path:
        d = get_image_in_folder(config.path[dataset])
        image_all_path.extend(d)

    images = [i for i in image_all_path if 'image' in i.as_posix()]
    masks = [m for m in image_all_path if 'masks' in m.as_posix()]
    df_train, df_test, df_val = split_train_test_eval(images, masks)
    print(f"df_train = {len(df_train)}")
    print(f"df_test = {len(df_test)}")
    print(f"df_val = {len(df_val)}")
    # df_train = df_train.head(2)
    # df_test = df_train
    # df_val = df_train
    contour_counts_train = count_contours_per_class(mask_paths=df_train.mask_path.values)
    contour_counts_test = count_contours_per_class(mask_paths=df_test.mask_path.values)
    contour_counts_val = count_contours_per_class(mask_paths=df_val.mask_path.values)

    contour_counts = {}
    for key in contour_counts_train.keys():
        contour_counts[key] = [contour_counts_train[key], contour_counts_test[key], contour_counts_val[key]]


    train_dataset = DatasetClass(df_train.image_path.values, df_train.mask_path.values, augmentations['train'])
    test_dataset = DatasetClass(df_test.image_path.values, df_test.mask_path.values, augmentations['test'])
    val_dataset = DatasetClass(df_val.image_path.values, df_val.mask_path.values, augmentations['test'])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.training.batch_size_train,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training.batch_size_test,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.training.batch_size_test,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn
    )

    return train_loader, test_loader, val_loader, contour_counts

def count_contours_per_class(
    mask_paths: List[str], 
    min_area: int = 10, 
    verbose: bool = True
) -> Dict[int, int]:
    """
    Подсчитывает количество контуров для каждого класса в наборе масок.

    Args:
        mask_paths: список путей к файлам масок
        min_area: минимальная площадь контура для учёта (в пикселях)
        verbose: показывать прогресс-бар

    Returns:
        Словарь {class_id: total_count}
    """
    contour_counts = {config.task.labels[class_id]: class_id for class_id in range(len(config.task.labels))}
    mask_iterator = tqdm(mask_paths, desc="Counting contours") if verbose else mask_paths

    for mask_path in mask_iterator:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            continue

        for class_id in range(len(config.task.labels)):
            binary_mask = (mask == class_id).astype(np.uint8) * 255
            # print(f"binary_mask.shape = {binary_mask.shape}")
            # print(f"type(binary_mask) = {type(binary_mask)}")
            contours, _ = cv2.findContours(
                binary_mask, 
                cv2.RETR_EXTERNAL,  
                cv2.CHAIN_APPROX_SIMPLE
            )

            valid_contours = [
                cnt for cnt in contours 
                if cv2.contourArea(cnt) >= min_area
            ]

            contour_counts[config.task.labels[class_id]] += len(valid_contours)

    return contour_counts



