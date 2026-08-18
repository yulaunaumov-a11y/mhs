import albumentations
from albumentations.core import serialization
from typing import Dict

def get_augmentations(config: Dict) -> Dict[str, albumentations.Compose]:
    """
    Build augmentation for train, val and test.
    
    param:
        config: Dict
            Aug config.
    returns:
        Aug for train, val and test.
    """
    
    train_augs = serialization.from_dict(config.augmentations.train)
    test_augs = serialization.from_dict(config.augmentations.test)
    
    return {"train": train_augs, "test": test_augs}