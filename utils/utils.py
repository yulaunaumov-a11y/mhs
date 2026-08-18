import addict
import albumentations
import pydoc
import inspect
import types
import typing as tp
import yaml

from pathlib import Path, PosixPath
from torch.utils.data import DataLoader
from utils.dataset_class import DatasetClass
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedShuffleSplit

def read_config(config_path: str) -> Dict:
    """
    Read config transform to addict dictionary.
    
    param:
        config_path: str
            Path to configuration file.
    
    returns:
        Configuration addict object.
    """
    with Path(config_path).open() as yf:
        return addict.Dict(yaml.safe_load(yf))
    
def get_image_in_folder(path_to_image: str) -> List[PosixPath]:
    """
    Search for image paths and image names.

    param:
        directory: str
            Directory for image search.

    returns:
        image_paths: list[PosixPath]
            List of image paths.
    """
    return [file_name for file_name in Path(path_to_image).rglob("*")
                   if file_name.suffix in [".png", ".jpeg", ".jpg", ".tif", ".tiff"] and ".ipynb_checkpoints" not in file_name.as_posix()]

def object_from_dict(
    dict_repr: Dict,
    parent: tp.Optional[Dict] = None,
    **additional_kwargs,
) -> tp.Any:
    """
    Parse dictionary and build instance of provided type.

    param:
        dict_repr: dict
            Dict with object description;
        parent: tp.Optional[dict]
            Parent dict;
        **additional_kwargs
            Additional parameters.

    returns:
        Object.
    """
    if dict_repr is None:
        return None
    object_type = dict_repr.pop("__name__")
    for param_name, param_value in additional_kwargs.items():
        dict_repr[param_name] = param_value
    if parent is not None:
        return getattr(parent, object_type)(**dict_repr)

    callable_ = pydoc.locate(object_type)

    if isinstance(callable_, types.FunctionType):
        return callable_

    args = []
    signature = inspect.signature(callable_)
    for param_name, param_value in signature.parameters.items():
        if param_value.kind == param_value.VAR_POSITIONAL:
            config_value = dict_repr.pop(param_name)
            if _is_iterable(config_value):
                args.extend(list(config_value))
            else:
                args.append(config_value)
    return callable_(*args, **dict_repr)
    