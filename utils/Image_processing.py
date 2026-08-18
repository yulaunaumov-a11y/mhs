import numpy as np
import cv2
from pathlib import Path, PosixPath
from ImgTiler import SplitImage

def get_rgb_image(
                red_image_path: PosixPath,
                green_image_path: PosixPath,
                blue_image_path: PosixPath
                ) -> np.ndarray:
    """
    Get RGB image from gray images.
    
    param:
        red_image_ventura_path: PosixPath
            Image of the red channel.
        green_image_ventura_path: PosixPath
            Image of the green channel.
        blue_image_ventura_path: PosixPath
            Image of the blue channel.
    Return:
       np.ndarray
           RGB image.
    """
    red_channel = cv2.imread(red_image_path, cv2.IMREAD_UNCHANGED)
    green_channel = cv2.imread(green_image_path, cv2.IMREAD_UNCHANGED)
    blue_channel = cv2.imread(blue_image_path, cv2.IMREAD_UNCHANGED)
    
    return cv2.merge([blue_channel, green_channel, red_channel])


def split_image_into_patch(image: np.ndarray,
                           masks: np.ndarray,
                           px_in_patch: int,
                           path_to_save: PosixPath,
                           name: str
                          ) -> None:
    """
    Split the image into patches.
    
    param:
        image: np.ndarray
            RGB image.
            
        masks: np.ndarray
            Masks imge.
            
        px_in_patch: int
            Number of pixels for the patch size, the patch is square.
    
        path_to_save: PosixPath
            Path to saving.
    Return:
       None
    """
    H = image.shape[0]
    W = image.shape[1]
    
    grid = (H // px_in_patch, W // px_in_patch)
    overlap = 0
    
    show_rects = False
    show_image = False
    path_to_image = path_to_save / "image"
    path_to_mask = path_to_save / "masks"
    save_success_img = []
    save_success_mask = []

    splitter_x = SplitImage(image, grid, overlap)
    splitter_y = SplitImage(masks, grid, overlap)

    tiles_x = splitter_x.split_image(show_rect=show_rects, show_tiles=show_image)
    tiles_y = splitter_y.split_image(show_rect=show_rects, show_tiles=show_image)
    filtered_x = []
    filtered_y = []
    for tile_x, tile_y in zip(tiles_x, tiles_y):
        if np.any(tile_y > 0):
            filtered_x.append(tile_x)
            filtered_y.append(tile_y)

    save_success_img = [cv2.imwrite(path_to_image / (name + str(idx) + ".png"), one_tile) 
                                        for idx, one_tile in enumerate(filtered_x)]

    save_success_mask = [cv2.imwrite(path_to_mask / (name + str(idx) + ".png"), one_tile) 
                                        for idx, one_tile in enumerate(filtered_y)]