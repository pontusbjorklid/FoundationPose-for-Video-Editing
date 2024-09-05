import os
import numpy as np
from tqdm import tqdm

def mask_occluded_parts(masks_path, rendered_images, rendered_depths):
    """
    Deals with hand occlusion. Hand_masks_path is path to folder of numpy arrays containing masks.

    Args:
        masks_path: Path to masks of occlusions
        rendered_images: Numpy array of rendered images of shape NxHxWx4
        rendered_depths: Nummpy array of rendered depth of shape NxHxW

    Returns:
        Masked rendered images, Numpy array of shape NxHxWx3
    """

    # Get a sorted list of .npy files in the directory
    file_list = sorted([f for f in os.listdir(masks_path) if f.endswith('.npy')])

    # Load each file and stack them into a single numpy array
    array_list = [np.load(os.path.join(masks_path, file)) for file in file_list]

    # Stack the arrays along a new dimension (axis 0)
    masks = np.stack(array_list, axis=0)

    masked_rendered_images = rendered_images
    masked_rendered_depths = rendered_depths
    for i in tqdm(range(masks.shape[0]), desc="Masking out occlusion"):
        mask = masks[i]
        img = rendered_images[i]
        depth_img = rendered_depths[i]

        boolean_mask = mask.astype(bool)
        # Set the pixels in the image to white where the mask is True
        white_pixels = np.ones_like(img) * 255
        depth_zero = np.zeros_like(depth_img)
        img[boolean_mask] = white_pixels[boolean_mask]
        depth_img[boolean_mask] = depth_zero[boolean_mask]


        masked_rendered_images[i] = img
        masked_rendered_depths[i] = depth_img
    
    return masked_rendered_images.astype(np.uint8), masked_rendered_depths