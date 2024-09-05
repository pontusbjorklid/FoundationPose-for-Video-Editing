import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

def overlay_object(background_img_folder, rendered_images, rendered_depths):
    """
    Overlayers rendered image onto background image for each frame

    Args:
        background_img_folder: Path to folder containing rgb background images
        rendered_iamges: Numpy array of rendered images, shape NxHxWx3
    """

    image_files = sorted([f for f in os.listdir(background_img_folder) if f.endswith('.png')])

    # Make sure number of frames fit
    if len(image_files) != rendered_images.shape[0]:
        print("Warning: The number of image files and text files do not match!")

    video = np.zeros((rendered_images.shape[0], rendered_images.shape[1], rendered_images.shape[2], 3))

    for i in tqdm(range(len(image_files)), desc="Overlay images"):
        background_frame_file = image_files[i]
        rendered_frame = rendered_images[i]
        rendered_depth = rendered_depths[i]

        img_path = os.path.join(background_img_folder, background_frame_file)

        background = Image.open(img_path)
        background = np.array(background)

        mask = (rendered_depth != 0).astype(np.uint8)
        mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        video[i] = np.where(mask_rgb, rendered_frame, background)
    
    return video