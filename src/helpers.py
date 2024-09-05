import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


def get_object_bounding_box(rendered_images: np.array):
    """
    Gets the rectangular area of every object in the rendered images.

    Args: 
        rendered_images: Numpy array with rendered images, shape NxHxWx3

    Returns:
        object_areas: Numpy array [Frames, 4] with entries (x, y, w, h)
    """
    F, H, W, _ = rendered_images.shape

    object_bb = np.zeros((rendered_images.shape[0], 4))

    all_white_col_indicator = H * 3 * 255
    all_white_row_indicator = W * 3 * 255

    for i in range(F):
        image = rendered_images[i]

        # Get x starting value and width of bounding box
        start_found = False
        for j in range(image.shape[1]):
            image_col = image[:, j, :]

            if not start_found and np.sum(image_col) != all_white_col_indicator:
                start_found = True
                x = j
            if start_found and np.sum(image_col) == all_white_col_indicator:
                start_found = False
                w = j - x

        if start_found == False:
            Warning("No object found in rendered image " + str(i))
        
        # Get y starting value and height of bounding box
        start_found = False
        for j in range(image.shape[0]):
            image_row = image[j, :, :]

            if not start_found and np.sum(image_row) != all_white_row_indicator:
                start_found = True
                y = j
            if start_found and np.sum(image_row) == all_white_row_indicator:
                start_found = False
                h = j - y

        object_bb[i] = np.array([x, y, w, h])

    return object_bb.astype(np.uint8)


def crop_bb_from_image(background_img_folder, bounding_boxes, output_folder):
    """
    Crops bounding box area from image and saves as png

    Args:
        background_img_folder: Path to folder of background images
        bounding boxes: Numpy array of bounding boxes
        output_folder: Path to folder where outputs are saved
    """

    image_files = sorted([f for f in os.listdir(background_img_folder) if f.endswith('.png')])


    for i in tqdm(range(bounding_boxes.shape[0]), desc="Crop Images"):

        img_path = os.path.join(background_img_folder, image_files[i])
        img = Image.open(img_path)
        img = np.array(img)

        bb = bounding_boxes[i]

        x = bb[0]
        y = bb[1]
        w = bb[2]
        h = bb[3]
        cropped_img = img[y:y+h+1, x:x+w+1, :]

        cropped_img = Image.fromarray(cropped_img.astype(np.uint8))
        cropped_img.save(f'{output_folder}/{i+1:07d}.png')
        

def images_to_video(frames, video_name, fps=30):
    """
    Convert numpy array of videoframes to mp4 video

    Args:
        frames: Numpy array, NxHxWx3
        video_name = Full name of output mp4 file
        fps: Framerate
    """
    # Read the first image to get the dimensions
    height, width, layers = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in frames:
       #frame = cv2.imread(image_path)
        bgr_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        video.write(bgr_image)  # Write the frame to the video

    # Release the VideoWriter object
    video.release()


# def vidwrite(fn, images, framerate=30, vcodec='libx264'):
#     if not isinstance(images, np.ndarray):
#         images = np.asarray(images)
#     n,height,width,channels = images.shape
#     process = (
#         ffmpeg
#             .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
#             .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
#             .overwrite_output()
#             .run_async(pipe_stdin=True)
#     )
#     for frame in images:
#         process.stdin.write(
#             frame
#                 .astype(np.uint8)
#                 .tobytes()
#         )
#     process.stdin.close()
#     process.wait()