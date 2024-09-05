import cv2
import os
import argparse

def images_to_video(image_folder, video_name, fps):
    # Get all the images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure the images are in the correct order

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)  # Write the frame to the video

    # Release the VideoWriter object
    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a sequence of images to a video.")
    parser.add_argument("-i", "--image_folder", type=str, required=True, help="Path to the folder containing the images.")
    parser.add_argument("-o", "--video_name", type=str, required=True, help="Name of the output video file.")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frames per second for the video (default: 30).")

    args = parser.parse_args()

    images_to_video(args.image_folder, args.video_name, args.fps)