import os
import numpy as np
import trimesh
import pyrender
import cv2
from tqdm import tqdm

def render_images_from_poses(pose_folder, mesh_path, img_size=(640, 480)):
    """
    Renders RGB images from the mesh for each pose in pose_folder

    Args:
        pose_folder: Folder containing a 4x4 pose matrix as txt file for each frame of the video
        mesh_path: Path to the 3D model you want to insert
        img_size: Resolution of the frames

    Returns:
        rendered_images: Numpy array: Frames x img_height x img_width x 3
    """

    img_width, img_height = img_size
    
    pose_files = sorted([f for f in os.listdir(pose_folder) if f.endswith('.txt')])

    renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)

    rendered_images = np.zeros((len(pose_files), img_height, img_width, 3))
    rendered_images_depth = np.zeros((len(pose_files), img_height, img_width, 3))

    # Iterate through the files and process them
    for i in tqdm(range(len(pose_files)), desc="Render Object for Frames"):

        pose_file = pose_files[i]
        pose_path = os.path.join(pose_folder, pose_file)
        
        # Render mesh
        pose = np.loadtxt(pose_path)

        # Make sure pose aligns with the camera frame used by pyrender
        camera_pose = np.linalg.inv(pose)
        camera_pose[:, 1:3] *= -1

        # Create pyrender scene
        scene = pyrender.Scene(ambient_light=[1,1,1, 1.0])

        # Get mesh
        mesh = trimesh.load(mesh_path, force="mesh")
        mesh = pyrender.Mesh.from_trimesh(mesh)

        # Define camera parameters
        camera = pyrender.IntrinsicsCamera(
            fx=385.54168701171875, fy=384.4963073730469, 
            cx=323.72271728515625, cy=243.59848022460938,
            znear=0.05, zfar=5000.0
        )

        # Add to scene
        scene.add(mesh)
        scene.add(camera, pose=camera_pose)

        # Render
        color, depth = renderer.render(scene)

        rendered_images[i] = color
        rendered_images_depth[i] = depth

    return rendered_images.astype(np.uint8)

def render_images_from_poses_with_lighting(pose_folder, mesh_path, light_coordinates=None, samples=10, img_size=(640, 480)):
    """
    Renders RGB images from the mesh for each pose in pose_folder. Places lights can be specified with light_coordinates.
    
    Args:
        pose_folder (str): Path to the folder containing pose files (.txt).
        mesh_path (str): Path to the mesh file (e.g., .obj).
        img_size (tuple): Image size (width, height) for rendering.
        light_coordinates (list of np.ndarray): List of 3D coordinates of lights in world space.

    Returns:
        rendered_images: Numpy array: Frames x 480 x 640 x 4
    """

    img_width, img_height = img_size
    
    pose_files = sorted([f for f in os.listdir(pose_folder) if f.endswith('.txt')])

    renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height, point_size=samples)

    rendered_images = np.zeros((len(pose_files), img_height, img_width, 3)) 
    rendered_depths = np.zeros((len(pose_files), img_height, img_width)) 
    
    # Iterate through the files and process them
    for i in tqdm(range(len(pose_files)), desc="Render Object for Frames"):
        # Construct the scene
        if light_coordinates is None:
            scene = pyrender.Scene(ambient_light=[1, 1, 1, 1.0])
        else:
            scene = pyrender.Scene()

        # Get pose
        pose_file = pose_files[i]
        pose_path = os.path.join(pose_folder, pose_file)
        pose = np.loadtxt(pose_path)

        # Align with pyrender camera frame convention
        camera_pose = np.linalg.inv(pose)
        camera_pose[:, 1:3] *= -1

        # Intel Realsense 455
        cam = pyrender.IntrinsicsCamera(
            fx=385.54168701171875, fy=384.4963073730469, 
            cx=323.72271728515625, cy=243.59848022460938,
            znear=0.05, zfar=5000.0
        )
        cam_node = pyrender.Node(camera=cam, matrix=camera_pose)

        # Mesh Node set-up
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        mesh_node = pyrender.Node(mesh=mesh)

        # Add mesh node
        scene.add_node(mesh_node)

        # Add camera node
        scene.add_node(cam_node)
                
        if light_coordinates is not None:        
            # Add light nodes
            for light_coord in light_coordinates:
                light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4.0)
                light_pose = np.eye(4)
                light_pose[:3, 3] = light_coord  # Set position of the light
                light_node = pyrender.Node(light=light, matrix=light_pose)
                scene.add_node(light_node)
                    
        # Render
        color, depth = renderer.render(scene)
        # # Set background of rendering transparent
        # alpha_channel = np.zeros_like(depth, dtype=np.uint8)
        # alpha_channel[depth != 0] = 255
        # color[:, :, 3] = alpha_channel
        
        rendered_images[i] = color
        rendered_depths[i] = depth

    return rendered_images.astype(np.uint8), rendered_depths


def get_top_light_sources(rgb_image_path, depth_map_path, intensity_threshold=200, top_n=3):
    """
    Identifies the top N brightest spots in an image and calculates their real-world coordinates and directions.
    -> VERY HEURISTICAL
    
    Args:
        rgb_image_path (str): Path to the RGB image.
        depth_map_path (str): Path to the depth map.
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix.
        intensity_threshold (int): Intensity threshold for light source detection.
        top_n (int): Number of top light sources to identify.
        
    Returns:
        real_world_coords (np.ndarray): Array of real-world coordinates of the top N light sources.
        directions (np.ndarray): Array of direction vectors for the top N light sources.
    """
    intrinsic_matrix = np.array([
    [385.54168701171875, 0, 323.72271728515625],
    [0, 384.4963073730469, 243.59848022460938],
    [0, 0, 1]
    ])

    # Load RGB image and depth map
    rgb_image = cv2.imread(rgb_image_path)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

    # Convert to grayscale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to isolate potential light sources
    _, binary_image = cv2.threshold(gray_image, intensity_threshold, 255, cv2.THRESH_BINARY)

    # Find contours of potential light sources
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store intensities and contours
    intensities = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            intensity = np.mean(gray_image[cy, cx])
            intensities.append((intensity, contour, (cx, cy)))

    # Sort contours based on intensity (brightest first)
    intensities.sort(reverse=True, key=lambda x: x[0])

    # Select the top N brightest spots
    brightest_spots = intensities[:top_n]

    # Initialize lists to store directions and real-world coordinates
    real_world_coords = []
    directions = []

    for intensity, contour, (cx, cy) in brightest_spots:
        # Compute real-world coordinates
        depth_value = depth_map[cy, cx]
        world_coords = image_to_world_coordinates(cx, cy, depth_value, intrinsic_matrix)
        real_world_coords.append(world_coords)

        # Compute gradient around the centroid to estimate direction
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

        # Calculate gradient direction at centroid
        gradient_x = sobelx[cy, cx]
        gradient_y = sobely[cy, cx]
        
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        if gradient_magnitude > 0:
            direction_x = gradient_x / gradient_magnitude
            direction_y = gradient_y / gradient_magnitude
            direction = np.array([direction_x, direction_y, 0])  # Assuming direction in 2D plane
            directions.append(direction)

    return np.array(real_world_coords), np.array(directions)


def image_to_world_coordinates(u, v, depth, intrinsic_matrix):
    """
    Convert image coordinates with depth to real-world coordinates.
    
    Args:
        u (int): x-coordinate in image space.
        v (int): y-coordinate in image space.
        depth (float): Depth value at the (u, v) position.
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix.
        
    Returns:
        np.ndarray: Real-world coordinates (X, Y, Z).
    """
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    return np.array([X, Y, Z])