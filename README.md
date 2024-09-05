# Practical - Applied Foundation Models: Project on Video Editing and AR

This is the code for our project on Video Editing and AR for the Applied Foundation Models Practical. Goal was to build a framework that allows the replacement of one object in a video by another object using computer vision foundation models. The project builds upon [FoundationPose](https://github.com/NVlabs/FoundationPose) and [Track-Anything](https://github.com/gaomingqi/Track-Anything).

Check out our project report [here](https://drive.google.com/file/d/1BngTZSLHP6cIUMecsCGy_NqMoW8SLl_Y/view?usp=drive_link).

## Results
Here a few edited videos and the corresponding original videos are depicted (might take a few seconds to load).

<p align="center">
  <img src="assets/videos/gun_ego.gif" width="200" />
  <img src="assets/videos/gun_ego_raw.gif" width="200" />
  <img src="assets/videos/sword_ego_short.gif" width="200" />
  <img src="assets/videos/sword_ego_raw_short.gif" width="200" />
</p>

<p align="center">
  <img src="assets/videos/sword_spin.gif" width="200" />
  <img src="assets/videos/sword_spin_raw.gif" width="200" />
  <img src="assets/videos/gun_run_short.gif" width="200" />
  <img src="assets/videos/gun_run_raw_short.gif" width="200" />
</p>




## Demo

In case you want to try our framework without recording your own video, demo data can be downloaded from [here]( https://drive.google.com/drive/folders/1wCDiG25If92xcxFBMIh6d-QjBcaW9ELb?usp=sharing). This folder contains a recorded RGBD video, results of FoundationPose and Track-Anything, and a mesh for inserting. 

## Usage

### 3D Models
Required 3D meshes (format has to be compatible with trimesh.load()):
* Textured 3D mesh of object that will be tracked in the video
* Textured 3D mesh of object to insert to video

Make sure that coordinate frames of the different meshes are aligned in a senseful way and the scales match!

### RGBD Video
Record a RGBD video with the object you want to track (and replace) in it. Fast movements of the object lead to error prone tracking with FoundationPose. If you want to deal with hand occlusions later on, it is beneficial to wear a colorful glove on the hand which is holding the object.   

Make sure that the RGB frames and the depth frames are aligned properly. The frames should be stored as individal .png files. Use 16bit unsigned values and a milimenter scale for the depth values. Also the camera intrinsic matrix is needed as a .txt file. If you are using an Intel Realsense camera the provided script realsense_align.py can be used to record and safe aligned videos. The script also extracts the camera intrinsics automatically.

Note that for the first frame a mask of the object is required. This can be obtained with Microsoft Paint, SAM, etc.

### FoundationPose
Set up FoundationPose as described [here](https://github.com/NVlabs/FoundationPose). The experimental conda environment should be working. Create an input folder as follows:
* INPUT/
    * rgb/
    * depth/
    * masks/
    * mesh/
    * cam_K.txt

Use the FoundationPose run_demo.py script with debugmode=2 to track the object. Afterwards safe the ob_in_cam/ folder, this contains the 4x4 pose matrices for each frame. To verify if your tracking worked you can check out the track_vis/ folder.

### Track-Anything (to deal with occlusions)
Set up Track-Anything as described [here](https://github.com/gaomingqi/Track-Anything) and get familiar with the tutorial. Upload the video in .mp4 format (to convert from single .png frames to .mp4 the provided pngs_to_mp4.py can be used). Create a mesk for the occluding object (e.g. the hand) and track it in every frame. Afterwards safe the mask folder from results.

### Postprocessing and Rendering
For the postprocessing and rendering see the documentation provided in main.ipynb. A conda env can be created with the given environment.yaml file.

## Acknowledgement
We would like to thank the authors of FoundationPose and Track-Anything. Also special thanks goes out to our supervisor Tarun Yenamandra.

## Authors
Jonathan Evers, Pontus Björklid, Utku Turkbey. Technical University Munich. Contact via github.






