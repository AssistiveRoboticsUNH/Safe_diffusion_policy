import h5py
import numpy as np
import pyvista as pv
from tqdm import tqdm  # Import tqdm for progress bars
import os

# Explicitly set PyVista backend to standalone
pv.global_theme.jupyter_backend = 'static'

# Path to the dataset
dataset_path = "demo.hdf5"
f_org = h5py.File(dataset_path, "r")

# Get the list of demos
demos = list(f_org["data"]["demo_4"]["obs"].keys())
print("Available demos:", demos)

demo_data_image = np.array(f_org["data"]["demo_4"]["obs"]['eye_in_hand_rgb'])

print(len(demo_data_image))
demo_data_end_effector = np.array(f_org["data"]["demo_4"]["obs"]['ee_states'])


print(len(demo_data_end_effector))

csv_directory = "lfd-safety/data_with_pose_matrices.csv"

import csv
import numpy as np

# List to store all rows from the CSV file
pose_matrices = []

# Replace 'example.csv' with your CSV file's path or name
with open(csv_directory, mode='r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    count = 0
    r_count =0
    image_matrics=[]
    for row in csv_reader:
        if count ==0:
            count =1
            continue
        if float(row[0])==0 and float(row[1])==0:
            if r_count ==0:
                image_matrics.append(np.array(row).astype(np.float64))
                r_count =1
                continue
            pose_matrices.append(np.array(image_matrics))
            image_matrics=[]
            image_matrics.append(np.array(row).astype(np.float64))
        else:
            image_matrics.append(np.array(row).astype(np.float64))
# Convert the list of poses into a NumPy array for structured data access
pose_matrices = np.array(pose_matrices)

# Example usage of the data
print("Total number of pose matrices:", len(pose_matrices))
print("Shape of the pose matrices array:", pose_matrices.shape)

# Example: Access the first pose matrix
print("First pose matrix:")
print(pose_matrices.shape)

from vivid123.generation_utils import generation_vivid123_new, prepare_vivid123_pipeline

ZERO123_MODEL_ID = "bennyguo/zero123-xl-diffusers"
VIDEO_MODEL_ID = "cerspense/zeroscope_v2_576w"
VIDEO_XL_MODEL_ID = "cerspense/zeroscope_v2_XL"
vivid123_pipe, xl_pipe = prepare_vivid123_pipeline(
        ZERO123_MODEL_ID=ZERO123_MODEL_ID, 
        VIDEO_MODEL_ID=VIDEO_MODEL_ID, 
        VIDEO_XL_MODEL_ID=VIDEO_XL_MODEL_ID
    )


config = {
    "delta_azimuth_end": 0.0,
    "delta_azimuth_start": 0.0,
    "delta_elevation_end": 0.0,
    "delta_elevation_start": 0.0,
    "delta_radius_end": 0.0,
    "delta_radius_start": 0.0,
    "eta": 0.5,
    "guidance_scale_video": 2.0,
    "guidance_scale_zero123": 6.0,
    "height": 256,
    "input_image_path": "duck_without_bg.png",
    "obj_name": "duck",
    "noise_identical_accross_frames": False,
    "num_frames": 1,
    "num_inference_steps": 50,
    "prompt": "a toy duck",
    "refiner_guidance_scale": 1.0,
    "refiner_strength": 0.1,
    "video_end_step_percentage": 1.0,
    "video_linear_end_weight": 0.5,
    "video_linear_start_weight": 1.0,
    "video_start_step_percentage": 0.0,
    "width": 256,
    "zero123_end_step_percentage": 1.0,
    "zero123_linear_end_weight": 1.0,
    "zero123_linear_start_weight": 1.0,
    "zero123_start_step_percentage": 0.0,
}

import torch
from carvekit.api.high import HiInterface

# Check doc strings for more information
interface = HiInterface(object_type="hairs-like",  # Can be "object" or "hairs-like".
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)

import os
from PIL import Image

output_folder = "vivid123_results/"
input_temp = "input_temp/"

# Loop through images and their corresponding pose matrices
for idx, (image_data, pose_matrix) in enumerate(zip(demo_data_image, pose_matrices)):
    # Extract the first three values from the pose matrix:
    # Assumption: pose_matrix = [azimuth, elevation, radius, x, y, z]
    image_path = os.path.join(input_temp, f"image_{idx}.png")
    # Check if the image is in BGR order (if shape[-1] equals 3) and convert it to RGB if needed
    if image_data.shape[-1] == 3:
        image_data = image_data[..., ::-1]  # Reverse the channel order (BGR -> RGB)
    
    # Save the image to the temporary path
    image = Image.fromarray(image_data)
    image.save(image_path)
    images_without_background = interface([image_path])
    cat_wo_bg = images_without_background[0]
    cat_wo_bg.save(image_path)
    input_img_base_counter= 0
    for poses in pose_matrix:

        azimuth, elevation, radius = poses[:3]
        # print(azimuth, elevation, radius)

        # Save the current image as a temporary file
        
        output_image_name = os.path.join(output_folder, f"output_{idx}_{input_img_base_counter}")
        input_img_base_counter +=1
        
        # Set the object name in your configuration dictionary
        config['obj_name'] = output_image_name

        
        config['input_image_path'] = image_path
        config["delta_azimuth_end"]= azimuth
        config["delta_azimuth_start"]= azimuth
        config["delta_elevation_end"]= elevation
        config["delta_elevation_start"]= elevation
        config["delta_radius_end"] = radius
        config["delta_radius_start"] = radius

        # Use the extracted pose values as needed (for example, printing or passing them to a function)
        print(
            f"Processing image {idx + 1}/{len(demo_data_image[2])} "
            f"with azimuth: {azimuth}, elevation: {elevation}, radius: {radius}..."
        )

        # Call your prediction function here using the updated `config`
        # predict_function(config)  # Example: Uncomment and adjust accordingly
        try:
            generation_vivid123_new(config=config, vivid123_pipe=vivid123_pipe, xl_pipe=xl_pipe)
        except Exception as e:
            print(e)
            # print("Failed for ",output_image_name)

print(f"All outputs saved to {output_folder}")