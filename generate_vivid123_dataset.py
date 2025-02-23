import h5py
import numpy as np
import pyvista as pv
from tqdm import tqdm  # Import tqdm for progress bars
import os
from PIL import Image
import csv
from vivid123.generation_utils import generation_vivid123_new, prepare_vivid123_pipeline
import torch
from carvekit.api.high import HiInterface

# Explicitly set PyVista backend to standalone
pv.global_theme.jupyter_backend = 'static'

# Path to the dataset
dataset_path = "demo_feb_7.hdf5"
f_org = h5py.File(dataset_path, "r")

# Get the list of demos
demos = list(f_org["data"]["demo_4"]["obs"].keys())
print("Available demos:", demos)

demo_data_image = np.array(f_org["data"]["demo_4"]["obs"]['eye_in_hand_rgb'])

print(len(demo_data_image))
demo_data_end_effector = np.array(f_org["data"]["demo_4"]["obs"]['ee_states'])


print(len(demo_data_end_effector))


ZERO123_MODEL_ID = "bennyguo/zero123-xl-diffusers"
VIDEO_MODEL_ID = "cerspense/zeroscope_v2_576w"
VIDEO_XL_MODEL_ID = "cerspense/zeroscope_v2_XL"
vivid123_pipe, xl_pipe = prepare_vivid123_pipeline(
        ZERO123_MODEL_ID=ZERO123_MODEL_ID, 
        VIDEO_MODEL_ID=VIDEO_MODEL_ID, 
        VIDEO_XL_MODEL_ID=VIDEO_XL_MODEL_ID
    )

# this is for removing background
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
    "generation_type": None
}


root_directory = "vivid123_dataset_for_detection/Eggplant_and_capsicum_data/"
output_folder = "synthesized/"
input_temp = "input_images/"

azimuth_ranges = [0.0, 360.0]
elevation_ranges = [0.0, 5.0]
radius_ranges = [1.0, 5.0]
sample_number= 10


config['obj_name'] = os.path.join(root_directory, output_folder)
os.makedirs(config["obj_name"], exist_ok=True)

input_image_dir = os.path.join(root_directory, input_temp)
os.makedirs(input_image_dir, exist_ok=True)


for idx, image_data in enumerate(demo_data_image):

    image_path = os.path.join(input_image_dir, f"image_{idx}.png")
    # Check if the image is in BGR order (if shape[-1] equals 3) and convert it to RGB if needed
    if image_data.shape[-1] == 3:
        image_data = image_data[..., ::-1]  # Reverse the channel order (BGR -> RGB)
    
    # Save the image to the temporary path
    image = Image.fromarray(image_data)
    image.save(image_path)
    images_without_background = interface([image_path])
    cat_wo_bg = images_without_background[0]
    cat_wo_bg.save(image_path)
    config['input_image_path'] = image_path
    print(
            f"Processing image {idx + 1}/{len(demo_data_image)} "
        )
    
    # generating data by changing azimuth
    config["delta_elevation_end"]= elevation_ranges[0]
    config["delta_elevation_start"]= elevation_ranges[0]
    config["delta_radius_end"] = radius_ranges[0]
    config["delta_radius_start"] = radius_ranges[0]
    config["delta_azimuth_end"]= azimuth_ranges[1]
    config["delta_azimuth_start"]= azimuth_ranges[0]
    config['num_frames'] = sample_number
    config['generation_type']="a_"+f"image_{idx}_"
    try:
        generation_vivid123_new(config=config, vivid123_pipe=vivid123_pipe, xl_pipe=xl_pipe)
    except Exception as e:
        print(e)

    # generating data by changing elevation
    config["delta_radius_end"] = radius_ranges[0]
    config["delta_radius_start"] = radius_ranges[0]
    config["delta_azimuth_end"]= azimuth_ranges[0]
    config["delta_azimuth_start"]= azimuth_ranges[0]
    config["delta_elevation_end"]= elevation_ranges[1]
    config["delta_elevation_start"]= elevation_ranges[0]
    config['generation_type']="e_"+f"image_{idx}_"
    try:
        generation_vivid123_new(config=config, vivid123_pipe=vivid123_pipe, xl_pipe=xl_pipe)
    except Exception as e:
        print(e)
    
    # generating data by changing radius
    config["delta_azimuth_end"]= azimuth_ranges[0]
    config["delta_azimuth_start"]= azimuth_ranges[0]
    config["delta_elevation_end"]= elevation_ranges[0]
    config["delta_elevation_start"]= elevation_ranges[0]
    config["delta_radius_end"] = radius_ranges[1]
    config["delta_radius_start"] = radius_ranges[0]
    config['generation_type']="r_"+f"image_{idx}_"
    try:
        generation_vivid123_new(config=config, vivid123_pipe=vivid123_pipe, xl_pipe=xl_pipe)
    except Exception as e:
        print(e)

    

print(f"All outputs saved to {config['obj_name']}")