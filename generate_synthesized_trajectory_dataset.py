import os
import random
import h5py
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from carvekit.api.high import HiInterface
from vivid123.generation_utils import generation_vivid123_new, prepare_vivid123_pipeline

########################################
# Helper Functions
########################################

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x**2 + y**2))
    return r, az, el

def sph2cart(r, az, el):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def make_cube(cx, cy, cz, rotation, side=0.005):
    half = side / 2.0
    corners = np.array([
        [-half, -half, -half],
        [-half, -half,  half],
        [-half,  half, -half],
        [-half,  half,  half],
        [ half, -half, -half],
        [ half, -half,  half],
        [ half,  half, -half],
        [ half,  half,  half]
    ])
    rotated = rotation.apply(corners)
    return rotated + np.array([cx, cy, cz])

def unit_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else np.array([1,0,0])

def generate_candidate_chain(positions, rpy):
    """
    Given positions (N,3) and rpy (N,3) from joint_states, generate a candidate chain.
    We compute a global reference (from first point to goal) and then, for each frame,
    sample candidate viewpoints (via spherical perturbations) and choose one candidate per frame.
    """
    n_points = len(positions)
    goal_pos = positions[-1]
    dx_ref = positions[0][0] - goal_pos[0]
    dy_ref = positions[0][1] - goal_pos[1]
    dz_ref = positions[0][2] - goal_pos[2]
    ref_r, ref_az, ref_el = cart2sph(dx_ref, dy_ref, dz_ref)
    
    # Variation parameters (you can adjust these)
    base_max_deg = 5.0
    base_max_side = 0.1
    base_max_roll = 50 * np.pi/180.0
    base_max_pitch = 10 * np.pi/180.0
    base_max_yaw = 10 * np.pi/180.0
    lambda_decay = 1.5
    num_samples = 10

    dataset = []
    for i in range(n_points):
        x, y, z = positions[i]
        roll_i, pitch_i, yaw_i = rpy[i]
        dx = x - goal_pos[0]
        dy = y - goal_pos[1]
        dz = z - goal_pos[2]
        r_val, az, el = cart2sph(dx, dy, dz)
        droll = roll_i - rpy[-1][0]
        dpitch = pitch_i - rpy[-1][1]
        dyaw = yaw_i - rpy[-1][2]
        for sample in range(num_samples):
            frac = np.exp(-lambda_decay * (i/(n_points-1))) if n_points > 1 else 1.0
            delta_angle = frac * (base_max_deg * np.pi/180.0)
            az_var = az + random.choice([-1,1]) * delta_angle
            el_var = el + random.choice([-1,1]) * delta_angle
            dx_var, dy_var, dz_var = sph2cart(r_val, az_var, el_var)
            x_new = goal_pos[0] + dx_var
            y_new = goal_pos[1] + dy_var
            z_new = goal_pos[2] + dz_var
            
            roll_off = frac * base_max_roll
            pitch_off = frac * base_max_pitch
            yaw_off = frac * base_max_yaw
            droll_var = droll + random.choice([-1,1]) * roll_off
            dpitch_var = dpitch + random.choice([-1,1]) * pitch_off
            dyaw_var = dyaw + random.choice([-1,1]) * yaw_off
            roll_new = rpy[-1][0] + droll_var
            pitch_new = rpy[-1][1] + dpitch_var
            yaw_new = rpy[-1][2] + dyaw_var
            
            rot_obj = R.from_euler('zyx', [roll_new, pitch_new, yaw_new])
            side_len = frac * base_max_side
            if side_len < 1e-9:
                continue
            corners = make_cube(x_new, y_new, z_new, rot_obj, side_len)
            r_center, az_center, el_center = cart2sph(x_new - goal_pos[0],
                                                      y_new - goal_pos[1],
                                                      z_new - goal_pos[2])
            delta_az = az_center - ref_az
            delta_el = el_center - ref_el
            delta_r = r_center - ref_r
            quat = rot_obj.as_quat()
            for corner in corners:
                dataset.append({
                    "trajectory_idx": i,
                    "pose_6d": [corner[0], corner[1], corner[2],
                                roll_new, pitch_new, yaw_new],
                    "delta_pose": [delta_az, delta_el, delta_r],
                    "in_fov": True
                })
    # Greedy chain: pick (for each frame) the first candidate marked in view.
    chain = []
    for i in range(n_points):
        group = [vp for vp in dataset if vp["trajectory_idx"] == i and vp["in_fov"]]
        if group:
            chain.append(group[0])
    return chain

########################################
# Main Processing: Read All Demos, Synthesize, and Save New HDF5
########################################

def process_and_save_synthesized_demos(input_hdf5="demo_duck_feb12.hdf5",
                                       output_hdf5="synthesized_trajectories.hdf5"):
    # Open the output file for writing.
    with h5py.File(output_hdf5, "w") as fout:
        data_grp = fout.create_group("data")
        # Open the input file.
        with h5py.File(input_hdf5, "r") as fin:
            demos = list(fin["data"].keys())
            print("Found demos in input:", demos)
            
            # Prepare vivid123 pipeline.
            ZERO123_MODEL_ID = "bennyguo/zero123-xl-diffusers"
            VIDEO_MODEL_ID = "cerspense/zeroscope_v2_576w"
            VIDEO_XL_MODEL_ID = "cerspense/zeroscope_v2_XL"
            vivid123_pipe, xl_pipe = prepare_vivid123_pipeline(
                ZERO123_MODEL_ID=ZERO123_MODEL_ID,
                VIDEO_MODEL_ID=VIDEO_MODEL_ID,
                VIDEO_XL_MODEL_ID=VIDEO_XL_MODEL_ID
            )
            # Prepare background removal interface.
            bg_interface = HiInterface(
                object_type="hairs-like",
                batch_size_seg=5,
                batch_size_matting=1,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                seg_mask_size=640,
                matting_mask_size=2048,
                trimap_prob_threshold=231,
                trimap_dilation=30,
                trimap_erosion_iters=5,
                fp16=False
            )
            
            # For each demo in the input file.
            for demo_idx, demo_key in enumerate(demos):
                print(f"\nProcessing demo {demo_key} as synthesized demo_{demo_idx}")
                demo_in = fin["data"][demo_key]
                obs_in = demo_in["obs"]
                # Read joint_states from original demo.
                joint_states = np.array(obs_in["joint_states"])  # shape (N,7)
                positions = joint_states[:, :3]
                rpy = joint_states[:, 3:6]
                # Generate candidate chain (trajectory) for this demo.
                candidate_chain = generate_candidate_chain(positions, rpy)
                if candidate_chain is None or len(candidate_chain) == 0:
                    print(f"No candidate chain generated for demo {demo_key}, skipping.")
                    continue

                # Prepare the reference image: take the first frame from "eye_in_hand_rgb"
                ref_img_data = np.array(obs_in["eye_in_hand_rgb"][0])
                # If necessary, convert from BGR to RGB.
                if ref_img_data.shape[-1] == 3:
                    ref_img_data = ref_img_data[..., ::-1]
                ref_img = Image.fromarray(ref_img_data)
                temp_ref_dir = "temp_ref_demos"
                os.makedirs(temp_ref_dir, exist_ok=True)
                ref_img_path = os.path.join(temp_ref_dir, f"demo_{demo_idx}_ref.png")
                ref_img.save(ref_img_path)
                # Run background removal on reference image.
                processed = bg_interface([ref_img_path])
                processed[0].save(ref_img_path)
                print(f"Reference image for demo {demo_key} processed and saved to {ref_img_path}")
                
                # Set up configuration for vivid123 synthesis.
                synth_config = {
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
                    "width": 256,
                    "input_image_path": ref_img_path,
                    "obj_name": "",  # will be set per candidate
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
                    "zero123_end_step_percentage": 1.0,
                    "zero123_linear_end_weight": 1.0,
                    "zero123_linear_start_weight": 1.0,
                    "zero123_start_step_percentage": 0.0,
                    "generation_type": "image"
                }
                
                # For each candidate pose in the chain, synthesize an image.
                synthesized_imgs = []
                new_joint_states = []  # We'll create (N,7); append a dummy value for 7th column.
                temp_synth_dir = os.path.join("temp_synth", f"demo_{demo_idx}")
                os.makedirs(temp_synth_dir, exist_ok=True)
                
                for i, cand in enumerate(candidate_chain):
                    delta_pose = cand.get("delta_pose", [0.0, 0.0, 0.0])
                    delta_az, delta_el, delta_r = delta_pose
                    synth_config["delta_azimuth_start"] = delta_az
                    synth_config["delta_azimuth_end"] = delta_az
                    synth_config["delta_elevation_start"] = delta_el
                    synth_config["delta_elevation_end"] = delta_el
                    synth_config["delta_radius_start"] = delta_r
                    synth_config["delta_radius_end"] = delta_r
                    # Create an output directory for this candidate.
                    out_dir = os.path.join(temp_synth_dir, f"point_{i}")
                    os.makedirs(out_dir, exist_ok=True)
                    synth_config["obj_name"] = out_dir
                    print(f"Demo {demo_idx} - Synthesizing point {i}: Δaz={delta_az:.3f}, Δel={delta_el:.3f}, Δr={delta_r:.3f}")
                    try:
                        generation_vivid123_new(config=synth_config,
                                                vivid123_pipe=vivid123_pipe,
                                                xl_pipe=xl_pipe)
                        synth_img_path = os.path.join(out_dir, "image000.png")
                        if os.path.exists(synth_img_path):
                            img = Image.open(synth_img_path)
                            img_arr = np.array(img)
                            synthesized_imgs.append(img_arr)
                            # Use candidate's pose_6d and add a dummy zero to create 7 columns.
                            pose = cand["pose_6d"] + [0.0]
                            new_joint_states.append(pose)
                        else:
                            print(f"Synthesized image not found for demo {demo_idx} point {i}")
                    except Exception as e:
                        print(f"Error synthesizing for demo {demo_idx} point {i}: {e}")
                
                if len(synthesized_imgs) == 0 or len(new_joint_states) == 0:
                    print(f"No synthesized data for demo {demo_idx}, skipping demo.")
                    continue
                
                # Convert lists to numpy arrays.
                synthesized_imgs_arr = np.stack(synthesized_imgs, axis=0)  # shape (N, H, W, 3)
                new_joint_states_arr = np.array(new_joint_states)          # shape (N, 7)
                
                # Create a new demo group in output file.
                demo_out_grp = data_grp.create_group(f"demo_{demo_idx}")
                obs_out = demo_out_grp.create_group("obs")
                obs_out.create_dataset("eye_in_hand_rgb", data=synthesized_imgs_arr, compression="gzip")
                obs_out.create_dataset("joint_states", data=new_joint_states_arr, compression="gzip")
                print(f"Written synthesized demo demo_{demo_idx} with {synthesized_imgs_arr.shape[0]} frames.")
                
        print("All demos processed.")
    print("Output saved to", output_hdf5)

if __name__ == "__main__":
    process_and_save_synthesized_demos(input_hdf5="/home/carl_ma/Riad/diffusion_policy/dataset/duck_merged.hdf5")
