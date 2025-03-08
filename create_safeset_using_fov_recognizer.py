import os
import random
import numpy as np
import cv2
import h5py
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import shutil

############################################
# Remove temporary folders at the start
############################################
temp_ref_folder = "temp_ref"
if os.path.exists(temp_ref_folder):
    shutil.rmtree(temp_ref_folder)

temp_syn_folder = "temp_synth"
if os.path.exists(temp_syn_folder):
    shutil.rmtree(temp_syn_folder)

############################################
# 1. White Background Helper
############################################
def add_white_background(im):
    """
    Composites an image with transparency onto a white background;
    otherwise, just converts to RGB.
    """
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    else:
        return im.convert("RGB")

############################################
# 2. DH Transform, Forward Kinematics and Pose Conversion
############################################
def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha),  a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha),  a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),                d              ],
        [0,              0,                            0,                            1              ]
    ], dtype=float)

def franka_forward_kinematics_panda_hand(joint_angles):
    q1, q2, q3, q4, q5, q6, q7 = joint_angles
    T1 = dh_transform(a=0,       alpha=0,         d=0.333,   theta=q1)
    T2 = dh_transform(a=0,       alpha=-np.pi/2,  d=0,       theta=q2)
    T3 = dh_transform(a=0,       alpha=np.pi/2,   d=0.316,   theta=q3)
    T4 = dh_transform(a=0.0825,  alpha=np.pi/2,   d=0,       theta=q4)
    T5 = dh_transform(a=-0.0825, alpha=-np.pi/2,  d=0.384,   theta=q5)
    T6 = dh_transform(a=0,       alpha=np.pi/2,   d=0,       theta=q6)
    T7 = dh_transform(a=0.088,   alpha=np.pi/2,   d=0,       theta=q7)
    T_flange = dh_transform(a=0, alpha=0, d=0.107, theta=0)
    Rz_neg_45 = dh_transform(a=0, alpha=0, d=0, theta=-np.pi/4)
    T_panda_hand = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ T_flange @ Rz_neg_45
    return T_panda_hand

def joint_angles_to_end_effector_pose(joint_angles):
    T = franka_forward_kinematics_panda_hand(joint_angles)
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    rotation_matrix = T[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # [qx, qy, qz, qw]
    return np.array([x, y, z, quat[0], quat[1], quat[2], quat[3]], dtype=float)

############################################
# 3. Cartesian <-> Spherical Conversions and Cube Creation
############################################
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
    rotated_corners = rotation.apply(corners)
    rotated_corners += np.array([cx, cy, cz])
    return rotated_corners

############################################
# 4. FoV Checker and Image Generation (unchanged from FOV-only code)
############################################
def generate_image_frame(img_width, img_height, point_coord, dot_dia=10):
    image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    cv2.circle(image, point_coord, dot_dia//2, (255,255,255), -1)
    return image

def generate_camera_image(camera_pose, camera_fov, point_obj,
                          img_width=640, img_height=480, dot_dia=10):
    cam_pos = np.array(camera_pose[:3])
    cam_quat = camera_pose[3:]
    cam_rot = R.from_quat(cam_quat).as_matrix()
    point_in_cam = np.linalg.inv(cam_rot) @ (np.array(point_obj) - cam_pos)
    h_fov, v_fov = np.radians(camera_fov)
    x_cam, y_cam, z_cam = point_in_cam
    h_angle = np.arctan2(y_cam, x_cam)
    v_angle = np.arctan2(z_cam, x_cam)
    img_x = int((h_angle + h_fov/2) / h_fov * img_width)
    img_y = int((v_angle + v_fov/2) / v_fov * img_height)
    return generate_image_frame(img_width, img_height, (img_x, img_y), dot_dia)

def is_inFOV(camera_pose, camera_fov, point_obj):
    cam_pos = np.array(camera_pose[:3])
    # Shortcut as in FOV-only code:
    if np.linalg.norm(np.array(point_obj) - cam_pos) < 1e-4:
        return True
    cam_quat = camera_pose[3:]
    cam_rot = R.from_quat(cam_quat).as_matrix()
    point_in_cam = np.linalg.inv(cam_rot) @ (np.array(point_obj) - cam_pos)
    h_fov, v_fov = np.radians(camera_fov)
    x_cam, y_cam, z_cam = point_in_cam
    if x_cam <= 0:
        return False
    h_angle = np.arctan2(y_cam, x_cam)
    v_angle = np.arctan2(z_cam, x_cam)
    return (abs(h_angle) <= h_fov/2) and (abs(v_angle) <= v_fov/2)

############################################
# 5. Vivid123 Synthesis Function (for FOV+Recog pipeline)
############################################
from vivid123.generation_utils import generation_vivid123_new, prepare_vivid123_pipeline
def synthesize_image_for_point(candidate, vivid123_pipe, xl_pipe, base_config, demo_num, traj_pt, sample):
    delta_pose = candidate.get("delta_pose", [0.0, 0.0, 0.0])
    delta_az, delta_el, delta_r = delta_pose
    config = base_config.copy()
    config["delta_azimuth_start"] = delta_az
    config["delta_azimuth_end"]   = delta_az
    config["delta_elevation_start"] = delta_el
    config["delta_elevation_end"]   = delta_el
    config["delta_radius_start"]    = delta_r
    config["delta_radius_end"]      = delta_r
    config["candidate_pose"] = candidate.get("pose_7d")
    
    out_dir = os.path.join("temp_synth", f"demo_{demo_num}_pt_{traj_pt}_sample_{sample}")
    os.makedirs(out_dir, exist_ok=True)
    config["obj_name"] = out_dir

    print(f"Generating synthesized image for demo {demo_num}, trajectory point {traj_pt}, sample {sample}...")
    generation_vivid123_new(config=config, vivid123_pipe=vivid123_pipe, xl_pipe=xl_pipe)
    img_path = os.path.join(out_dir, "image000.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Synthesized image not found in {out_dir}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read synthesized image from {img_path}")
    img = cv2.resize(img, (320, 240))
    print(f"Synthesized image generated at {img_path}")
    return img

############################################
# 6. Object Detector Functions
############################################
def extract_feature(image_tensor, vision_encoder, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        conv_out = vision_encoder(image_tensor)
        if conv_out.dim() == 4:
            feature = torch.nn.functional.adaptive_avg_pool2d(conv_out, (1, 1)).view(1, -1)
        elif conv_out.dim() == 2:
            feature = conv_out
        else:
            raise ValueError("Unexpected output dimensions from vision encoder.")
    return feature.cpu().numpy()[0]

def is_object_visible(cv2_img, vision_encoder, device, knn_classifier):
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    image_tensor = transform(pil_img)
    feature = extract_feature(image_tensor, vision_encoder, device)
    pred = knn_classifier.predict([feature])[0]
    return (pred == 1)

############################################
# 7. Background Removal using HiInterface
############################################
from carvekit.api.high import HiInterface

############################################
# 8a. Candidate Generation for FOV-Only Safe Set
############################################
def generate_candidates_for_demo_fov_only(demo_poses, num_samples=5):
    n_points = demo_poses.shape[0]
    deg_to_rad = np.pi / 180.0
    max_deg = 5.0            # degrees
    max_side = 0.1
    max_roll = 50 * deg_to_rad
    max_pitch = 10 * deg_to_rad
    max_yaw = 10 * deg_to_rad
    lambda_decay = 1.5
    camera_fov = (80, 80)
    
    goal_pose = demo_poses[-1]
    goal_pos = goal_pose[:3]
    print(f"FOV-only: Goal position: {goal_pos}")
    
    dataset = []
    total_candidates = 0
    
    for i in range(n_points):
        demo_pose = demo_poses[i]  # 7D demonstration pose
        x, y, z = demo_pose[:3]
        dx = x - goal_pos[0]
        dy = y - goal_pos[1]
        dz = z - goal_pos[2]
        r_val, az, el = cart2sph(dx, dy, dz)
        for sample in range(num_samples):
            frac = np.exp(-lambda_decay * (i / (n_points - 1))) if n_points > 1 else 1.0
            delta_angle = frac * (max_deg * deg_to_rad)
            az_var = az + random.choice([-1, 1]) * delta_angle
            el_var = el + random.choice([-1, 1]) * delta_angle
            dx_var, dy_var, dz_var = sph2cart(r_val, az_var, el_var)
            x_new = goal_pos[0] + dx_var
            y_new = goal_pos[1] + dy_var
            z_new = goal_pos[2] + dz_var
            
            roll_off = frac * max_roll
            pitch_off = frac * max_pitch
            yaw_off = frac * max_yaw
            droll = random.choice([-1, 1]) * roll_off
            dpitch = random.choice([-1, 1]) * pitch_off
            dyaw = random.choice([-1, 1]) * yaw_off
            rot_obj = R.from_euler('zyx', [droll, dpitch, dyaw])
            quat = rot_obj.as_quat()
            quat = quat / np.linalg.norm(quat)
            
            valid_candidate_poses = []
            cube_candidates = make_cube(x_new, y_new, z_new, rot_obj, side=frac * max_side)
            for corner in cube_candidates:
                total_candidates += 1
                candidate_pose = [corner[0], corner[1], corner[2],
                                  quat[0], quat[1], quat[2], quat[3]]
                if is_inFOV(candidate_pose, camera_fov, goal_pos):
                    valid_candidate_poses.append(candidate_pose)
            
            if not valid_candidate_poses:
                candidate_center = [x_new, y_new, z_new, quat[0], quat[1], quat[2], quat[3]]
                total_candidates += 1
                if is_inFOV(candidate_center, camera_fov, goal_pos):
                    valid_candidate_poses.append(candidate_center)
            if not valid_candidate_poses:
                print(f"FOV-only: Using demonstration pose from traj {i} as fallback.")
                valid_candidate_poses.append(demo_pose.tolist())
            
            for pose in valid_candidate_poses:
                candidate = {
                    "pose_7d": pose,
                    "trajectory_idx": i,
                    "in_fov": True
                }
                dx_delta = x_new - x
                dy_delta = y_new - y
                dz_delta = z_new - z
                r_delta, az_delta, el_delta = cart2sph(dx_delta, dy_delta, dz_delta)
                candidate["delta_pose"] = [az_delta, el_delta, r_delta]
                dataset.append(candidate)
    print(f"FOV-only: Total candidates generated: {total_candidates}")
    return dataset

############################################
# 8b. Candidate Generation for FOV+Recog Safe Set
############################################
def generate_candidates_for_demo_fov_recog(demo_poses, num_samples, vision_encoder, device, knn_classifier, 
                                           vivid123_pipe, xl_pipe, base_config, bg_interface):
    n_points = demo_poses.shape[0]
    deg_to_rad = np.pi / 180.0
    max_deg = 5.0            # degrees
    max_side = 0.1
    max_roll = 50 * deg_to_rad
    max_pitch = 10 * deg_to_rad
    max_yaw = 10 * deg_to_rad
    lambda_decay = 1.5
    camera_fov = (80, 80)
    
    goal_pose = demo_poses[-1]
    goal_pos = goal_pose[:3]
    print(f"FOV+Recog: Goal position: {goal_pos}")
    
    dataset = []
    total_candidates = 0
    valid_candidates = 0
    
    for i in range(n_points):
        demo_pose = demo_poses[i]
        x, y, z = demo_pose[:3]
        dx = x - goal_pos[0]
        dy = y - goal_pos[1]
        dz = z - goal_pos[2]
        r_val, az, el = cart2sph(dx, dy, dz)
        for sample in range(num_samples):
            frac = np.exp(-lambda_decay * (i / (n_points - 1))) if n_points > 1 else 1.0
            delta_angle = frac * (max_deg * deg_to_rad)
            az_var = az + random.choice([-1, 1]) * delta_angle
            el_var = el + random.choice([-1, 1]) * delta_angle
            dx_var, dy_var, dz_var = sph2cart(r_val, az_var, el_var)
            x_new = goal_pos[0] + dx_var
            y_new = goal_pos[1] + dy_var
            z_new = goal_pos[2] + dz_var
            
            roll_off = frac * max_roll
            pitch_off = frac * max_pitch
            yaw_off = frac * max_yaw
            droll = random.choice([-1, 1]) * roll_off
            dpitch = random.choice([-1, 1]) * pitch_off
            dyaw = random.choice([-1, 1]) * yaw_off
            rot_obj = R.from_euler('zyx', [droll, dpitch, dyaw])
            quat = rot_obj.as_quat()
            quat = quat / np.linalg.norm(quat)
            
            valid_candidate_poses = []
            cube_candidates = make_cube(x_new, y_new, z_new, rot_obj, side=frac * max_side)
            for corner in cube_candidates:
                total_candidates += 1
                candidate_pose = [corner[0], corner[1], corner[2],
                                  quat[0], quat[1], quat[2], quat[3]]
                if is_inFOV(candidate_pose, camera_fov, goal_pos):
                    valid_candidate_poses.append(candidate_pose)
            if not valid_candidate_poses:
                candidate_center = [x_new, y_new, z_new, quat[0], quat[1], quat[2], quat[3]]
                total_candidates += 1
                if is_inFOV(candidate_center, camera_fov, goal_pos):
                    print(f"FOV+Recog: Fallback candidate (center) from traj {i}, sample {sample} accepted by FOV.")
                    valid_candidate_poses.append(candidate_center)
                # else:
                    # print(f"FOV+Recog: Fallback candidate (center) from traj {i}, sample {sample} failed FOV.")
            if not valid_candidate_poses:
                print(f"FOV+Recog: Using demonstration pose from traj {i} as fallback candidate.")
                valid_candidate_poses.append(demo_pose.tolist())
            
            for pose in valid_candidate_poses:
                candidate = {
                    "pose_7d": pose,
                    "trajectory_idx": i,
                    "in_fov": True
                }
                dx_delta = x_new - x
                dy_delta = y_new - y
                dz_delta = z_new - z
                r_delta, az_delta, el_delta = cart2sph(dx_delta, dy_delta, dz_delta)
                candidate["delta_pose"] = [az_delta, el_delta, r_delta]
                
                try:
                    print(f"FOV+Recog: Synthesizing image for candidate from traj {i}, sample {sample}...")
                    img = synthesize_image_for_point(candidate, vivid123_pipe, xl_pipe, base_config, demo_num=0, traj_pt=i, sample=sample)
                except Exception as e:
                    print(f"FOV+Recog: Error synthesizing image for candidate (traj {i}, sample {sample}): {e}")
                    continue
                
                temp_img_path = os.path.join("temp_synth", f"demo_0_pt_{i}_sample_{sample}_temp.png")
                cv2.imwrite(temp_img_path, img)
                processed_img = bg_interface([temp_img_path])[0]
                processed_img = add_white_background(processed_img)
                img_processed = cv2.cvtColor(np.array(processed_img), cv2.COLOR_RGB2BGR)
                
                if is_object_visible(img_processed, vision_encoder, device, knn_classifier):
                    valid_candidates += 1
                    candidate["image"] = img_processed
                    dataset.append(candidate)
                    print(f"FOV+Recog: Candidate at traj {i}, sample {sample} accepted by object detector.")
                else:
                    print(f"FOV+Recog: Candidate at traj {i}, sample {sample} rejected by object detector.")
    print(f"FOV+Recog: Total candidates generated: {total_candidates}")
    print(f"FOV+Recog: Candidates passed both FOV and object detection: {valid_candidates}")
    return dataset

############################################
# 9. Load Vision Encoder Function
############################################
def get_modified_resnet(name: str, weights=None):
    import torchvision.models as models
    import torch.nn as nn
    resnet = getattr(models, name)(weights=weights)
    for n, module in list(resnet.named_modules()):
        if isinstance(module, nn.BatchNorm2d):
            setattr(resnet, n, nn.GroupNorm(num_groups=32, num_channels=module.num_features))
    resnet.avgpool = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=2),
        torch.nn.Softmax(dim=2),
        torch.nn.Flatten(start_dim=1)
    )
    resnet.fc = torch.nn.Identity()
    return resnet

def load_vision_encoder(device):
    print("Loading vision encoder...")
    vision_encoder = get_modified_resnet('resnet18', weights=None)
    ckpt_path = "bg_removed_train_dp/after_train_500_epochs.ckpt"
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vision_encoder."):
                new_state_dict[k[len("vision_encoder."):]] = v
            else:
                new_state_dict[k] = v
        vision_encoder.load_state_dict(new_state_dict, strict=False)
        print("Loaded trained vision encoder weights successfully.")
    else:
        print("Checkpoint not found. Using default weights.")
    vision_encoder.to(device)
    vision_encoder.eval()
    for param in vision_encoder.parameters():
        param.requires_grad = False
    return vision_encoder

############################################
# 10. Main Execution: Generate and Save Two Safe Sets
############################################
def main():
    print("Starting safe set generation process...")
    # Updated input file
    input_file = "duck_optimal_30_bg_removed.hdf5"
    in_h5 = h5py.File(input_file, "r")
    demo_keys = list(in_h5["data"].keys())
    print("Available Demos:", demo_keys)
    
    # Load vision encoder and kNN classifier for the FOV+Recog pipeline.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_encoder = load_vision_encoder(device)
    with open("train_dp_base_optimal_30_duck/knn_model_base_optimal_duck_30.pkl", "rb") as f:
        knn_classifier = pickle.load(f)
    print("Vision encoder and kNN classifier loaded.")
    
    # Set up vivid123 pipeline.
    ZERO123_MODEL_ID = "bennyguo/zero123-xl-diffusers"
    vivid123_pipe, xl_pipe = prepare_vivid123_pipeline(
        ZERO123_MODEL_ID=ZERO123_MODEL_ID,
        VIDEO_MODEL_ID="cerspense/zeroscope_v2_576w",
        VIDEO_XL_MODEL_ID="cerspense/zeroscope_v2_XL"
    )
    print("Vivid123 pipeline initialized.")
    
    # Base configuration for vivid123 synthesis.
    base_config = {
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
        "input_image_path": "",  # To be set per demo after BG removal.
        "obj_name": "dummy",
        "noise_identical_accross_frames": False,
        "num_frames": 1,
        "num_inference_steps": 50,  # Updated number of inference steps.
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
    
    all_candidates_fov_only = []
    all_candidates_fov_recog = []
    
    for demo_name in demo_keys:
        print(f"\nProcessing Demo: {demo_name}")
        demo_data = in_h5["data"][demo_name]
        obs_group = demo_data["obs"]
        if "joint_states" not in obs_group:
            print(f"Demo {demo_name} has no joint_states. Skipping demo.")
            continue
        joint_states = obs_group["joint_states"][:]  # shape (T,7)
        print(f"Number of joint states in demo {demo_name}: {joint_states.shape[0]}")
        demo_poses = []
        for js in joint_states:
            demo_pose = joint_angles_to_end_effector_pose(js)
            demo_poses.append(demo_pose)
        demo_poses = np.array(demo_poses)
        
        # Process the reference image for BG removal.
        if "eye_in_hand_rgb_bg_removed" in obs_group:
            ref_img_data = np.array(obs_group["eye_in_hand_rgb_bg_removed"][0])
            if ref_img_data.shape[-1] == 3:
                ref_img_data = ref_img_data[..., ::-1]
            ref_img = Image.fromarray(ref_img_data)
            os.makedirs(temp_ref_folder, exist_ok=True)
            ref_img_path = os.path.join(temp_ref_folder, f"{demo_name}_ref.png")
            ref_img.save(ref_img_path)
            print(f"Processing reference image for demo {demo_name}...")
            bg_interface_ref = HiInterface(
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
            processed_ref = bg_interface_ref([ref_img_path])[0]
            processed_ref.save(ref_img_path)
            print(f"Reference image processed and saved at {ref_img_path}")
            base_config["input_image_path"] = ref_img_path
        else:
            print(f"Demo {demo_name} has no reference image.")
        
        # Create a BG removal interface for synthesized images.
        bg_interface_synth = HiInterface(
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
        
        # Generate candidates for FOV-only safe set.
        candidates_fov_only = generate_candidates_for_demo_fov_only(demo_poses, num_samples=5)
        if candidates_fov_only:
            print(f"Demo {demo_name} (FOV-only): {len(candidates_fov_only)} candidate viewpoints passed.")
            all_candidates_fov_only.extend(candidates_fov_only)
        else:
            print(f"Demo {demo_name} (FOV-only): No candidate viewpoints passed.")
        
        # Generate candidates for FOV+Recognition safe set.
        candidates_fov_recog = generate_candidates_for_demo_fov_recog(demo_poses, num_samples=5,
                                                                      vision_encoder=vision_encoder,
                                                                      device=device,
                                                                      knn_classifier=knn_classifier,
                                                                      vivid123_pipe=vivid123_pipe,
                                                                      xl_pipe=xl_pipe,
                                                                      base_config=base_config,
                                                                      bg_interface=bg_interface_synth)
        if candidates_fov_recog:
            print(f"Demo {demo_name} (FOV+Recog): {len(candidates_fov_recog)} candidate viewpoints passed.")
            all_candidates_fov_recog.extend(candidates_fov_recog)
        else:
            print(f"Demo {demo_name} (FOV+Recog): No candidate viewpoints passed.")
    in_h5.close()
    
    # Compute safe set from FOV-only candidates (using 3D positions).
    if all_candidates_fov_only:
        safe_set_fov_only = np.array([cand["pose_7d"][:3] for cand in all_candidates_fov_only])
        print(f"Total valid candidate viewpoints (FOV-only): {len(safe_set_fov_only)}")
        try:
            hull_fov = ConvexHull(safe_set_fov_only, qhull_options="QJ")
            np.savez("safe_set_6d_fov.npz", safe_set=safe_set_fov_only, hull_equations=hull_fov.equations, hull_vertices=hull_fov.vertices)
            print("FOV-only safe set with convex hull saved to 'safe_set_6d_fov.npz'.")
        except Exception as e:
            print(f"Error computing convex hull for FOV-only safe set: {e}. Saving safe set without hull.")
            np.savez("safe_set_6d_fov.npz", safe_set=safe_set_fov_only)
    else:
        print("No valid FOV-only candidate viewpoints found.")
    
    # Compute safe set from FOV+Recognition candidates.
    if all_candidates_fov_recog:
        safe_set_fov_recog = np.array([cand["pose_7d"][:3] for cand in all_candidates_fov_recog])
        print(f"Total valid candidate viewpoints (FOV+Recog): {len(safe_set_fov_recog)}")
        try:
            hull_recog = ConvexHull(safe_set_fov_recog, qhull_options="QJ")
            np.savez("safe_set_6d_fv_rg.npz", safe_set=safe_set_fov_recog, hull_equations=hull_recog.equations, hull_vertices=hull_recog.vertices)
            print("FOV+Recog safe set with convex hull saved to 'safe_set_6d_fv_rg.npz'.")
        except Exception as e:
            print(f"Error computing convex hull for FOV+Recog safe set: {e}. Saving safe set without hull.")
            np.savez("safe_set_6d_fv_rg.npz", safe_set=safe_set_fov_recog)
    else:
        print("No valid FOV+Recog candidate viewpoints found.")
    
    print("Safe set generation process completed.")

if __name__ == "__main__":
    print("Initializing models and pipelines...")
    main()
