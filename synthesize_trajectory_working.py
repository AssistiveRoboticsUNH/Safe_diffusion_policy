import os
import random
import cv2
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from PIL import Image
from tqdm import tqdm

# ================[ Same DH + FK as in your Safe-Set Code ]================== #
def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha),  a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha),  a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d              ],
        [0,              0,                            0,                            1              ]
    ], dtype=float)

def franka_forward_kinematics_panda_hand(joint_angles):
    """
    Returns a 4x4 transform from the panda base to the 'panda_hand' link.
    Uses the same approach as your safe-set code, but includes the rotation
    about Z by -π/4 to get from flange -> panda_hand.
    """
    q1, q2, q3, q4, q5, q6, q7 = joint_angles
    # Up to the flange:
    T1 = dh_transform(a=0,       alpha=0,         d=0.333,   theta=q1)
    T2 = dh_transform(a=0,       alpha=-np.pi/2,  d=0,       theta=q2)
    T3 = dh_transform(a=0,       alpha=np.pi/2,   d=0.316,   theta=q3)
    T4 = dh_transform(a=0.0825,  alpha=np.pi/2,   d=0,       theta=q4)
    T5 = dh_transform(a=-0.0825, alpha=-np.pi/2,  d=0.384,   theta=q5)
    T6 = dh_transform(a=0,       alpha=np.pi/2,   d=0,       theta=q6)
    T7 = dh_transform(a=0.088,   alpha=np.pi/2,   d=0,       theta=q7)
    T_flange = dh_transform(a=0, alpha=0, d=0.107, theta=0)

    # Extra rotation flange -> panda_hand
    Rz_neg_45 = dh_transform(a=0, alpha=0, d=0, theta=-np.pi/4)

    T_panda_hand = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ T_flange @ Rz_neg_45
    return T_panda_hand

def joint_angles_to_end_effector_pose(joint_angles):
    """
    Convert 7 joint angles -> [x, y, z, qx, qy, qz, qw], referencing the 'panda_hand'.
    """
    T = franka_forward_kinematics_panda_hand(joint_angles)
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    rotation_matrix = T[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # [qx,qy,qz,qw]
    return np.array([x, y, z, quat[0], quat[1], quat[2], quat[3]], dtype=float)


# ================[ Your Original “Synthesis” Code ]================== #
from vivid123.generation_utils import generation_vivid123_new, prepare_vivid123_pipeline
from carvekit.api.high import HiInterface

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

def unit_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return np.array([1, 0, 0])
    return v / norm

def generate_candidate_lists(positions, rpy_list, num_samples=10):
    T = positions.shape[0]
    goal_pos = positions[-1]
    dx_ref = positions[0][0] - goal_pos[0]
    dy_ref = positions[0][1] - goal_pos[1]
    dz_ref = positions[0][2] - goal_pos[2]
    ref_r, ref_az, ref_el = cart2sph(dx_ref, dy_ref, dz_ref)
    
    base_max_deg = 5.0
    base_max_side = 0.1
    base_max_roll = 50 * np.pi/180.0
    base_max_pitch = 10 * np.pi/180.0
    base_max_yaw = 10 * np.pi/180.0
    lambda_decay = 1.5

    candidate_lists = []
    for i in range(T):
        x, y, z = positions[i]
        roll_i, pitch_i, yaw_i = rpy_list[i]
        dx = x - goal_pos[0]
        dy = y - goal_pos[1]
        dz = z - goal_pos[2]
        r_val, az, el = cart2sph(dx, dy, dz)
        droll = roll_i - rpy_list[-1][0]
        dpitch = pitch_i - rpy_list[-1][1]
        dyaw = yaw_i - rpy_list[-1][2]
        candidates_at_i = []
        for sample in range(num_samples):
            frac = np.exp(-lambda_decay*(i/(T - 1))) if T > 1 else 1.0
            delta_angle = frac * (base_max_deg * np.pi/180.0)
            az_var = az + random.choice([-1, 1]) * delta_angle
            el_var = el + random.choice([-1, 1]) * delta_angle
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
            roll_new = rpy_list[-1][0] + droll_var
            pitch_new = rpy_list[-1][1] + dpitch_var
            yaw_new = rpy_list[-1][2] + dyaw_var

            r_center, az_center, el_center = cart2sph(x_new - goal_pos[0],
                                                      y_new - goal_pos[1],
                                                      z_new - goal_pos[2])
            delta_az = az_center - ref_az
            delta_el = el_center - ref_el
            delta_r  = r_center - ref_r

            candidate = {
                "pose_6d": [x_new, y_new, z_new, roll_new, pitch_new, yaw_new],
                "delta_pose": [delta_az, delta_el, delta_r]
            }
            candidates_at_i.append(candidate)
        candidate_lists.append(candidates_at_i)
    return candidate_lists

def sample_candidate_chain(candidate_lists):
    chain = []
    for candidates in candidate_lists:
        chain.append(random.choice(candidates))
    return chain

def convert_pose6d_to_7d(pose6d, gripper_state):
    """
    Convert a 6D pose [x, y, z, roll, pitch, yaw] into a 7D action:
    [x, y, z, qx, qy, qz, gripper_state].
    We'll interpret (roll, pitch, yaw) as euler 'zyx'.
    """
    x, y, z, roll, pitch, yaw = pose6d
    quat = R.from_euler('zyx', [roll, pitch, yaw]).as_quat()  # [qx,qy,qz,qw]
    return np.array([x, y, z, quat[0], quat[1], quat[2], gripper_state], dtype=np.float32)

from vivid123.generation_utils import generation_vivid123_new, prepare_vivid123_pipeline

def synthesize_image_for_point(candidate, vivid123_pipe, xl_pipe, base_config, demo_num, chain_num, point_num):
    delta_pose = candidate.get("delta_pose", [0.0, 0.0, 0.0])
    delta_az, delta_el, delta_r = delta_pose
    config = base_config.copy()
    config["delta_azimuth_start"] = delta_az
    config["delta_azimuth_end"]   = delta_az
    config["delta_elevation_start"] = delta_el
    config["delta_elevation_end"]   = delta_el
    config["delta_radius_start"]    = delta_r
    config["delta_radius_end"]      = delta_r

    out_dir = os.path.join("temp_synth", f"demo_{demo_num}_chain_{chain_num}_point_{point_num}")
    os.makedirs(out_dir, exist_ok=True)
    config["obj_name"] = out_dir

    generation_vivid123_new(config=config, vivid123_pipe=vivid123_pipe, xl_pipe=xl_pipe)
    img_path = os.path.join(out_dir, "image000.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Synthesized image not found in {out_dir}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read synthesized image from {img_path}")
    img = cv2.resize(img, (320, 240))
    return img

from carvekit.api.high import HiInterface

def main():
    # number of candidate chains per main demo
    num_chains_per_demo = 3

    # Setup pipelines
    ZERO123_MODEL_ID = "bennyguo/zero123-xl-diffusers"
    vivid123_pipe, xl_pipe = prepare_vivid123_pipeline(
        ZERO123_MODEL_ID=ZERO123_MODEL_ID,
        VIDEO_MODEL_ID="cerspense/zeroscope_v2_576w",
        VIDEO_XL_MODEL_ID="cerspense/zeroscope_v2_XL"
    )
    # base config
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
        "input_image_path": "",
        "obj_name": "dummy",
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
    # Background removal
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

    input_file = "/home/carl_lab/Riad/duck_optimal_30.hdf5"
    in_h5 = h5py.File(input_file, "r")
    main_demo_keys = list(in_h5["data"].keys())
    main_demo_keys.sort()

    output_file = "generated_trajectories_s3.hdf5"
    out_h5 = h5py.File(output_file, "w")
    out_data_grp = out_h5.create_group("data")

    global_demo_counter = 0

    for main_key in main_demo_keys:
        print(f"\nProcessing main demo: {main_key}")
        demo_grp_in = in_h5["data"][main_key]
        obs_grp_in = demo_grp_in["obs"]

        # read joint_states and gripper_states
        joint_states = np.array(obs_grp_in["joint_states"])  # shape (T,7)
        gripper_states = np.array(obs_grp_in["gripper_states"])  # shape (T,1)
        T = joint_states.shape[0]

        # # positions = joint_states[:, :3]
        # # rpy_list = joint_states[:, 3:6]
        # However, you might actually want the EEF from the 'panda_hand' perspective:
        # Suppose your policy or code expects positions/rpy from the 7D EEF pose
        # We'll just interpret the last 4 as something else. 
        # If your joint_states are actually [x,y,z,roll,pitch,yaw,gripper], fine.
        # But if they are real robot J1..J7, you might do forward kinematics:

        positions = []   # We'll store the EEF pos from panda_hand
        rpy_list = []    # Euler angles: (roll, pitch, yaw)
        for js in joint_states:
            pose_7d = joint_angles_to_end_effector_pose(js)  # shape (7,)
            # extract position
            x, y, z = pose_7d[:3]
            # convert quaternion -> euler zyx for roll, pitch, yaw
            quat = pose_7d[3:7]
            eul_zyx = R.from_quat(quat).as_euler('zyx')
            roll, pitch, yaw = eul_zyx[0], eul_zyx[1], eul_zyx[2]
            positions.append([x, y, z])
            rpy_list.append([roll, pitch, yaw])
        positions = np.array(positions)
        rpy_list = np.array(rpy_list)

        # generate candidate lists
        candidate_lists = generate_candidate_lists(positions, rpy_list, num_samples=10)
        candidate_chains = []
        for _ in range(num_chains_per_demo):
            chain = sample_candidate_chain(candidate_lists)
            candidate_chains.append(chain)

        # Use the first eye_in_hand_rgb as reference
        ref_img_data = np.array(obs_grp_in["eye_in_hand_rgb"][0])
        if ref_img_data.shape[-1] == 3:
            ref_img_data = ref_img_data[..., ::-1]  # BGR->RGB if needed
        from PIL import Image
        ref_img = Image.fromarray(ref_img_data)
        temp_ref_folder = "temp_ref"
        os.makedirs(temp_ref_folder, exist_ok=True)
        ref_img_path = os.path.join(temp_ref_folder, f"{main_key}_eye_in_hand_rgb.png")
        ref_img.save(ref_img_path)

        # Remove background
        processed_ref = bg_interface([ref_img_path])[0]
        processed_ref.save(ref_img_path)
        base_config["input_image_path"] = ref_img_path

        for chain_idx, chain in enumerate(candidate_chains):
            
            synthesized_imgs = []
            actions = []
            joint_states_new = []
            ee_states = []
            gripper_states_new = []
            print("Saving output images.....")
            for t, candidate in tqdm(enumerate(chain), total=len(chain), desc="Synth points"):
                print(f"Synthesizing trajectory {global_demo_counter}")
                try:
                    img = synthesize_image_for_point(candidate, vivid123_pipe, xl_pipe, base_config,
                                                     demo_num=global_demo_counter, chain_num=chain_idx, point_num=t)
                except Exception as e:
                    print(f"Error synthesizing image at time {t}: {e}")
                    img = np.zeros((240, 320, 3), dtype=np.uint8)
                synthesized_imgs.append(img)

                # use corresponding gripper from original data
                gripper = gripper_states[t, 0] if t < len(gripper_states) else 0.0
                action_7d = convert_pose6d_to_7d(candidate["pose_6d"], gripper)
                actions.append(action_7d)
                joint_states_new.append(action_7d)
                eef = np.zeros(16, dtype=np.float32)
                eef[:7] = action_7d
                ee_states.append(eef)
                gripper_states_new.append([gripper])

            actions = np.array(actions, dtype=np.float32)   # shape (T,7)
            joint_states_new = np.array(joint_states_new, dtype=np.float32)  # shape (T,7)
            ee_states = np.array(ee_states, dtype=np.float32)  # shape (T,16)
            gripper_states_new = np.array(gripper_states_new, dtype=np.float32)  # shape (T,1)
            eye_in_hand_rgb_arr = np.stack(synthesized_imgs, axis=0)  # shape (T,240,320,3)

            demo_name_out = f"demo_{global_demo_counter}"
            demo_grp_out = out_data_grp.create_group(demo_name_out)
            demo_grp_out.create_dataset("actions", data=actions)
            obs_grp_out = demo_grp_out.create_group("obs")
            obs_grp_out.create_dataset("joint_states", data=joint_states_new)
            obs_grp_out.create_dataset("ee_states", data=ee_states)
            obs_grp_out.create_dataset("gripper_states", data=gripper_states_new)
            obs_grp_out.create_dataset("eye_in_hand_rgb", data=eye_in_hand_rgb_arr)
            print(f"Saved generated trajectory {demo_name_out} with {actions.shape[0]} steps.")
            global_demo_counter += 1

    in_h5.close()
    out_h5.close()
    print(f"\nAll main demos processed. Saved to '{output_file}'.")


if __name__ == "__main__":
    main()
