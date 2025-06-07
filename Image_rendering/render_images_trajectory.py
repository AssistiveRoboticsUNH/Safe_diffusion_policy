import os
import numpy as np
from matplotlib import pyplot as plt
from render_for_robot_class_v3_module import GaussianSplatRenderer

import torch
import torchvision.utils as vutils

# -----------------------------------------------------------------------------
# 1) Initialize renderer (adjust model_path as needed)
# -----------------------------------------------------------------------------
renderer = GaussianSplatRenderer(
    model_path="/home/carl_lab/akash/gaussian-splatting/output/franka_outputs"
)

# -----------------------------------------------------------------------------
# 2) Load all saved “raw” poses from the .npy file
#    (this file was created in the main pipeline as raw_trajectory.npy)
# -----------------------------------------------------------------------------
trajectory_file = "/home/carl_lab/akash/gaussian-splatting/output/franka_outputs/raw_trajectory.npy"
poses_raw = np.load(trajectory_file)  # shape: (N_steps, 4, 4)

# -----------------------------------------------------------------------------
# 3) Create an output directory to hold per‐pose renders
# -----------------------------------------------------------------------------
output_dir = "test_2"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 4) Iterate over each raw pose, render, rotate, crop to square, and save
# -----------------------------------------------------------------------------
for idx, raw_pose in enumerate(poses_raw):
    # 4.1) Invert raw_pose (base→camera) → camera→world for the renderer:
    cam2world = np.linalg.inv(raw_pose)
    # cam2world = raw_pose

    # 4.2) Render the (3, H, W) RGB tensor
    rendered_tensor = renderer.get_rendered_image(cam2world, pose_type="4x4")
    # rendered_tensor is a torch.Tensor of shape (3, H, W) in [0,1].

    # 4.3) (Optional) Rotate anticlockwise by 90°:
    rendered_tensor = torch.rot90(rendered_tensor, k=1, dims=[1, 2])

    # 4.4) Center‐crop to a square: height and width might differ,
    #      so pick min_dim = min(H, W) and crop equally on both sides.
    C, H, W = rendered_tensor.shape
    min_dim = min(H, W)
    top = (H - min_dim) // 2
    left = (W - min_dim) // 2
    # Crop: channels stay the same; crop rows [top:top+min_dim], cols [left:left+min_dim]
    square_tensor = rendered_tensor[:, top : top + min_dim, left : left + min_dim]

    # 4.5) Save with torchvision (now shape = (3, min_dim, min_dim))
    out_path = os.path.join(output_dir, f"pose_{idx:04d}.png")
    vutils.save_image(square_tensor, out_path)

    print(f"[+] Saved square‐cropped image for pose {idx} → {out_path}")

print("All poses rendered, cropped to square, and saved.")
