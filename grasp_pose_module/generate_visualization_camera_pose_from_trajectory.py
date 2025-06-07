#!/usr/bin/env python3
"""
save_visible_steps_matplotlib.py

Iterates over ALL poses in:
  /home/carl_lab/akash/gaussian-splatting/output/franka_outputs/raw_trajectory.npy

For each pose index i:
  1. raw_pose_noninv = traj[i]; raw_pose = inv(raw_pose_noninv)   → base→gripper
  2. Compute world→camera = inv(inv(W2B) @ raw_pose @ gTc), get xyz_cam0
  3. Brute-force flipY/flipZ to pick the maximum “in-frame” count
  4. Determine which Gaussians are visible (rgb window = 1280×720)
  5. Use Matplotlib’s 3D scatter to plot:
        • All Gaussians (gray, small dots)
        • Visible Gaussians (red, small dots)
     from a fixed elev/azim viewpoint (e.g. elev=30°, azim=45°)
  6. Save each figure as “visualizations/step_{i:03d}.png”.

This runs entirely on CPU via Matplotlib and will produce non-black, step-specific images.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D plotting)

# ───────────────────────────────────────────────────────────────────────
# 1) Gaussian-splatting imports (identical to your environment)
# ───────────────────────────────────────────────────────────────────────
sys.path.append("feature-splatting-inria")
sys.path.append(".")
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene


# ───────────────────────────────────────────────────────────────────────
# 2) Constants: world→base and gripper→camera transforms
# ───────────────────────────────────────────────────────────────────────
W2B = np.array([
    [ 0.986680115832,  0.12442503, -0.10478913,  0.373036 ],
    [ 0.16035478,     -0.8522923,   0.49787968, -0.0069075],
    [-0.02736227,     -0.50805142, -0.86089202,  0.4207423],
    [ 0,               0,            0,          1        ]
], dtype=np.float64)

gTc = np.array([
    [ 0.55898662, -0.82831147, -0.03786894,  0.04419197],
    [ 0.80713679,  0.55402063, -0.20393957,  0.11418125],
    [ 0.18990565,  0.08343408,  0.97825078, -0.00579795],
    [ 0,           0,           0,           1         ]
], dtype=np.float64)

# ───────────────────────────────────────────────────────────────────────
# 3) Helpers: load intrinsics, project, in-image test
# ───────────────────────────────────────────────────────────────────────
def load_intrinsics(path):
    """
    Load camera intrinsics from .npz (fx, fy, ppx, ppy).
    Returns (fx, fy, cx, cy).
    """
    d = np.load(path)
    return float(d["fx"]), float(d["fy"]), float(d["ppx"]), float(d["ppy"])

def project_graphics(p, fx, fy, cx, cy):
    """
    Given p (N×3) in camera frame, return (u, v, Z):
      u = fx * (X / Z) + cx
      v = -fy * (Y / Z) + cy
    """
    X, Y, Z = p[:,0], p[:,1], p[:,2]
    u = fx * (X / Z) + cx
    v = -fy * (Y / Z) + cy
    return u, v, Z

def in_img(u, v, z, W, H):
    """
    Given arrays (u, v, z) and image dims (W, H),
    return boolean mask (N,) where z>1e-6 and 0 <= round(u)<W and 0 <= round(v)<H.
    """
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    return (z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

# ───────────────────────────────────────────────────────────────────────
# 4) CLI argument parsing
# ───────────────────────────────────────────────────────────────────────
def make_args():
    p = argparse.ArgumentParser(__doc__)
    ModelParams(p)
    PipelineParams(p)
    OptimizationParams(p)

    # Set your defaults for model_path and source_path:
    p.set_defaults(
        model_path="/home/carl_lab/akash/gaussian-splatting/output/franka_outputs",
        source_path="/home/carl_lab/Riad/graspbility/GraspSplats/franka_data"
    )
    p.add_argument(
        "--intrinsics", type=str, default="rgb_intrinsics.npz",
        help="Camera intrinsics .npz (fx, fy, ppx, ppy)."
    )
    p.add_argument(
        "--output-folder", type=str, default="visualizations",
        help="Folder in which to save step_XXX.png images."
    )
    return get_combined_args(p)

# ───────────────────────────────────────────────────────────────────────
# 5) MAIN: loop through poses, invert, project, plot via Matplotlib, save
# ───────────────────────────────────────────────────────────────────────
def main():
    a = make_args()

    # 5.1) Load Gaussian model & scene once → world‐frame Gaussians
    g = GaussianModel(a.sh_degree, a.distill_feature_dim)
    g.training_setup(a)
    Scene(a, g, load_iteration=-1, shuffle=False)
    xyz_world = g.get_xyz.detach().cpu().numpy()  # shape: (N, 3)

    # 5.2) Load intrinsics once; fix “image” window = 1280×720
    fx, fy, cx, cy = load_intrinsics(os.path.join(a.source_path, a.intrinsics))
    W, H = 1280, 720

    # 5.3) Load all raw poses from raw_trajectory.npy: shape = (N_steps, 4, 4)
    traj_path = os.path.join(a.model_path, "raw_trajectory.npy")
    traj = np.load(traj_path)
    if traj.ndim != 3 or traj.shape[1:] != (4,4):
        raise ValueError(f"Expected raw_trajectory.npy of shape (N,4,4), got {traj.shape}")
    N_steps = traj.shape[0]
    print(f"[+] Loaded {N_steps} poses from '{traj_path}'")

    # 5.4) Prepare output folder
    os.makedirs(a.output_folder, exist_ok=True)

    # 5.5) Precompute inverses of gTc and W2B once
    try:
        inv_gTc = np.linalg.inv(gTc)
    except np.linalg.LinAlgError:
        raise RuntimeError("gTc is singular; cannot invert.")
    invW2B = np.linalg.inv(W2B)

    # 5.6) Set up a fixed Matplotlib 3D view:
    #      We’ll use elev=30°, azim=45° as a consistent viewpoint.
    elev = 30
    azim = 45

    # 5.7) Compute axis limits so that all Gaussians fit in view:
    x_min, x_max = xyz_world[:,0].min(), xyz_world[:,0].max()
    y_min, y_max = xyz_world[:,1].min(), xyz_world[:,1].max()
    z_min, z_max = xyz_world[:,2].min(), xyz_world[:,2].max()

    # Expand bounds slightly for padding:
    pad = 0.05 * np.maximum(x_max-x_min, np.maximum(y_max-y_min, z_max-z_min))
    x_lim = (x_min - pad, x_max + pad)
    y_lim = (y_min - pad, y_max + pad)
    z_lim = (z_min - pad, z_max + pad)

    # 5.8) Loop over each pose index i
    for i in range(N_steps):
        print(f"\n[STEP {i+1}/{N_steps}] Processing pose index = {i}")

        # 5.8.1) Invert raw pose so that raw_pose = base→gripper
        raw_pose_noninv = traj[i]
        raw_pose        = np.linalg.inv(raw_pose_noninv)

        # 5.8.2) Compute world→camera exactly as in your single-pose code
        bTg   = raw_pose
        wTcam = np.linalg.inv(np.linalg.inv(W2B) @ bTg @ gTc)
        R_wc, t_wc = wTcam[:3,:3], wTcam[:3,3]
        xyz_cam0   = (R_wc @ xyz_world.T).T + t_wc  # (N, 3) in camera frame

        # 5.8.3) Brute-force four (flipY, flipZ) to maximize visible count
        variants = {
            ( 1,  1): xyz_cam0,
            ( 1, -1): xyz_cam0 * [1, -1,  1],
            (-1,  1): xyz_cam0 * [1,  1, -1],
            (-1, -1): xyz_cam0 * [1, -1, -1],
        }
        best_cnt     = -1
        best_xyz_cam = None

        for flips, pts_cam in variants.items():
            u, v, z = project_graphics(pts_cam, fx, fy, cx, cy)
            in_mask = in_img(u, v, z, W, H)
            cnt     = int(in_mask.sum())
            if cnt > best_cnt:
                best_cnt     = cnt
                best_xyz_cam = pts_cam

        print(f"    → {best_cnt} splats are in-frame after flips")

        # 5.8.4) Compute final visible subset in world coords
        u_f, v_f, z_f   = project_graphics(best_xyz_cam, fx, fy, cx, cy)
        mask_final      = in_img(u_f, v_f, z_f, W, H)
        xyz_vis_world   = xyz_world[mask_final]  # (M_vis, 3)

        # 5.8.5) Plot with Matplotlib’s 3D axes:
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(projection='3d')

        # (a) Plot all Gaussians in gray:
        ax.scatter(
            xyz_world[:,0], xyz_world[:,1], xyz_world[:,2],
            c='gray', s=0.5, linewidth=0, alpha=0.3
        )

        # (b) Plot visible subset in red:
        ax.scatter(
            xyz_vis_world[:,0], xyz_vis_world[:,1], xyz_vis_world[:,2],
            c='red', s=0.5, linewidth=0, alpha=1.0
        )

        # Set axis limits (so camera doesn’t auto-rescale each frame):
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)

        # Hide grid lines and axes ticks for a cleaner look:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1,1,1))  # equal aspect ratio

        # Fix the camera viewpoint to (elev, azim):
        ax.view_init(elev=elev, azim=azim)

        # Save to disk:
        out_name = f"step_{i:03d}.png"
        out_path = os.path.join(a.output_folder, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"    → Saved '{out_path}'")

    print(f"\n[+] Done. All {N_steps} images saved in '{a.output_folder}'")

if __name__ == "__main__":
    main()
