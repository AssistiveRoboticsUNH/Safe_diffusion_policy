#!/usr/bin/env python3


import os
import sys
import argparse
import tempfile

import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ───────────────────────────────────────────────────────────────────────
# 1) Gaussian‐splatting & Grasp imports (unchanged)
# ───────────────────────────────────────────────────────────────────────
sys.path.append("feature-splatting-inria")
sys.path.append(".")
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene, skip_feat_decoder
import featsplat_editor
from gaussian_edit import edit_utils
from grasping.grasping_utils import sample_grasps, plot_gripper_pro_max

# from grasping.grasping_utils import sample_grasps

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
# 3) Helpers: load intrinsics, project, in‐image test
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
    Given p (N×3) in camera frame, return (u, v, Z) arrays:
      u = fx * (X/Z) + cx
      v = -fy * (Y/Z) + cy
    """
    X, Y, Z = p[:,0], p[:,1], p[:,2]
    u = fx * (X / Z) + cx
    v = -fy * (Y / Z) + cy
    return u, v, Z

def in_img(u, v, z, W, H):
    """
    Given projected u, v, z and image dims W,H, return bool mask (N,)
    where z>1e-6, 0 <= round(u)<W, 0 <= round(v)<H.
    """
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    return (z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

# ───────────────────────────────────────────────────────────────────────
# 4) CLI argument parsing
# ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(__doc__)
    ModelParams(p)
    p.set_defaults(model_path="/home/carl_lab/akash/gaussian-splatting/output/franka_outputs")
    p.set_defaults(source_path="/home/carl_lab/Riad/graspbility/GraspSplats/franka_data")
    PipelineParams(p)
    OptimizationParams(p)

    # CLIP segmentation queries
    p.add_argument(
        "-q", "--object-query", default="Red candy",
        help="CLIP object query (default: 'Red candy')"
    )
    p.add_argument(
        "-n", "--negative-query", default="Green candy",
        help="Optional negative CLIP query (default: 'Green candy')"
    )
    p.add_argument(
        "-p", "--part-query", default="Red candy",
        help="Optional CLIP part query (default: 'Red candy')"
    )
    p.add_argument(
        "--obj-pos-thresh", type=float, default=0.75,
        help="Positive‐query similarity threshold (default: 0.75)"
    )
    p.add_argument(
        "--obj-neg-thresh", type=float, default=0.7,
        help="Negative‐query similarity threshold (default: 0.7)"
    )
    p.add_argument(
        "--part-thresh", type=float, default=0.7,
        help="Part‐query similarity threshold (default: 0.7)"
    )

    # Intrinsics (we don’t use rgb-file here; fix W,H = 1280×720)
    p.add_argument(
        "--intrinsics", default="rgb_intrinsics.npz",
        help="Camera intrinsics .npz with keys 'fx','fy','ppx','ppy'."
    )

    # TSDF (unused for visualization)
    p.add_argument(
        "--tsdf-ply",
        default="sparse/0/points3D.ply",
        help="TSDF PLY file (camera coords)—not used in this script."
    )

    # Grasp‐sampling hyperparams
    p.add_argument(
        "--topk", type=int, default=1,
        help="Number of top grasps to keep (default: 1)."
    )
    p.add_argument(
        "--dist-thresh", type=float, default=0.02,
        help="Min‐distance threshold (default: 0.02 m) for collision filter."
    )

    p.add_argument(
        "--output-folder", type=str, default="visualizations_grasps",
        help="Directory to save step_XXX.png images."
    )

    return get_combined_args(p)

# ───────────────────────────────────────────────────────────────────────
# 5) MAIN: iterate over each camera pose, compute grasp, plot via Matplotlib
# ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 5.1) Load GaussianModel & Scene (camera frame) once → world‐frame Gaussians
    g = GaussianModel(args.sh_degree, args.distill_feature_dim)
    g.training_setup(args)
    Scene(args, g, load_iteration=-1, shuffle=False)
    xyz_world = g.get_xyz.detach().cpu().numpy()  # shape: (N,3)

    # 5.2) Load camera intrinsics once; fix W=1280, H=720
    fx, fy, cx, cy = load_intrinsics(os.path.join(args.source_path, args.intrinsics))
    W, H = 1280, 720

    # 5.3) Load raw trajectory: shape = (N_steps, 4, 4)
    traj_path = os.path.join(args.model_path, "raw_trajectory.npy")
    traj = np.load(traj_path)
    if traj.ndim != 3 or traj.shape[1:] != (4,4):
        raise ValueError(f"Expected raw_trajectory.npy shape (N,4,4), got {traj.shape}")
    N_steps = traj.shape[0]
    print(f"[+] Loaded {N_steps} poses from '{traj_path}'")

    # 5.4) Prepare output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # 5.5) Precompute inv(gTc) and inv(W2B)
    try:
        inv_gTc = np.linalg.inv(gTc)
    except np.linalg.LinAlgError:
        raise RuntimeError("gTc is singular; cannot invert.")
    invW2B = np.linalg.inv(W2B)

    # 5.6) Matplotlib 3D viewpoint: fixed elev/azim
    elev = 0
    azim = 0

    # 5.7) Loop over each camera‐trajectory pose
    for i in range(1):
        print(f"\n[STEP {i+1}/{N_steps}] Processing camera pose index = {i}")

        # 5.7.1) Invert raw pose: raw_pose = inv(traj[i]) → base→gripper
        raw_pose_noninv = traj[i]
        raw_pose        = np.linalg.inv(raw_pose_noninv)

        # 5.7.2) Compute world→camera transform
        bTg    = raw_pose
        wTcam  = np.linalg.inv(np.linalg.inv(W2B) @ bTg @ gTc)  # world→camera
        R_wc, t_wc = wTcam[:3,:3], wTcam[:3,3]
        xyz_cam0 = (R_wc @ xyz_world.T).T + t_wc  # (N,3) in camera frame

        # 5.7.3) Brute‐force flipY/flipZ to maximize “visible” count
        variants = {
            ( 1,  1): xyz_cam0,
            ( 1, -1): xyz_cam0 * [1, -1,  1],
            (-1,  1): xyz_cam0 * [1,  1, -1],
            (-1, -1): xyz_cam0 * [1, -1, -1],
        }
        best_cnt     = -1
        best_xyz_cam = None
        visible_mask = None

        for flips, pts_cam in variants.items():
            u, v, z = project_graphics(pts_cam, fx, fy, cx, cy)
            mask_vis = in_img(u, v, z, W, H)
            cnt = int(mask_vis.sum())
            if cnt > best_cnt:
                best_cnt     = cnt
                best_xyz_cam = pts_cam
                visible_mask = mask_vis.copy()

        print(f"    → {best_cnt} Gaussians visible in camera frame")

        # 5.7.4) Extract visible splats (camera coords)
        xyz_cam_vis = best_xyz_cam[visible_mask]  # (M_vis,3)

        # 5.7.5) CLIP‐based segmentation on all Gaussians
        decoder = skip_feat_decoder(args.distill_feature_dim, part_level=True).cuda()
        decoder.load_state_dict(torch.load(
            os.path.join(args.model_path, "feat_decoder.pth"), map_location="cpu"
        ))
        decoder.eval()
        segm = featsplat_editor.clip_segmenter(g, decoder)

        with torch.no_grad():
            sim_pos = segm.compute_similarity_one(args.object_query, level="object")
        sim_pos = sim_pos.cpu().numpy() if torch.is_tensor(sim_pos) else sim_pos
        mask_pos = (sim_pos > args.obj_pos_thresh)

        if args.negative_query:
            with torch.no_grad():
                sim_neg = segm.compute_similarity_one(args.negative_query, level="object")
            sim_neg = sim_neg.cpu().numpy() if torch.is_tensor(sim_neg) else sim_neg
            mask_pos &= (sim_neg < args.obj_neg_thresh)

        if args.part_query:
            with torch.no_grad():
                sim_part = segm.compute_similarity_one(args.part_query, level="part")
            sim_part = sim_part.cpu().numpy() if torch.is_tensor(sim_part) else sim_part
            mask_pos &= (sim_part > args.part_thresh)

        # 5.7.6) Intersect segmentation with visible_mask
        segmented_visible_mask = mask_pos & visible_mask

        # 5.7.7) Post‐process on camera‐frame coords of visible splats
        pts_cam_vis_all = best_xyz_cam  # (N,3) camera coords
        segmented_visible_mask = edit_utils.cluster_instance(pts_cam_vis_all, segmented_visible_mask)
        segmented_visible_mask = edit_utils.flood_fill(pts_cam_vis_all, segmented_visible_mask)

        num_vis     = int(visible_mask.sum())
        num_seg_vis = int(segmented_visible_mask.sum())
        print(f"    → {num_seg_vis} / {num_vis} visible splats after segmentation")

        if num_seg_vis < 100:
            print("    [WARNING] Too few segmented splats; skipping grasp sampling.")
            xyz_cam_seg_vis = np.zeros((0,3))
            best_pose_cam   = None
        else:
            xyz_cam_seg_vis = pts_cam_vis_all[segmented_visible_mask]  # (M_seg,3)

            # 5.7.9) Sample grasps on segmented cloud
            pcd_seg_cam = o3d.geometry.PointCloud()
            pcd_seg_cam.points = o3d.utility.Vector3dVector(xyz_cam_seg_vis)
            pcd_seg_cam.paint_uniform_color([1.0, 0.0, 0.0])  # red

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pcd")
            os.close(tmp_fd)
            o3d.io.write_point_cloud(tmp_path, pcd_seg_cam)
            local_poses_cam, local_scores = sample_grasps(tmp_path, if_global=False)
            os.remove(tmp_path)

            if len(local_poses_cam) == 0:
                print("    [WARNING] No grasps found; skipping grasp visualization.")
                best_pose_cam = None
            else:
                final_grasps = []
                ones_seg = np.ones((xyz_cam_seg_vis.shape[0], 1))
                homo_seg_cam = np.hstack([xyz_cam_seg_vis, ones_seg])   # (M_seg,4)
                obj_pts_base = (W2B @ homo_seg_cam.T).T[:, :3]          # (M_seg,3)

                for (pose_cam, score) in zip(local_poses_cam, local_scores):
                    R_cam = pose_cam[:3,:3]
                    t_cam = pose_cam[:3,3]

                    # a) rotation → base
                    R_base_raw = W2B[:3,:3] @ R_cam
                    base_gaze = R_base_raw[:, 2].copy()
                    dir_xy = base_gaze.copy()
                    dir_xy[2] = 0.0
                    norm_xy = np.linalg.norm(dir_xy)
                    if norm_xy < 1e-6:
                        x_gripper = np.array([1.0, 0.0, 0.0])
                    else:
                        x_gripper = dir_xy / norm_xy
                    z_gripper = np.array([0.0, 0.0, -1.0])
                    y_gripper = np.cross(z_gripper, x_gripper)
                    y_gripper /= np.linalg.norm(y_gripper)
                    R_base_hacked = np.column_stack((x_gripper, y_gripper, z_gripper))

                    # b) translation → base
                    t_base = (W2B @ np.hstack([t_cam, 1.0]))[:3]

                    # c) distance filter
                    t_mid_cam  = t_cam + 0.05 * R_cam[:,2]
                    t_mid_base = (W2B @ np.hstack([t_mid_cam, 1.0]))[:3]
                    dists = np.linalg.norm(obj_pts_base - t_mid_base, axis=1)
                    if np.min(dists) < args.dist_thresh:
                        continue

                    pose_base = np.eye(4, dtype=np.float64)
                    pose_base[:3,:3] = R_base_hacked
                    pose_base[:3,3]  = t_base
                    final_grasps.append((pose_base, score, pose_cam))

                final_grasps = sorted(final_grasps, key=lambda x: -x[1])[: args.topk]
                if not final_grasps:
                    print("    [WARNING] No grasp passed filtering.")
                    best_pose_cam = None
                else:
                    _, best_score, best_pose_cam = final_grasps[0]
                    print(f"    [+] Best grasp score = {best_score:.4f}")

        # ─────────────────────────────────────────────────────────────────
        # 5.8) Plot in CAMERA frame via Matplotlib + full gripper mesh
        #       but now mirror X_plot so the correct face is forward
        # ─────────────────────────────────────────────────────────────────

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(projection='3d')

        # ─────────────────────────────────────────────────────────────────────
        # (A) Plot VISIBLE splats in gray
        #    Remap camera‐coords → plot‐coords:
        #      X_plot = -X_cam
        #      Y_plot = -Z_cam   (so forward points have negative Y_plot)
        #      Z_plot =  Y_cam
        # ─────────────────────────────────────────────────────────────────────
        if xyz_cam_vis.shape[0] > 0:
            X_plot_vis = -xyz_cam_vis[:, 0]    # NEGATE X_cam
            Y_plot_vis = -xyz_cam_vis[:, 2]    # NEGATE Z_cam
            Z_plot_vis =  xyz_cam_vis[:, 1]    # Y_cam → Z_plot

            ax.scatter(
                X_plot_vis,
                Y_plot_vis,
                Z_plot_vis,
                c='gray', s=0.5, alpha=0.3, linewidths=0
            )

        # ─────────────────────────────────────────────────────────────────────
        # (B) Plot SEGMENTED+visible splats in red
        #    Again: X_plot = -X_cam, Y_plot = -Z_cam, Z_plot = Y_cam
        # ─────────────────────────────────────────────────────────────────────
        if xyz_cam_seg_vis.shape[0] > 0:
            X_plot_seg = -xyz_cam_seg_vis[:, 0]   # NEGATE X_cam
            Y_plot_seg = -xyz_cam_seg_vis[:, 2]   # NEGATE Z_cam
            Z_plot_seg =  xyz_cam_seg_vis[:, 1]   # Y_cam → Z_plot

            ax.scatter(
                X_plot_seg,
                Y_plot_seg,
                Z_plot_seg,
                c='red', s=0.5, alpha=1.0, linewidths=0
            )

        # ─────────────────────────────────────────────────────────────────────
        # (C) Plot the gripper MESH using plot_gripper_pro_max
        #    Apply the same remapping to the gripper’s vertices
        # ─────────────────────────────────────────────────────────────────────
        if best_pose_cam is not None:
            center_cam = best_pose_cam[:3, 3]
            R_cam      = best_pose_cam[:3, :3]

            gripper_cam_mesh = plot_gripper_pro_max(
                center_cam,
                R_cam,
                0.08,  # outer “palm” radius
                0.06   # inner “finger” radius
            )

            verts_cam = np.asarray(gripper_cam_mesh.vertices)   # (V,3)
            tris      = np.asarray(gripper_cam_mesh.triangles)  # (F,3)

            # Remap each vertex from camera→plot:
            #   X_plot = -X_cam
            #   Y_plot = -Z_cam
            #   Z_plot =  Y_cam
            verts_plot = np.zeros_like(verts_cam)
            verts_plot[:, 0] = -verts_cam[:, 0]   # NEGATE X_cam
            verts_plot[:, 1] = -verts_cam[:, 2]   # NEGATE Z_cam
            verts_plot[:, 2] =  verts_cam[:, 1]   # Y_cam → Z_plot

            poly3d = [[verts_plot[v_idx] for v_idx in face] for face in tris]
            gripper_collection = Poly3DCollection(
                poly3d,
                facecolors=(0.2, 0.2, 0.2, 0.8),  # dark gray, semi‐transparent
                edgecolors='none'
            )
            ax.add_collection3d(gripper_collection)

        # ─────────────────────────────────────────────────────────────────────
        # (D) Compute plot‐space axis limits (using the camera→plot remap)
        # ─────────────────────────────────────────────────────────────────────
        if (xyz_cam_vis.shape[0] > 0) or (xyz_cam_seg_vis.shape[0] > 0):
            if xyz_cam_seg_vis.shape[0] == 0:
                all_cam_pts = xyz_cam_vis
            else:
                all_cam_pts = np.vstack((xyz_cam_vis, xyz_cam_seg_vis))

            # In camera coords
            X_all = all_cam_pts[:, 0]   # X_cam
            Y_all = all_cam_pts[:, 2]   # Z_cam
            Z_all = all_cam_pts[:, 1]   # Y_cam

            x_min, x_max = X_all.min(), X_all.max()
            y_min, y_max = Y_all.min(), Y_all.max()
            z_min, z_max = Z_all.min(), Z_all.max()

            pad = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-3)

            # Set PLOT limits:
            #  X_plot ∈ [-(x_max+pad), -(x_min−pad)]  ← because X_plot = -X_cam
            #  Y_plot ∈ [-(y_max+pad), -(y_min−pad)]  ← because Y_plot = -Z_cam
            #  Z_plot ∈ [ (z_min−pad),   (z_max+pad) ]← because Z_plot = Y_cam
            ax.set_xlim(-(x_max + pad), -(x_min - pad))
            ax.set_ylim(-(y_max + pad), -(y_min - pad))
            ax.set_zlim(z_min - pad, z_max + pad)

        # ─────────────────────────────────────────────────────────────────────
        # (E) White background + hide axes
        # ─────────────────────────────────────────────────────────────────────
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1))

        # ─────────────────────────────────────────────────────────────────────
        # (F) FIXED viewpoint: put Matplotlib’s camera on +Y_plot (azim=90)
        #     This means Matplotlib looks down the negative‐Y_plot axis. Since
        #     Y_plot = -Z_cam, any Z_cam>0 ("forward" in camera) lands at Y_plot<0,
        #     and thus appears directly in front of the viewer.
        # ─────────────────────────────────────────────────────────────────────
        ax.view_init(elev=0, azim=90)

        # ─────────────────────────────────────────────────────────────────────
        # (G) Save the figure
        # ─────────────────────────────────────────────────────────────────────
        out_name = f"step_{i:03d}.png"
        out_path = os.path.join(args.output_folder, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"    → Saved '{out_path}'")




    print(f"\n[+] Done. All {N_steps} grasp images saved in '{args.output_folder}'")

if __name__ == "__main__":
    main()
