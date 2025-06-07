#!/usr/bin/env python3

import os
import sys
import argparse
import tempfile

import numpy as np
import torch
import open3d as o3d
import cv2
from spatialmath import SO3

# ───────────────────────────────────────────────────────────────────────
# 1) Gaussian Splatting imports
# ───────────────────────────────────────────────────────────────────────
sys.path.append("feature-splatting-inria")
sys.path.append(".")
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene, skip_feat_decoder
import featsplat_editor
from gaussian_edit import edit_utils

# ───────────────────────────────────────────────────────────────────────
# 2) Grasp‐sampling imports
# ───────────────────────────────────────────────────────────────────────
from grasping.grasping_utils import sample_grasps, plot_gripper_pro_max

# ───────────────────────────────────────────────────────────────────────
# 3) Constants: Transforms for visibility check & grasp‐orientation hack
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
# 4) Helpers: Load intrinsics, project, in‐image test
# ───────────────────────────────────────────────────────────────────────
def load_intrinsics(path):
    """
    Load camera intrinsics from a .npz file with keys 'fx','fy','ppx','ppy'.
    Returns (fx, fy, cx, cy).
    """
    d = np.load(path)
    return float(d["fx"]), float(d["fy"]), float(d["ppx"]), float(d["ppy"])

def project_graphics(p, fx, fy, cx, cy):
    """
    Given p of shape (N,3) in camera frame, return (u, v, Z) arrays:
      u = fx * (X/Z) + cx
      v = -fy * (Y/Z) + cy
    """
    X, Y, Z = p[:,0], p[:,1], p[:,2]
    u = fx * (X / Z) + cx
    v = -fy * (Y / Z) + cy
    return u, v, Z

def in_img(u, v, z, W, H):
    """
    Given projected arrays u,v,z and image width W, height H,
    return boolean mask (N,) where:
      z > 1e-6, 0 <= round(u) < W, 0 <= round(v) < H
    """
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    return (z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

# ───────────────────────────────────────────────────────────────────────
# 5) Argument parsing
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
        "-q", "--object-query", default="Green candy",
        help="CLIP object query (default: 'Red candy')"
    )
    p.add_argument(
        "-n", "--negative-query", default="Red candy",
        help="Optional negative CLIP query (default: 'Green candy')"
    )
    p.add_argument(
        "-p", "--part-query", default="Green candy",
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

    # Pose file & RGB/intrinsics
    p.add_argument(
        "--pose-file", type=str,
        default="poses/8.npy",
        help="Path to a .npy file containing a 4×4 base→gripper pose."
    )
    p.add_argument(
        "--rgb-file", default="images/8.png",
        help="RGB image path (used to read W,H for cropping)."
    )
    p.add_argument(
        "--intrinsics", default="rgb_intrinsics.npz",
        help="Camera intrinsics .npz with 'fx','fy','ppx','ppy'."
    )

    # TSDF for visualizing scene (unused in final sampling, but included for interface)
    p.add_argument(
        "--tsdf-ply",
        default="sparse/0/points3D.ply",
        help="Path to TSDF PLY (camera-frame) for scene visualization."
    )

    # Grasp-sampling filters
    p.add_argument(
        "--topk", type=int, default=1,
        help="Number of top grasps to keep (default: 1)."
    )
    p.add_argument(
        "--dist-thresh", type=float, default=0.02,
        help="Min‐distance threshold (default: 0.02 m) for collision filter."
    )

    # Visualization toggles
    p.add_argument(
        "--camviz", action="store_true",
        help="If set, show 3D visualization in CAMERA coords (visible, segmented, best grasp)."
    )
    p.add_argument(
        "--point-size", type=float, default=1.4,
        help="Point size for 3D visualization."
    )

    return get_combined_args(p)

# ───────────────────────────────────────────────────────────────────────
# 6) Main: Crop → Segment → Sample Grasps → Transform → Filter → Pick Best → Visualize (CAMERA only)
# ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 6.1) Load GaussianModel & Scene → get all Gaussians in world frame
    g = GaussianModel(args.sh_degree, args.distill_feature_dim)
    g.training_setup(args)
    Scene(args, g, load_iteration=-1, shuffle=False)
    xyz_world = g.get_xyz.detach().cpu().numpy()  # (N,3)
    # print(f"[DEBUG] Loaded {xyz_world.shape[0]} Gaussians in world frame.")

    # 6.2) Compute world→camera transform (wTcam)
    bTg = np.load(os.path.join(args.source_path, args.pose_file))  # base→gripper
    wTcam = np.linalg.inv(np.linalg.inv(W2B) @ bTg @ gTc)
    R_wc, t_wc = wTcam[:3,:3], wTcam[:3,3]
    # print(f"[DEBUG] bTg (base→gripper):\n{bTg}")
    # print(f"[DEBUG] Computed wTcam (world→camera):\n{wTcam}")

    # 6.3) Transform world → camera (unflipped)
    xyz_cam0 = (R_wc @ xyz_world.T).T + t_wc  # (N,3)
    # print(f"[DEBUG] Transformed all Gaussians to CAMERA frame. Sample points:")
    for i in range(0, len(xyz_cam0), max(1, len(xyz_cam0)//5)):
        print(f"  Camera coords sample {i}: {xyz_cam0[i]}")

    # 6.4) Load intrinsics & detect center-crop
    fx, fy, cx, cy = load_intrinsics(os.path.join(args.source_path, args.intrinsics))
    H, W, _ = cv2.imread(os.path.join(args.source_path, args.rgb_file)).shape
    # print(f"[DEBUG] Image size: W={W}, H={H}; Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    W_full, H_full = int(round(2*cx)), int(round(2*cy))
    crop_left = max((W_full - W)//2, 0)
    crop_top  = max((H_full - H)//2, 0)
    if crop_left or crop_top:
        print(f"[INFO] crop detected  left={crop_left}px  top={crop_top}px")

    cx -= crop_left
    cy -= crop_top

    # 6.5) Brute-force (flipY, flipZ) handedness search for visible mask
    variants = {
        ( 1,  1): xyz_cam0,
        ( 1, -1): xyz_cam0 * [1, -1,  1],
        (-1,  1): xyz_cam0 * [1,  1, -1],
        (-1, -1): xyz_cam0 * [1, -1, -1],
    }
    best_cnt      = -1
    best_key      = None
    best_xyz      = None
    visible_mask  = None

    for flips, pts in variants.items():
        u, v, z = project_graphics(pts, fx, fy, cx, cy)
        u -= crop_left
        v -= crop_top
        mask_vis = in_img(u, v, z, W, H)  # (N,)
        cnt = int(mask_vis.sum())
        if cnt > best_cnt:
            best_cnt     = cnt
            best_key     = flips
            best_xyz     = pts
            visible_mask = mask_vis.copy()

    print(f"[INFO] Visibility flips chosen  Y={best_key[0]}  Z={best_key[1]}  → {best_cnt} visible Gaussians")

    # Save visible Gaussians in camera frame
    xyz_cam_vis = best_xyz[visible_mask]  # (M_vis, 3)
    idxs_visible = np.nonzero(visible_mask)[0]
    # print(f"[DEBUG] Number of visible (cropped) Gaussians: {xyz_cam_vis.shape[0]}")
    centroid_cam_vis = np.mean(xyz_cam_vis, axis=0)
    # print(f"[DEBUG] Centroid of cropped Gaussians in CAMERA frame: {centroid_cam_vis}")

    # 6.6) CLIP‐based segmentation on all Gaussians
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

    # 6.7) Intersect segmentation with visibility
    segmented_visible_mask = mask_pos & visible_mask

    # 6.8) Post‐process on camera‐frame coords of visible Gaussians
    pts_cam_vis = best_xyz  # (N,3) camera‐frame coords after flipping
    segmented_visible_mask = edit_utils.cluster_instance(pts_cam_vis, segmented_visible_mask)
    segmented_visible_mask = edit_utils.flood_fill(pts_cam_vis, segmented_visible_mask)

    num_vis     = int(visible_mask.sum())
    num_seg_vis = int(segmented_visible_mask.sum())
    print(f"[INFO] {num_seg_vis} / {num_vis} visible Gaussians match '{args.object_query}' after segmentation")

    if num_seg_vis < 100:
        print("[WARNING] No visible+segmented Gaussians; cannot sample grasps.")
        return

    # Extract camera‐frame coords of segmented+visible Gaussians
    xyz_cam_seg_vis = pts_cam_vis[segmented_visible_mask]  # (M_seg, 3) camera
    centroid_cam_seg = np.mean(xyz_cam_seg_vis, axis=0)
    # print(f"[DEBUG] Centroid of cropped+segmented Gaussians in CAMERA frame: {centroid_cam_seg}")

    # ─────────────────────────────────────────────────────────────────
    # 6.9) Run the 6‐DOF sampler on the cropped+segmented cloud (camera‐frame)
    # ─────────────────────────────────────────────────────────────────
    pcd_seg_cam = o3d.geometry.PointCloud()
    pcd_seg_cam.points = o3d.utility.Vector3dVector(xyz_cam_seg_vis)
    pcd_seg_cam.paint_uniform_color([1.0, 0.0, 0.0])  # red

    fd, tmp_path = tempfile.mkstemp(suffix=".pcd")
    os.close(fd)
    o3d.io.write_point_cloud(tmp_path, pcd_seg_cam)

    local_poses_cam, local_scores = sample_grasps(tmp_path, if_global=False)
    os.remove(tmp_path)

    if len(local_poses_cam) == 0:
        print("[WARNING] No grasps found on the cropped+segmented Gaussians.")
        return

    # print(f"[DEBUG] Sampler found {len(local_poses_cam)} candidate grasps in CAMERA frame.")

    # Debug: print top‐5 candidates in camera frame
    # for idx, (pose_cam, score) in enumerate(sorted(zip(local_poses_cam, local_scores),
    #                                                key=lambda x: -x[1])[:5]):
    #     print(f"[DEBUG] Candidate {idx} (score={score:.4f}) in CAMERA frame:\n{pose_cam}")

    # ─────────────────────────────────────────────────────────────────
    # 6.10) Convert each candidate → BASE, force vertical, distance‐filter
    # ─────────────────────────────────────────────────────────────────
    final_grasps = []
    # Precompute segmented points in base for collision check:
    ones_seg = np.ones((xyz_cam_seg_vis.shape[0], 1))
    homo_seg_cam = np.hstack([xyz_cam_seg_vis, ones_seg])   # (M_seg, 4)
    obj_pts_base = (W2B @ homo_seg_cam.T).T[:, :3]          # (M_seg, 3)

    for (pose_cam, score) in zip(local_poses_cam, local_scores):
        R_cam = pose_cam[:3, :3]
        t_cam = pose_cam[:3, 3]

        # a) Transform rotation into base
        R_base_raw = W2B[:3, :3] @ R_cam

        #    Force final approach = base –Z
        base_gaze = R_base_raw[:, 2].copy()   # camera’s +Z in base
        dir_xy = base_gaze.copy()
        dir_xy[2] = 0.0
        norm_xy = np.linalg.norm(dir_xy)
        if norm_xy < 1e-6:
            # fallback if camera is nearly vertical
            x_gripper = np.array([1.0, 0.0, 0.0])
        else:
            x_gripper = dir_xy / norm_xy   # gripper’s X in base

        z_gripper = np.array([0.0, 0.0, -1.0])
        y_gripper = np.cross(z_gripper, x_gripper)
        y_gripper /= np.linalg.norm(y_gripper)

        R_base_hacked = np.column_stack((x_gripper, y_gripper, z_gripper))

        # b) Compute translation in base
        t_base = (W2B @ np.hstack([t_cam, 1.0]))[:3]

        # c) Distance‐check: midpoint = t_cam + 0.05 * camera+Z
        t_mid_cam = t_cam + 0.05 * R_cam[:, 2]
        t_mid_base = (W2B @ np.hstack([t_mid_cam, 1.0]))[:3]
        dists = np.linalg.norm(obj_pts_base - t_mid_base, axis=1)
        if np.min(dists) < args.dist_thresh:
            # reject if too close/collision
            continue

        # d) Keep this candidate
        pose_base = np.eye(4, dtype=np.float64)
        pose_base[:3, :3] = R_base_hacked
        pose_base[:3, 3]  = t_base

        final_grasps.append((pose_base, score, pose_cam))

    # 6.11) Sort + keep top‐K
    final_grasps = sorted(final_grasps, key=lambda x: -x[1])[: args.topk]
    if not final_grasps:
        print("[WARNING] No grasps passed the distance/collision filter.")
        return

    best_pose_base, best_score, best_pose_cam = final_grasps[0]
    print("\n[+] Best sampled grasp pose (CAMERA frame):")
    print(f"{best_pose_cam}")
    print(f"[+] Score: {best_score:.4f}")

    # ─────────────────────────────────────────────────────────────────
    # 6.12) Visualize in CAMERA: cropped vs. segmented vs. best-sampled gripper
    # ─────────────────────────────────────────────────────────────────
    if args.camviz:
        print("[INFO] Launching CAMERA‐FRAME visualization (crop + segment + best sampled) …")
        cam_vis = o3d.visualization.Visualizer()
        cam_vis.create_window("CAMERA FRAME: Visible (gray), Segmented (red), Best Sampled Gripper")

        # (A) Cropped (visible) cloud in gray
        pcd_vis_cam = o3d.geometry.PointCloud()
        pcd_vis_cam.points = o3d.utility.Vector3dVector(xyz_cam_vis)
        pcd_vis_cam.paint_uniform_color([0.45, 0.45, 0.45])
        cam_vis.add_geometry(pcd_vis_cam)

        # (B) Cropped+segmented cloud in red
        pcd_seg_cam = o3d.geometry.PointCloud()
        pcd_seg_cam.points = o3d.utility.Vector3dVector(xyz_cam_seg_vis)
        pcd_seg_cam.paint_uniform_color([1.0, 0.0, 0.0])
        cam_vis.add_geometry(pcd_seg_cam)

        # (C) Draw the chosen gripper frame (in camera coords) using plot_gripper_pro_max:
        #     We know best_pose_cam is a 4×4 in camera frame.  The gripper‐frame mesh
        #     from plot_gripper_pro_max expects (center, rotation, outer_radius, inner_radius).
        center_cam = best_pose_cam[:3, 3]
        R_cam     = best_pose_cam[:3, :3]
        gripper_cam_mesh = plot_gripper_pro_max(
            center_cam,
            R_cam,
            0.08,   # outer “palm” radius (in meters)
            0.06    # inner “finger” radius (in meters)
        )
        cam_vis.add_geometry(gripper_cam_mesh)

        # (D) Draw camera origin triad (pure camera‐frame reference)
        cam_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        cam_vis.add_geometry(cam_origin)

        opt_cam = cam_vis.get_render_option()
        opt_cam.point_size       = args.point_size
        opt_cam.background_color = [0, 0, 0]
        cam_vis.run()
        cam_vis.destroy_window()

if __name__ == "__main__":
    main()
