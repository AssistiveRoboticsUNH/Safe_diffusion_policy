#!/usr/bin/env python3
"""
fixed_grasp_orientation_full_pipeline_true_lookat.py

Same overall flow as before, but the trajectory is generated so that at
each intermediate pose, the camera’s local –Z axis points directly at
the final grasp position (red candy). We then save all inverted poses
(so GaussianSplatRenderer can consume them) to raw_trajectory.npy.

Usage:
    python fixed_grasp_orientation_full_pipeline_true_lookat.py \
        --pose-file /path/to/poses/0.npy \
        --tsdf-ply /path/to/sparse/0/points3D.ply \
        --visualize \
        [--steps 50] [--dist-thresh 0.02] [--ee-pose …] [--object-query …] etc.
"""

import os
import sys
import argparse
import tempfile

import numpy as np
import torch
import open3d as o3d

from scipy.linalg import logm, expm
from spatialmath import SO3  # for rotation‐matrix utilities

# Add GraspSplats to Python path (no change):
sys.path.append("feature-splatting-inria")
sys.path.append(".")
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene, skip_feat_decoder
import featsplat_editor
from gaussian_edit import edit_utils
from grasping.grasping_utils import sample_grasps, plot_gripper_pro_max

# -----------------------------------------------------------------------------
# 1) In‐base “+90° about Y” and “180° about Z” rotations for hack (no change)
# -----------------------------------------------------------------------------
Ry_base = SO3.Ry(np.pi / 2).data[0]
Rz_base = SO3.Rz(np.pi).data[0]

# -----------------------------------------------------------------------------
# 2) Utility: write camera‐frame PCD to a temporary PCD on disk (no change)
# -----------------------------------------------------------------------------
def write_tmp_pcd(pcd: o3d.geometry.PointCloud) -> str:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.02))
    pcd.orient_normals_consistent_tangent_plane(100)
    fd, path = tempfile.mkstemp(suffix=".pcd")
    os.close(fd)
    o3d.io.write_point_cloud(path, pcd)
    return path

# -----------------------------------------------------------------------------
# 3) “Look-at” so that camera’s local –Z points at the target
# -----------------------------------------------------------------------------
import numpy as np

def lookat_posZ(cam_origin: np.ndarray,
                target_point: np.ndarray,
                world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """
    Build a 3×3 rotation R such that:
      • camera’s local +Z axis points directly from cam_origin → target_point
      • local +X = normalize(world_up × +Z) (unless nearly collinear)
      • local +Y = (+Z) × (+X)

    Args:
      cam_origin   : (3,) camera location in BASE coords
      target_point : (3,) target (red candy) location in BASE coords
      world_up    : (3,) world “up” vector (default = [0,0,1])

    Returns:
      R (3×3) whose columns are [x_cam, y_cam, z_cam], 
      where z_cam = (target_point – cam_origin)/||…|| (i.e. local +Z toward target).
    """
    # 1) Vector from camera → candy
    v = target_point - cam_origin
    dist = np.linalg.norm(v)
    if dist < 1e-6:
        return np.eye(3)

    # 2) z_cam = + (target – origin)/||…|| 
    z_cam = v / dist

    # 3) x_cam = normalize(world_up × z_cam), unless collinear
    dot = float(world_up.dot(z_cam))
    if abs(dot) > 0.999:
        tmp = np.array([1.0, 0.0, 0.0])
        x_cam = np.cross(tmp, z_cam)
    else:
        x_cam = np.cross(world_up, z_cam)
    x_cam /= np.linalg.norm(x_cam)

    # 4) y_cam = z_cam × x_cam  (right‐handed)
    y_cam = np.cross(z_cam, x_cam)
    y_cam /= np.linalg.norm(y_cam)

    # 5) Stack columns [x_cam, y_cam, z_cam]
    return np.column_stack((x_cam, y_cam, z_cam))


# -----------------------------------------------------------------------------
# 4) Convert raw camera‐pose → transformed camera‐pose (camera→base), no change
# -----------------------------------------------------------------------------
def convert_raw_to_transformed(raw: np.ndarray, W2B: np.ndarray, tsdf_ply: str) -> (np.ndarray, str):
    """
    raw:      4×4 camera‐pose loaded from .npy (either “base→cam” or “cam→world”).
    W2B:      4×4 world→base transform (same as pipeline).
    tsdf_ply: path to TSDF PLY (camera coords) for table plane fit.

    Returns:
      M_trans: 4×4 camera→base pose (M0), chosen as inv(raw) or W2B@raw so that
               the camera origin stands above the table. Rotation is unchanged.
      mode:    "inv"  if M_trans = inv(raw)   (raw was base→cam)
               "w2b"  if M_trans = W2B @ raw  (raw was cam→world)
    """
    if raw.shape != (4, 4):
        raise ValueError(f"Expected raw to be 4×4, got {raw.shape}")

    try:
        cand1 = np.linalg.inv(raw)
    except np.linalg.LinAlgError:
        cand1 = None
    cand2 = W2B @ raw

    tsdf = o3d.io.read_point_cloud(tsdf_ply)
    tsdf.transform(W2B)
    plane_model, inliers = tsdf.segment_plane(
        distance_threshold=0.005, ransac_n=3, num_iterations=1000
    )
    a, b, c, d = plane_model
    if c < 0:
        a, b, c, d = -a, -b, -c, -d

    inlier_pts = np.asarray(tsdf.points)[inliers]
    table_z = float(np.mean(inlier_pts[:, 2]))

    valid1 = (cand1 is not None) and (cand1[2, 3] > table_z + 0.01)
    valid2 = (cand2[2, 3] > table_z + 0.01)

    if valid1 and not valid2:
        return cand1.copy(), "inv"
    elif valid2 and not valid1:
        return cand2.copy(), "w2b"
    elif valid1 and valid2:
        if cand1[2, 3] >= cand2[2, 3]:
            return cand1.copy(), "inv"
        else:
            return cand2.copy(), "w2b"
    else:
        raise RuntimeError(
            f"Neither inv(raw) (z={cand1[2,3] if cand1 is not None else 'NaN'}) "
            f"nor W2B@raw (z={cand2[2,3]}) is above table_z={table_z:.3f}."
        )

# -----------------------------------------------------------------------------
# 5) Convert a list of camera→base poses → list of raw poses (no change)
# -----------------------------------------------------------------------------
def convert_transformed_to_raw_motion(traj_base: list, W2B: np.ndarray, mode: str) -> list:
    """
    traj_base: list of 4×4 camera→base poses [M₀..Mₙ].
    W2B:       4×4 world→base.
    mode:      "inv"  → rawᵢ = inv(Mᵢ)
               "w2b"  → rawᵢ = inv(W2B) @ Mᵢ

    Returns:
      traj_raw: list of 4×4 raw poses [raw₀..rawₙ], matching the original convention.
    """
    traj_raw = []
    if mode == "inv":
        for M_i in traj_base:
            traj_raw.append(np.linalg.inv(M_i))
    elif mode == "w2b":
        W2B_inv = np.linalg.inv(W2B)
        for M_i in traj_base:
            traj_raw.append(W2B_inv @ M_i)
    else:
        raise ValueError(f"Mode must be 'inv' or 'w2b', got '{mode}'")
    return traj_raw

# -----------------------------------------------------------------------------
# 6) Argument parsing (no change)
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(__doc__)

    ModelParams(parser)
    parser.set_defaults(model_path="/home/carl_lab/akash/gaussian-splatting/output/franka_outputs")
    parser.set_defaults(source_path="/home/carl_lab/Riad/graspbility/GraspSplats/franka_data")
    PipelineParams(parser)
    OptimizationParams(parser)

    parser.add_argument(
        "-q", "--object-query", default="Red candy",
        help="CLIP object query (default: 'Red candy')"
    )
    parser.add_argument(
        "-n", "--negative-query", default="Green candy",
        help="Optional negative CLIP query (default: 'Green candy')"
    )
    parser.add_argument(
        "-p", "--part-query", default="Red candy",
        help="Optional CLIP part query (default: 'Red candy')"
    )

    parser.add_argument(
        "--obj-pos-thresh", type=float, default=0.75,
        help="Positive‐query similarity threshold (default: 0.75)"
    )
    parser.add_argument(
        "--obj-neg-thresh", type=float, default=0.7,
        help="Negative‐query similarity threshold (default: 0.7)"
    )
    parser.add_argument(
        "--part-thresh", type=float, default=0.7,
        help="Part‐query similarity threshold (default: 0.7)"
    )

    parser.add_argument(
        "--tsdf-ply",
        default="/home/carl_lab/Riad/graspbility/GraspSplats/franka_data/sparse/0/points3D.ply",
        help="Filename of TSDF PLY (in camera coords)"
    )

    parser.add_argument(
        "--topk", type=int, default=1,
        help="Number of top grasps to keep (default: 1)"
    )
    parser.add_argument(
        "--dist-thresh", type=float, default=0.02,
        help="Min‐distance threshold (default: 0.02)"
    )

    parser.add_argument(
        "--visualize", action="store_true",
        help="If set, show Open3D visualization"
    )

    parser.add_argument(
        "--ee-pose", nargs=16, type=float, default=None,
        help="Initial EE pose as 16 floats (row-major 4×4) in base frame."
    )

    parser.add_argument(
        "--pose-file", type=str,
        default="/home/carl_lab/Riad/graspbility/GraspSplats/franka_data/poses/2.npy",
        help="Path to a .npy file containing a 4×4 camera-pose (raw)."
    )

    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of trajectory steps (default: 50)."
    )

    return get_combined_args(parser)

# -----------------------------------------------------------------------------
# 7) Main pipeline
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    # 7.1) Load GaussianModel & Scene (camera frame) — no change
    g = GaussianModel(args.sh_degree, args.distill_feature_dim)
    g.training_setup(args)
    Scene(args, g, load_iteration=-1, shuffle=False)
    pts_cam_all = g.get_xyz.detach().cpu().numpy()  # (N×3) in camera frame

    # 7.2) CLIP segmentation on Gaussians (camera frame) — no change
    decoder = skip_feat_decoder(args.distill_feature_dim, part_level=True).cuda()
    decoder.load_state_dict(torch.load(
        os.path.join(args.model_path, "feat_decoder.pth"), map_location="cpu"
    ))
    decoder.eval()
    segm = featsplat_editor.clip_segmenter(g, decoder)

    with torch.no_grad():
        sim_pos = segm.compute_similarity_one(args.object_query, level="object")
    if torch.is_tensor(sim_pos):
        sim_pos = sim_pos.cpu().numpy()
    mask = sim_pos > args.obj_pos_thresh

    if args.negative_query:
        with torch.no_grad():
            sim_neg = segm.compute_similarity_one(args.negative_query, level="object")
        if torch.is_tensor(sim_neg):
            sim_neg = sim_neg.cpu().numpy()
        mask &= (sim_neg < args.obj_neg_thresh)

    if args.part_query:
        with torch.no_grad():
            sim_part = segm.compute_similarity_one(args.part_query, level="part")
        if torch.is_tensor(sim_part):
            sim_part = sim_part.cpu().numpy()
        mask &= (sim_part > args.part_thresh)

    mask = edit_utils.cluster_instance(pts_cam_all, mask)
    mask = edit_utils.flood_fill(pts_cam_all, mask)

    # 7.3) Build segmented-object PCD in camera frame — no change
    gauss_fg = edit_utils.select_gaussians(g, mask)
    xyz_cam_fg = gauss_fg._xyz.detach().cpu().numpy()
    pcd_fg_cam = o3d.geometry.PointCloud()
    pcd_fg_cam.points = o3d.utility.Vector3dVector(xyz_cam_fg)
    pcd_fg_cam.paint_uniform_color([1.0, 0.0, 0.0])

    scene_pcd_cam = o3d.io.read_point_cloud(os.path.join(args.source_path, args.tsdf_ply))
    out_dir = os.path.join(args.model_path, "point_cloud_for_grasp")
    os.makedirs(out_dir, exist_ok=True)
    overlay_cam = scene_pcd_cam + pcd_fg_cam
    o3d.io.write_point_cloud(os.path.join(out_dir, "segmentation_overlay_cam.ply"), overlay_cam)
    print("[+] Saved camera-frame overlay (scene + red Gaussians)")

    # 7.4) Sample grasps in camera frame — no change
    tmp_cam = write_tmp_pcd(pcd_fg_cam)
    local_poses_cam, local_scores = sample_grasps(tmp_cam, if_global=False)
    os.remove(tmp_cam)

    if len(local_poses_cam) == 0:
        print("[WARNING] No grasps found in camera frame")
        return

    # -----------------------------------------------------------------------------
    # 7.5) FIX-ORIENTATION & transform each grasp to BASE (no change)
    # -----------------------------------------------------------------------------
    W2B = np.array([
        [ 0.9866801158315599,   0.12442503082301001,  -0.10478912504316718,  0.3730359970326709 ],
        [ 0.16035477915117624, -0.8522923025192382,    0.49787968011341105, -0.006907504865351453 ],
        [-0.027362270117755733, -0.5080514174482224,  -0.8608920161105309,   0.4207423250737201 ],
        [ 0.0,                  0.0,                   0.0,                  1.0               ],
    ], dtype=np.float64)

    final_grasps = []
    ones = np.ones((xyz_cam_fg.shape[0], 1))
    homo_fg = np.hstack([xyz_cam_fg, ones])      # (M×4) in camera homogeneous
    obj_pts_base = (W2B @ homo_fg.T).T[:, :3]     # (M×3) in BASE

    for (pose_cam, score) in zip(local_poses_cam, local_scores):
        R_cam = pose_cam[:3, :3]
        t_cam = pose_cam[:3, 3]

        R_base_raw = W2B[:3, :3] @ R_cam
        R_base_hacked = R_base_raw @ Ry_base
        xdir = R_base_hacked[:, 0]
        if np.dot(xdir, np.array([1.0, 0.0, 0.0])) < 0.0:
            R_base_hacked = R_base_hacked @ Rz_base

        pose_base = np.eye(4)
        pose_base[:3, :3] = R_base_hacked
        t_base = (W2B @ np.hstack([t_cam, 1.0]))[:3]
        pose_base[:3, 3] = t_base

        t_mid_cam = t_cam + 0.05 * pose_cam[:3, 2]
        t_mid_base = (W2B @ np.hstack([t_mid_cam, 1.0]))[:3]
        dists = np.linalg.norm(obj_pts_base - t_mid_base, axis=1)
        if np.min(dists) < args.dist_thresh:
            continue

        final_grasps.append((pose_base, score))

    final_grasps = sorted(final_grasps, key=lambda x: -x[1])[: args.topk]
    if not final_grasps:
        print("[WARNING] No grasps passed filtering in base frame")
        return

    best_pose_base, best_score = final_grasps[0]

    # 7.6) Fit table plane in BASE (for sampling M0) — no change
    scene_pcd_cam.transform(W2B)
    plane_model, inliers = scene_pcd_cam.segment_plane(
        distance_threshold=0.005, ransac_n=3, num_iterations=1000
    )
    a, b, c, d = plane_model
    if c < 0:
        a, b, c, d = -a, -b, -c, -d

    inlier_pts = np.asarray(scene_pcd_cam.points)[inliers]
    table_z_base = float(np.mean(inlier_pts[:, 2]))

    # 7.7) Determine initial M0 (camera→base) — no change
    raw = np.load(args.pose_file)
    if raw.shape != (4, 4):
        raise RuntimeError(f"Pose file shape {raw.shape}, expected (4×4)")

    try:
        cand1 = np.linalg.inv(raw)
    except np.linalg.LinAlgError:
        cand1 = None
    cand2 = W2B @ raw

    valid1 = (cand1 is not None) and (cand1[2, 3] > table_z_base + 0.01)
    valid2 = (cand2[2, 3] > table_z_base + 0.01)

    if valid1 and not valid2:
        M0 = cand1.copy()
        mode = "inv"
        print(f"[+] Using inv(raw) as M0; M0.z = {cand1[2,3]:.3f} (table_z={table_z_base:.3f})")
    elif valid2 and not valid1:
        M0 = cand2.copy()
        mode = "w2b"
        print(f"[+] Using W2B@raw as M0; M0.z = {cand2[2,3]:.3f} (table_z={table_z_base:.3f})")
    elif valid1 and valid2:
        if cand1[2, 3] >= cand2[2, 3]:
            M0 = cand1.copy()
            mode = "inv"
            print(f"[+] Both inv(raw) & W2B@raw above table; picked inv(raw) (z={cand1[2,3]:.3f})")
        else:
            M0 = cand2.copy()
            mode = "w2b"
            print(f"[+] Both inv(raw) & W2B@raw above table; picked W2B@raw (z={cand2[2,3]:.3f})")
    else:
        raise RuntimeError(
            f"Neither inv(raw) (z={cand1[2,3] if cand1 is not None else 'NaN'}) nor "
            f"W2B@raw (z={cand2[2,3]}) above table_z={table_z_base:.3f}."
        )

    # If user provided --ee-pose, overwrite M0 (skip raw logic)—no change

    # -----------------------------------------------------------------------------
    # 7.8) Build a smooth, “always look-at” trajectory of length N = args.steps
    # -----------------------------------------------------------------------------
    N = args.steps
    p0 = M0[:3, 3].copy()                # initial camera position in BASE
    pg = best_pose_base[:3, 3].copy()    # target (red candy) in BASE

    traj_base = []  # will hold each M_i^{base}
    for i in range(N):
        alpha = i / (N - 1)              # linear interpolation factor in [0,1]
        # 1) interpolate camera origin
        p_i = (1 - alpha) * p0 + alpha * pg

        # 2) recompute R_i so that camera’s local +Z points at pg
        R_i = lookat_posZ(p_i, pg, world_up=np.array([0.0, 0.0, 1.0]))
        # M_i[:3, :3] = R_i

        # 3) assemble M_i^{base}
        M_i = np.eye(4, dtype=np.float64)
        M_i[:3, :3] = R_i
        M_i[:3, 3] = p_i
        traj_base.append(M_i)

    # -----------------------------------------------------------------------------
    # 7.9) Convert each M_i^{base} → raw convention and save to disk
    # -----------------------------------------------------------------------------
    if mode in ("inv", "w2b"):
        traj_raw = convert_transformed_to_raw_motion(traj_base, W2B, mode)
    else:
        traj_raw = []

    # Save all raw poses into raw_trajectory.npy
    if traj_raw:
        raw_arr = np.stack(traj_raw, axis=0)   # shape: (N, 4, 4)
        save_path = os.path.join(args.model_path, "raw_trajectory.npy")
        np.save(save_path, raw_arr)
        print(f"[+] Saved raw trajectory (N={N}) to '{save_path}'")

    # -------------------------------------
    # 7.10) (Optional) Print or visualize results
    # -------------------------------------
    if args.visualize:
        #  Render each base-pose frame in Open3D exactly as before:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Draw the static TSDF (base)
        vis.add_geometry(scene_pcd_cam)

        # Draw segmented object Gaussians in red
        pcd_obj_base = o3d.geometry.PointCloud()
        pcd_obj_base.points = o3d.utility.Vector3dVector(obj_pts_base)
        pcd_obj_base.paint_uniform_color([1.0, 0.0, 0.0])
        vis.add_geometry(pcd_obj_base)

        # Draw the best grasp itself
        vis.add_geometry(plot_gripper_pro_max(
            best_pose_base[:3,3], best_pose_base[:3,:3], 0.08, 0.06
        ))

        # Draw all camera poses along the trajectory (small triads)
        for M_i in traj_base:
            small_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            small_frame.transform(M_i)
            vis.add_geometry(small_frame)

        # Draw line segments between them (green)
        traj_pts = [M_i[:3,3] for M_i in traj_base]
        lines = [[i, i+1] for i in range(len(traj_pts)-1)]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(traj_pts),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector([[0.0,1.0,0.0]] * len(lines))
        vis.add_geometry(ls)

        # Draw a bigger triad at the final grasp (size=0.06)
        end_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        end_frame.transform(best_pose_base)
        vis.add_geometry(end_frame)

        vis.run()
        vis.destroy_window()

    # 7.11) Print best grasp
    print("\n[+] Best grasp pose in BASE frame (4×4):")
    print(np.array_str(best_pose_base, precision=6, suppress_small=True))
    print(f"[+] Score: {best_score:.4f}")

if __name__ == "__main__":
    main()
