#!/usr/bin/env python3

import os
import sys
import argparse
from types import SimpleNamespace
import math
import tempfile 

import numpy as np
import torch
import open3d as o3d
import cv2
import pyrealsense2 as rs
from PIL import Image

# Imports for Transformer-based Segmentation
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Original Imports
import panda_py
from panda_py import libfranka
from scipy.spatial.transform import Rotation, Slerp

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ───────────────────────────────────────────────────────────────────────
# 1) Gaussian Splatting and Camera imports
# ───────────────────────────────────────────────────────────────────────
sys.path.append("feature-splatting-inria")
sys.path.append(".")
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene, skip_feat_decoder
import featsplat_editor
from gaussian_edit import edit_utils

# ───────────────────────────────────────────────────────────────────────
# 2) Grasp-sampling imports
# ───────────────────────────────────────────────────────────────────────
from grasping.grasping_utils import sample_grasps, plot_gripper_pro_max

# ───────────────────────────────────────────────────────────────────────
# 3) Constants and Helper Functions
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


def project_graphics(p, fx, fy, cx, cy):
    # This function remains unchanged
    X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
    Z_safe = np.maximum(Z, 1e-8)
    u = fx * (X / Z_safe) + cx
    v = -fy * (Y / Z_safe) + cy
    return u, v, Z

def in_img(u, v, z, W, H):
    # This function remains unchanged
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    return (z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)


def visualize_geometries(geometries, window_name="Open3D", point_size=1.5):
    # This function remains unchanged
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name)
    for geom in geometries:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    print(f"\n[INFO] Running visualization '{window_name}'. Close the window to continue.")
    vis.run()
    vis.destroy_window()

def draw_grasp_pose(image, grasp_pose_cam, intrinsics, is_best=True):
    # This function remains unchanged
    overlay = image.copy()
    alpha = 0.5
    if is_best:
        palm_color, finger_color, approach_color = (0, 255, 0), (255, 0, 0), (0, 0, 255)
    else:
        palm_color, finger_color, approach_color = (150, 150, 150), (180, 180, 180), (200, 200, 200)
        alpha = 0.25
    w, t, l = 0.04, 0.005, 0.05
    palm_pts_local = np.array([[0, -w, -t], [0, w, -t], [0, w, t], [0, -w, t]])
    finger1_pts_local = np.array([[0, w, -t], [0, w, t], [l, w, t], [l, w, -t]])
    finger2_pts_local = np.array([[0, -w, -t], [0, -w, t], [l, -w, t], [l, -w, -t]])
    approach_start_local = np.array([0, 0, 0])
    approach_end_local = np.array([l + 0.02, 0, 0])
    polygons = [(palm_pts_local, palm_color), (finger1_pts_local, finger_color), (finger2_pts_local, finger_color)]
    R, t_vec = grasp_pose_cam[:3, :3], grasp_pose_cam[:3, 3]
    for points_local, color in polygons:
        points_cam = (R @ points_local.T).T + t_vec
        projected_pts, is_visible = [], True
        for p in points_cam:
            if p[2] <= 0: is_visible = False; break
            u, v = int(intrinsics.fx * p[0] / p[2] + intrinsics.ppx), int(intrinsics.fy * p[1] / p[2] + intrinsics.ppy)
            projected_pts.append([u, v])
        if is_visible:
            cv2.fillPoly(overlay, [np.array(projected_pts, dtype=np.int32)], color)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    app_start_cam, app_end_cam = (R @ approach_start_local) + t_vec, (R @ approach_end_local) + t_vec
    if app_start_cam[2] > 0 and app_end_cam[2] > 0:
        u1, v1 = int(intrinsics.fx * app_start_cam[0] / app_start_cam[2] + intrinsics.ppx), int(intrinsics.fy * app_start_cam[1] / app_start_cam[2] + intrinsics.ppy)
        u2, v2 = int(intrinsics.fx * app_end_cam[0] / app_end_cam[2] + intrinsics.ppx), int(intrinsics.fy * app_end_cam[1] / app_end_cam[2] + intrinsics.ppy)
        cv2.line(image, (u1, v1), (u2, v2), approach_color, 3)
    return image

def generate_trajectory(start_pose, end_pose, num_waypoints):
    # This function remains unchanged
    R_start, t_start = start_pose[:3, :3], start_pose[:3, 3]
    R_end, t_end = end_pose[:3, :3], end_pose[:3, 3]
    key_rots = Rotation.from_matrix([R_start, R_end])
    slerp = Slerp([0, 1], key_rots)
    times = np.linspace(0, 1, num_waypoints)
    interp_rots = slerp(times).as_matrix()
    interp_trans = np.array([np.linspace(s, e, num_waypoints) for s, e in zip(t_start, t_end)]).T
    trajectory = [np.identity(4) for _ in range(num_waypoints)]
    for i in range(num_waypoints):
        trajectory[i][:3, :3] = interp_rots[i]
        trajectory[i][:3, 3] = interp_trans[i]
    return trajectory


def parse_args():
    # This function remains unchanged
    p = argparse.ArgumentParser(__doc__)
    ModelParams(p)
    p.set_defaults(model_path="/home/franka_deoxys/riad/GraspSplats/franka_outputs", source_path="/home/franka_deoxys/riad/GraspSplats/franka_data")
    PipelineParams(p)
    OptimizationParams(p)
    p.add_argument("-q", "--object-query", default="the red candy", help="CLIP object query")
    p.add_argument("-n", "--negative-query", default="the green candy", help="Optional negative CLIP query for live segmentation.")
    p.add_argument("--obj-pos-thresh", type=float, default=0.70)
    p.add_argument("--obj-neg-thresh", type=float, default=0.7)
    p.add_argument("--live-seg-thresh", type=float, default=0.5)
    p.add_argument("-p", "--part-query", default="the red candy", help="Optional CLIP part query.")
    p.add_argument("--part-thresh", type=float, default=0.7)
    p.add_argument("--rgb-file", default="images/8.png")
    p.add_argument("--intrinsics", default="rgb_intrinsics.npz")
    p.add_argument("--no-debug-plots", action="store_true")
    p.add_argument("--point-size", type=float, default=1.4)
    p.add_argument("--camviz", action="store_true")
    p.add_argument("--vertical-thresh", type=float, default=0.7)
    p.add_argument("--roll-tolerance", type=float, default=0.3)
    p.add_argument("--clearance", type=float, default=0.02)
    p.add_argument("--generate-trajectory", action="store_true")
    p.add_argument("--trajectory-file", type=str, default="trajectory.npy")
    p.add_argument("--pre-grasp-offset", type=float, default=0.12)
    return get_combined_args(p)


def main():
    args = parse_args()

    # Part 0: Robot connection
    print("[INFO] Connecting to robot to get current pose...")
    try:
        hostname = "172.16.0.2"
        panda = panda_py.Panda(hostname)
        robot_start_pose = panda.get_pose()
    except Exception as e:
        print(f"[ERROR] Failed to connect to robot or get its pose: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Part 1: Model Loading & Initial Segmentation
    print("\n[INFO] Part 1: Loading model and segmenting object...")
    g = GaussianModel(args.sh_degree, args.distill_feature_dim)
    g.training_setup(args)
    Scene(args, g, load_iteration=-1, shuffle=False)
    xyz_world = g.get_xyz.detach().cpu().numpy()
    cam_in_world_pose_static = np.linalg.inv(W2B) @ robot_start_pose @ gTc
    world2cam_static = np.linalg.inv(cam_in_world_pose_static)
    xyz_cam0 = (world2cam_static[:3, :3] @ xyz_world.T).T + world2cam_static[:3, 3]
    d = np.load(os.path.join(args.source_path, args.intrinsics))
    fx, fy, cx, cy = float(d["fx"]), float(d["fy"]), float(d["ppx"]), float(d["ppy"])
    H, W, _ = cv2.imread(os.path.join(args.source_path, args.rgb_file)).shape
    W_full, H_full = int(round(2 * cx)), int(round(2 * cy))
    crop_left, crop_top = max((W_full - W)//2, 0), max((H_full - H)//2, 0)
    if crop_left or crop_top: cx -= crop_left; cy -= crop_top
    variants = {(1, 1): xyz_cam0, (1, -1): xyz_cam0 * [1,1,-1], (-1, 1): xyz_cam0 * [1,-1,1], (-1,-1): xyz_cam0 * [1,-1,-1]}
    best_cnt, best_key, best_xyz, visible_mask = -1, None, None, None
    for flips, pts in variants.items():
        u, v, z = project_graphics(pts, fx, fy, cx, cy)
        u -= crop_left; v -= crop_top
        mask_vis = in_img(u, v, z, W, H)
        if mask_vis.sum() > best_cnt: best_cnt, best_key, best_xyz, visible_mask = mask_vis.sum(), flips, pts, mask_vis.copy()
    decoder = skip_feat_decoder(args.distill_feature_dim, part_level=True).cuda()
    decoder.load_state_dict(torch.load(os.path.join(args.model_path, "feat_decoder.pth"), map_location="cpu"))
    decoder.eval()
    segm = featsplat_editor.clip_segmenter(g, decoder)
    with torch.no_grad(): sim_pos = segm.compute_similarity_one(args.object_query, level="object")
    mask_pos = (sim_pos > args.obj_pos_thresh)
    # The negative query for GS model segmentation remains as before.
    if args.negative_query and args.obj_neg_thresh:
        with torch.no_grad(): sim_neg = segm.compute_similarity_one(args.negative_query, level="object")
        mask_pos &= (sim_neg < args.obj_neg_thresh)
    segmented_visible_mask = mask_pos & visible_mask
    segmented_visible_mask = edit_utils.cluster_instance(best_xyz, segmented_visible_mask)
    segmented_visible_mask = edit_utils.flood_fill(best_xyz, segmented_visible_mask)
    object_points = best_xyz[segmented_visible_mask]
    if len(object_points) < 100:
        print("[ERROR] Segmentation resulted in too few points. Cannot track.")
        sys.exit(1)
    
    pcd_object_model = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_points))
    voxel_size = 0.005
    pcd_object_model, _ = pcd_object_model.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_object_model = pcd_object_model.voxel_down_sample(voxel_size)
    pcd_object_model.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    print("\n[INFO] Loading live segmentation model (CLIPSeg)...")
    live_seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    live_seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    live_seg_model.to(device)
    print(f"CLIPSeg model loaded successfully on '{device}'.")

    print("\n[INFO] Part 2: Initializing live camera for tracking and dynamic grasp planning...")
    pipeline = rs.pipeline()
    config = rs.config()
    live_cam_W, live_cam_H = 1280, 720
    config.enable_stream(rs.stream.depth, live_cam_W, live_cam_H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, live_cam_W, live_cam_H, rs.format.bgr8, 30)
    
    base_T_cam = robot_start_pose @ gTc

    try:
        profile = pipeline.start(config)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
        camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        align = rs.align(rs.stream.color)
        
        final_transform = np.identity(4)
        is_tracking = False
        
        frame_count = 0
        grasp_poses_to_draw = [] 
        last_good_grasp_base = None
        last_obj_pts_base = None
        
        print(f"[INFO] Starting live loop... Looking for '{args.object_query}'. Press 'q' to quit.")
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame: continue

            color_image_bgr = np.asanyarray(color_frame.get_data())
            
            color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
            input_image_pil = Image.fromarray(color_image_rgb)

            # --- NEW: Live Negative Query Logic ---
            prompts = [args.object_query]
            if args.negative_query:
                prompts.append(args.negative_query)

            inputs = live_seg_processor(text=prompts, images=[input_image_pil] * len(prompts), return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = live_seg_model(**inputs)
            
            # Process the heatmaps
            logits = outputs.logits.cpu()
            mask_probs_positive = torch.sigmoid(logits[0]).squeeze()

            if len(prompts) > 1:
                mask_probs_negative = torch.sigmoid(logits[1]).squeeze()
                # Subtract the negative heatmap from the positive one
                final_mask_probs = torch.clamp(mask_probs_positive - mask_probs_negative, min=0)
            else:
                final_mask_probs = mask_probs_positive

            mask_pil = Image.fromarray(final_mask_probs.numpy())
            # --- END of Negative Query Logic ---

            mask_resized = mask_pil.resize((live_cam_W, live_cam_H))
            binary_mask = (np.array(mask_resized) > args.live_seg_thresh).astype(np.uint8)

            depth_image_u16 = np.asanyarray(depth_frame.get_data())
            depth_image_m = depth_image_u16.astype(np.float32) * depth_scale
            depth_image_m[binary_mask == 0] = 0
            depth_image_m[depth_image_m > 2.0] = 0

            o3d_depth = o3d.geometry.Image(depth_image_m)
            o3d_color_for_pcd = o3d.geometry.Image(color_image_rgb)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color_for_pcd, o3d_depth, depth_scale=1.0, depth_trunc=2.0, convert_rgb_to_intensity=False)
            pcd_live_object_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intrinsics)

            if pcd_live_object_raw.has_points():
                pcd_live_object_for_icp = pcd_live_object_raw.voxel_down_sample(voxel_size)
                pcd_live_object_for_icp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

                init_transform_for_icp = final_transform

                # If track is lost OR if we haven't started tracking yet, try to acquire the object
                if not is_tracking or (is_tracking and pcd_live_object_for_icp.get_center()[2] == 0): # Re-acquire if detection is empty
                    is_tracking = False # Ensure we are in searching mode
                    centroid_gs = pcd_object_model.get_center()
                    centroid_live = pcd_live_object_for_icp.get_center()
                    init_transform_for_icp = np.identity(4)
                    init_transform_for_icp[:3, 3] = centroid_live - centroid_gs
                
                reg_result = o3d.pipelines.registration.registration_icp(pcd_object_model, pcd_live_object_for_icp, voxel_size * 2.0, init_transform_for_icp, o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
                
                if reg_result.fitness > 0.15:
                    final_transform = reg_result.transformation
                    if not is_tracking:
                        print("[INFO] Object acquired. Tracking locked.")
                        is_tracking = True
                else:
                    if is_tracking:
                        print("[WARNING] Tracking quality low, returning to search mode.")
                    is_tracking = False
            else:
                if is_tracking:
                    print("[WARNING] Object lost from view, returning to search mode.")
                is_tracking = False

            # Grasp planning logic
            if frame_count % 20 == 0 and is_tracking:
                # Use the live point cloud for the most accurate grasp planning
                if pcd_live_object_raw.has_points():
                    xyz_cam_current = np.asarray(pcd_live_object_raw.points)
                    with tempfile.NamedTemporaryFile(suffix=".pcd", delete=True) as tmp:
                        o3d.io.write_point_cloud(tmp.name, pcd_live_object_raw)
                        local_poses_cam, _ = sample_grasps(tmp.name, if_global=False)
                    
                    valid_grasps = []
                    if local_poses_cam:
                        object_bbox = pcd_live_object_raw.get_axis_aligned_bounding_box()
                        object_bbox.scale(1.1, object_bbox.get_center())
                        obj_pts_base = (base_T_cam @ np.hstack([xyz_cam_current, np.ones((xyz_cam_current.shape[0], 1))]).T).T[:, :3]
                        
                        for pose_cam in local_poses_cam:
                            if len(object_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector([pose_cam[:3, 3]]))) == 0: continue
                            R_base = base_T_cam[:3, :3] @ pose_cam[:3, :3]
                            if np.dot(R_base[:, 0], [0, 0, -1.0]) < args.vertical_thresh: continue
                            if abs(R_base[:, 1][2]) > args.roll_tolerance: continue
                            valid_grasps.append({'base': base_T_cam @ pose_cam, 'cam': pose_cam})
                    
                    grasp_poses_to_draw = [g['cam'] for g in valid_grasps[:5]] if valid_grasps else []
                    if valid_grasps:
                        last_good_grasp_base = valid_grasps[0]['base']
                        last_obj_pts_base = obj_pts_base.copy()

            if not is_tracking:
                grasp_poses_to_draw = []

            # Visualization
            blended_image = color_image_bgr.copy()
            if np.any(binary_mask):
                overlay_color = np.array([0, 0, 255])
                overlay = np.zeros_like(blended_image, dtype=np.uint8)
                overlay[binary_mask == 1] = overlay_color
                blended_image = cv2.addWeighted(blended_image, 1.0, overlay, 0.4, 0)

            if is_tracking:
                pcd_tracked_object_vis = o3d.geometry.PointCloud(pcd_object_model).transform(final_transform)
                points_3d = np.asarray(pcd_tracked_object_vis.points)
                if points_3d.shape[0] > 0:
                    valid_z = points_3d[:, 2] > 0.1
                    points_3d_valid = points_3d[valid_z]
                    if points_3d_valid.shape[0] > 0:
                        projected_points, _ = cv2.projectPoints(points_3d_valid, np.zeros(3), np.zeros(3), camera_matrix, None)
                        depths = points_3d_valid[:, 2]
                        sorted_indices = np.argsort(depths)[::-1]
                        for i in sorted_indices:
                            pt_2d = (int(projected_points[i][0][0]), int(projected_points[i][0][1]))
                            depth = depths[i]
                            radius_scale_factor = 1.0
                            radius = int(radius_scale_factor / depth)
                            if 0 <= pt_2d[0] < live_cam_W and 0 <= pt_2d[1] < live_cam_H:
                                cv2.circle(blended_image, pt_2d, max(1, radius), (0, 255, 0), -1)

            if grasp_poses_to_draw:
                blended_image = draw_grasp_pose(blended_image, grasp_poses_to_draw[0], intr, is_best=True)
                for pose in grasp_poses_to_draw[1:]:
                    blended_image = draw_grasp_pose(blended_image, pose, intr, is_best=False)

            cv2.imshow("Live Tracking and Grasping - Press 'q' to quit", blended_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    finally:
        print("\nStopping camera pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()
    
    # Final Pose and Trajectory Generation
    if last_good_grasp_base is not None and last_obj_pts_base is not None:
        grasp_pose_in_research_convention = last_good_grasp_base
        T_conversion = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        final_robot_pose_imprecise_height = grasp_pose_in_research_convention @ T_conversion
        min_z_in_base = np.min(last_obj_pts_base[:, 2])
        final_robot_pose = final_robot_pose_imprecise_height.copy()
        final_robot_pose[2, 3] = min_z_in_base + args.clearance
        print("\n" + "="*50)
        print("[+] Final Robot Grasp Pose (Robot Base Frame, Z-Forward, Height Corrected)")
        print("="*50)
        print(f"Final Z-height set to {final_robot_pose[2, 3]:.4f} m for table clearance.")
        np.set_printoptions(precision=4, suppress=True)
        print(final_robot_pose)
        print("="*50 + "\n")
        if args.generate_trajectory:
            print(f"[INFO] Generating trajectory from current pose to grasp pose...")
            pre_grasp_pose = final_robot_pose.copy()
            pre_grasp_pose[2, 3] += args.pre_grasp_offset
            traj_phase1 = generate_trajectory(robot_start_pose, pre_grasp_pose, num_waypoints=30)
            traj_phase2 = generate_trajectory(pre_grasp_pose, final_robot_pose, num_waypoints=10)
            full_trajectory = np.array(traj_phase1 + traj_phase2)
            np.save(args.trajectory_file, full_trajectory)
            print(f"[SUCCESS] Trajectory with {len(full_trajectory)} waypoints saved to '{args.trajectory_file}'")
    else:
        print("\n[WARNING] No valid top-down grasp was found during the session.")

    if args.camviz and grasp_poses_to_draw:
        print("[INFO] Launching final 3D visualization...")
        pcd_final_object = o3d.geometry.PointCloud(pcd_object_model).transform(final_transform)
        pcd_final_object.paint_uniform_color([1.0, 0.0, 0.0])
        gripper_mesh = plot_gripper_pro_max(grasp_poses_to_draw[0][:3, 3], grasp_poses_to_draw[0][:3, :3], 0.1, 0.09)
        cam_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        visualize_geometries([pcd_final_object, gripper_mesh, cam_origin], "Final Grasp Visualization", args.point_size)

    print("\n[INFO] Pipeline finished successfully.")

if __name__ == "__main__":
    main()