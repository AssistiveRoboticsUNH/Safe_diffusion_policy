#!/usr/bin/env python3

import os
import sys
import argparse
import tempfile
import numpy as np
import torch
import open3d as o3d

sys.path.append("feature-splatting-inria")
sys.path.append(".")
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene, skip_feat_decoder
import featsplat_editor
from gaussian_edit import edit_utils
from grasping.grasping_utils import sample_grasps, plot_gripper_pro_max

# π/2 about Y, π about Z (as in realbot_ui.py)
Ry = np.array([[0.0, 0.0,  1.0],
               [0.0, 1.0,  0.0],
               [-1.0,0.0,  0.0]], dtype=np.float64)
Rz = np.array([[-1.0,0.0,  0.0],
               [ 0.0,-1.0, 0.0],
               [ 0.0, 0.0,  1.0]], dtype=np.float64)

def get_world2base() -> np.ndarray:
    return np.array([
        [0.987512752039565,  0.12239545440329831, -0.09918627576764462,  0.3727374623352868],
        [0.15566726404202447, -0.8548697390574798,  0.49493982679661963, -0.006433565113063563],
        [-0.02421296068050752,-0.5041994466331823, -0.863247734170138,    0.4205687356798313],
        [0.0,                  0.0,                  0.0,                  1.0],
    ], dtype=np.float64)

def write_tmp_pcd(pcd: o3d.geometry.PointCloud) -> str:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.02))
    pcd.orient_normals_consistent_tangent_plane(100)
    fd, path = tempfile.mkstemp(suffix=".pcd")
    os.close(fd)
    o3d.io.write_point_cloud(path, pcd)
    return path

def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    OptimizationParams(parser)

    parser.add_argument("-q", "--object-query", default="Red Candy",
                        help="Positive text query")
    parser.add_argument("-n", "--negative-query", default="Green Candy",
                        help="Negative text query (optional)")
    parser.add_argument("-p", "--part-query", default="Red Candy",
                        help="Part-level text query (optional)")
    parser.add_argument("--obj-pos-thresh", type=float, default=0.70,
                        help="Positive similarity threshold")
    parser.add_argument("--obj-neg-thresh", type=float, default=0.7,
                        help="Negative similarity threshold")
    parser.add_argument("--part-thresh",    type=float, default=0.7,
                        help="Part similarity threshold")

    parser.add_argument("--tsdf-ply", default="/home/carl_lab/Riad/graspbility/GraspSplats/franka_data/sparse/0/points3D.ply",
                        help="Path to TSDF PLY under source_path (e.g. 'sparse/0/points3D.ply')")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of final grasps to print")
    parser.add_argument("--dist-thresh", type=float, default=0.01,
                        help="Midpoint-to-object max distance filter")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize overlay & grasps in Open3D")

    return get_combined_args(parser)

def main():
    args = parse_args()

    # 1) Setup Gaussian model & scene
    g = GaussianModel(args.sh_degree, args.distill_feature_dim)
    g.training_setup(args)
    Scene(args, g, load_iteration=-1, shuffle=False)

    # 2) CLIP decoder & segmenter
    decoder = skip_feat_decoder(args.distill_feature_dim, part_level=True).cuda()
    ckpt = os.path.join(args.model_path, "feat_decoder.pth")
    decoder.load_state_dict(torch.load(ckpt, map_location="cpu"))
    decoder.eval()
    segm = featsplat_editor.clip_segmenter(g, decoder)

    # 3) Compute similarity mask
    pts = g.get_xyz.detach().cpu().numpy()
    with torch.no_grad():
        sim_pos = segm.compute_similarity_one(args.object_query, level="object")
    sim_pos = sim_pos.cpu().numpy() if isinstance(sim_pos, torch.Tensor) else sim_pos
    mask = sim_pos > args.obj_pos_thresh

    if args.negative_query:
        with torch.no_grad():
            sim_neg = segm.compute_similarity_one(args.negative_query, level="object")
        sim_neg = sim_neg.cpu().numpy() if isinstance(sim_neg, torch.Tensor) else sim_neg
        mask &= (sim_neg < args.obj_neg_thresh)

    if args.part_query:
        with torch.no_grad():
            sim_part = segm.compute_similarity_one(args.part_query, level="part")
        sim_part = sim_part.cpu().numpy() if isinstance(sim_part, torch.Tensor) else sim_part
        mask &= (sim_part > args.part_thresh)

    # 4) EXACT realbot_ui.py clustering + flood-fill
    mask = edit_utils.cluster_instance(pts, mask)
    mask = edit_utils.flood_fill(pts, mask)

    # 5) Overlay & save segmented cloud
    W2B    = get_world2base()
    out_dir = os.path.join(args.model_path, "point_cloud_for_grasp")
    os.makedirs(out_dir, exist_ok=True)

    gauss_fg  = edit_utils.select_gaussians(g, mask)
    xyz_fg_w  = gauss_fg._xyz.detach().cpu().numpy()
    homo_fg   = np.hstack([xyz_fg_w, np.ones((xyz_fg_w.shape[0],1))])
    xyz_fg_b  = (W2B @ homo_fg.T).T[:, :3]

    scene = o3d.io.read_point_cloud(os.path.join(args.source_path, args.tsdf_ply))
    scene.transform(W2B)

    pcd_fg = o3d.geometry.PointCloud()
    pcd_fg.points = o3d.utility.Vector3dVector(xyz_fg_b)
    pcd_fg.paint_uniform_color([1, 0, 0])

    overlay = scene + pcd_fg
    overlay_path = os.path.join(out_dir, "segmentation_overlay.ply")
    o3d.io.write_point_cloud(overlay_path, overlay)
    print(f"[+] Saved overlay → {overlay_path}")

    if args.visualize:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([overlay, axis],
                                          window_name="Segmentation Overlay")

    # 6) Local grasp sampling & filtering
    # 6a) Select & transform Gaussians to base frame
    gauss_sel = edit_utils.select_gaussians(g, mask)
    xyz_w     = gauss_sel._xyz.detach().cpu().numpy()
    homo_sel  = np.hstack([xyz_w, np.ones((xyz_w.shape[0],1))])
    xyz_b     = (W2B @ homo_sel.T).T[:, :3]
    gauss_sel._xyz = torch.from_numpy(xyz_b).to(g.get_xyz.device)

    # 6b) Save & sample
    save_path = os.path.join(out_dir, "local_object_gaussians.ply")
    gauss_sel.save_ply(save_path)
    tmp_pcd, poses, scores = None, None, None
    tmp_pcd = write_tmp_pcd(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_b)))
    poses, scores = sample_grasps(tmp_pcd, if_global=False)
    os.remove(tmp_pcd)
    print(f"[INFO] Sampled {len(poses)} raw local grasps")

    # 6c) Filter — build exec_poses & vis_poses separately
    obj_pts     = xyz_b
    exec_poses  = []
    vis_poses   = []
    final_scores= []

    for sc, p in zip(scores, poses):
        # exec orientation: rotate last joint π/2 about Y
        g_exec = p.copy()
        g_exec[:3,:3] = p[:3,:3] @ Ry
        # flip if x-axis backwards
        if np.dot(g_exec[:3,0], [1,0,0]) < 0:
            g_exec[:3,:3] = g_exec[:3,:3] @ Rz
        # reject tilt > 45°
        dz = -g_exec[:3,2]
        if np.arccos(np.dot(dz,[0,0,1]) / np.linalg.norm(dz)) > (np.pi/4):
            continue
        # lower midpoint by 5cm for test
        mid = p[:3,3] + 0.05 * p[:3,2]
        # require within 2cm of object cloud
        if np.min(np.linalg.norm(obj_pts - mid, axis=1)) < 0.02:
            exec_poses.append(g_exec)
            vis_poses.append(p.copy())   # raw for viz
            final_scores.append(sc)

    print(f"[INFO] {len(exec_poses)} grasps after filtering")

    if not exec_poses:
        print("[WARNING] No local grasps passed filtering.")
        return

    # 7) Print & visualize top-k
    sc_arr = np.array(final_scores)
    norm   = (sc_arr - sc_arr.min()) / (sc_arr.max() - sc_arr.min())
    order  = np.argsort(norm)[-args.topk:]

    for i in order:
        ctr = exec_poses[i][:3,3]
        print(f"Grasp #{i}: {ctr}, score={norm[i]:.3f}")

    if args.visualize:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        grippers = [
            plot_gripper_pro_max(vis_poses[i][:3,3],
                                 vis_poses[i][:3,:3],
                                 0.08, 0.06)
            for i in order
        ]
        o3d.visualization.draw_geometries([scene, axis, *grippers],
                                          window_name="Top-k Grasps")

    print("Headless pipeline complete.")


if __name__ == "__main__":
    main()
