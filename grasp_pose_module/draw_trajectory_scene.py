#!/usr/bin/env python3
"""
combined_grasp_pipeline_with_fixed_xy.py

Headless realbot_ui.py flow + GraspSplats screw‐trajectory, with
random start **height** only (Z varies) while X,Y stay at robot‐base origin.

Steps:
  1) CLIP‐based segmentation
  2) Overlay & save segmented point cloud
  3) Local object grasp generation & filtering
  4) Fit table plane & sample initial EE pose at (0,0,Z) above table
  5) Screw‐exponential interpolation from M0→Mg
  6) Visualization of scene, grasps, and trajectory
"""
import os, sys, argparse, tempfile
import numpy as np
import torch
import open3d as o3d
from scipy.linalg import logm, expm

sys.path.append("feature-splatting-inria")
sys.path.append(".")
from scipy.spatial.transform import Rotation
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene, skip_feat_decoder
import featsplat_editor
from gaussian_edit import edit_utils
from grasping.grasping_utils import sample_grasps, plot_gripper_pro_max

# π/2 about Y, π about Z (realbot_ui orientation hack)
Ry = np.array([[0.0, 0.0,  1.0],
               [0.0, 1.0,  0.0],
               [-1.0,0.0,  0.0]], dtype=np.float64)
Rz = np.array([[-1.0,0.0,  0.0],
               [ 0.0,-1.0, 0.0],
               [ 0.0, 0.0,  1.0]], dtype=np.float64)

def write_tmp_pcd(pcd:o3d.geometry.PointCloud)->str:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.02))
    pcd.orient_normals_consistent_tangent_plane(100)
    fd,path=tempfile.mkstemp(suffix=".pcd"); os.close(fd)
    o3d.io.write_point_cloud(path, pcd)
    return path

def screw_trajectory(M0:np.ndarray, Mg:np.ndarray, steps:int=100):
    ΔM = np.linalg.inv(M0) @ Mg
    ξ_hat = logm(ΔM).real
    return [M0 @ expm(ξ_hat * α) for α in np.linspace(0,1,steps)]

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
    parser.add_argument("--obj-pos-thresh", type=float, default=0.75)
    parser.add_argument("--obj-neg-thresh", type=float, default=0.7)
    parser.add_argument("--part-thresh",    type=float, default=0.7)
    parser.add_argument("--tsdf-ply", default="/home/carl_lab/Riad/graspbility/GraspSplats/franka_data/sparse/0/points3D.ply",
                        help="Path to TSDF PLY under source_path (e.g. 'sparse/0/points3D.ply')")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--dist-thresh", type=float, default=0.02)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--ee-pose", nargs=16, type=float, default=None,
                        help="Initial EE pose as 16 floats (row-major 4×4)")
    return get_combined_args(parser)

def main():
    args = parse_args()

    # 1) Gaussian model & scene
    g = GaussianModel(args.sh_degree, args.distill_feature_dim)
    g.training_setup(args)
    Scene(args, g, load_iteration=-1, shuffle=False)

    # 2) CLIP segmentation
    decoder = skip_feat_decoder(args.distill_feature_dim, part_level=True).cuda()
    decoder.load_state_dict(torch.load(
        os.path.join(args.model_path,"feat_decoder.pth"), map_location="cpu"))
    decoder.eval()
    segm = featsplat_editor.clip_segmenter(g, decoder)

    pts = g.get_xyz.detach().cpu().numpy()
    with torch.no_grad():
        sim_pos = segm.compute_similarity_one(args.object_query, level="object")
    if torch.is_tensor(sim_pos): sim_pos = sim_pos.cpu().numpy()
    mask = sim_pos > args.obj_pos_thresh

    if args.negative_query:
        with torch.no_grad():
            sim_neg = segm.compute_similarity_one(
                args.negative_query, level="object")
        if torch.is_tensor(sim_neg): sim_neg = sim_neg.cpu().numpy()
        mask &= (sim_neg < args.obj_neg_thresh)

    if args.part_query:
        with torch.no_grad():
            sim_part = segm.compute_similarity_one(
                args.part_query, level="part")
        if torch.is_tensor(sim_part): sim_part = sim_part.cpu().numpy()
        mask &= (sim_part > args.part_thresh)

    # 3) Cluster + flood-fill
    mask = edit_utils.cluster_instance(pts, mask)
    mask = edit_utils.flood_fill(pts, mask)

    # 4) Overlay & save segmented cloud
    W2B = np.array([
        [0.987512752039565,  0.12239545440329831, -0.09918627576764462,  0.3727374623352868],
        [0.15566726404202447, -0.8548697390574798,  0.49493982679661963, -0.006433565113063563],
        [-0.02421296068050752,-0.5041994466331823, -0.863247734170138,    0.4205687356798313],
        [0.0,                  0.0,                  0.0,                  1.0],
    ], dtype=np.float64)

    out_dir = os.path.join(args.model_path,"point_cloud_for_grasp")
    os.makedirs(out_dir,exist_ok=True)

    gauss_fg = edit_utils.select_gaussians(g, mask)
    xyz_w    = gauss_fg._xyz.detach().cpu().numpy()
    homo     = np.hstack([xyz_w, np.ones((xyz_w.shape[0],1))])
    xyz_b    = (W2B @ homo.T).T[:,:3]

    scene_pcd = o3d.io.read_point_cloud(
        os.path.join(args.source_path,args.tsdf_ply))
    scene_pcd.transform(W2B)

    pcd_fg = o3d.geometry.PointCloud()
    pcd_fg.points = o3d.utility.Vector3dVector(xyz_b)
    pcd_fg.paint_uniform_color([1,0,0])
    overlay = scene_pcd + pcd_fg
    o3d.io.write_point_cloud(
        os.path.join(out_dir,"segmentation_overlay.ply"), overlay)
    print("[+] Saved overlay")

    # 5) Local grasp sampling + filtering
    gauss_sel = edit_utils.select_gaussians(g, mask)
    gauss_sel._xyz = torch.from_numpy(xyz_b).to(g.get_xyz.device)
    gauss_sel.save_ply(
        os.path.join(out_dir,"local_object_gaussians.ply"))

    tmp = write_tmp_pcd(pcd_fg)
    local_poses, local_scores = sample_grasps(tmp, if_global=False)
    os.remove(tmp)

    obj_pts = np.asarray(pcd_fg.points)
    final   = []
    for M, sc in zip(local_poses, local_scores):
        M2 = M.copy()
        M2[:3,:3] = M[:3,:3] @ Ry
        if np.dot(M2[:3,0],[1,0,0])<0:
            M2[:3,:3] = M2[:3,:3] @ Rz
        ang = np.arccos(
            np.dot(-M2[:3,2],[0,0,1]) / np.linalg.norm(M2[:3,2]))
        if ang>np.pi/4: continue
        mid = M[:3,3] + 0.05*M[:3,2]
        if np.min(np.linalg.norm(obj_pts-mid,axis=1))<0.02:
            final.append((M2,sc))
    final = sorted(final, key=lambda x:-x[1])[:args.topk]
    if not final:
        print("[WARNING] No grasps passed filtering"); return

    # 6) Fit table plane
    plane_model, inliers = scene_pcd.segment_plane(
        distance_threshold=0.005, ransac_n=3, num_iterations=1000)
    [a,b,c,d] = plane_model
    if c<0: a,b,c,d = -a,-b,-c,-d
    inlier_pts = np.asarray(scene_pcd.points)[inliers]
    table_z    = inlier_pts[:,2].mean()

    # 7) Visualization: scene + grasp mesh
    vis = o3d.visualization.Visualizer(); vis.create_window()
    vis.add_geometry(scene_pcd)
    for M2,_ in final:
        vis.add_geometry(plot_gripper_pro_max(
            M2[:3,3], M2[:3,:3], 0.08, 0.06))

    # 8) Initial EE pose M0 at (0,0,Z)
    ee_vals = getattr(args, "ee_pose", None)
    if ee_vals is not None:
        M0 = np.array(ee_vals).reshape(4,4)
    else:
        z0 = np.random.uniform(table_z + 0.05, table_z + 0.5)
        R0 = np.eye(3)  # robot-base axes
        M0 = np.eye(4)
        M0[:3,:3], M0[:3,3] = R0, [0.0, 0.0, z0]

    # draw initial EE frame
    init_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1)
    init_frame.transform(M0)
    vis.add_geometry(init_frame)

    # 9) Screw trajectory & line
    Mg   = final[0][0]
    traj = screw_trajectory(M0, Mg, steps=100)
    traj_pts = [T[:3,3] for T in traj]
    lines    = [[i,i+1] for i in range(len(traj_pts)-1)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(traj_pts),
        lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([[0,1,0]]*len(lines))
    vis.add_geometry(ls)

    # start/end frames
    sf = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05); sf.transform(traj[0]); vis.add_geometry(sf)
    ef = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05); ef.transform(traj[-1]); vis.add_geometry(ef)

    vis.run(); vis.destroy_window()


if __name__=="__main__":
    main()
