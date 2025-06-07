#!/usr/bin/env python3


import os, sys, argparse, numpy as np, open3d as o3d, cv2
sys.path.append("feature-splatting-inria")
sys.path.append(".")
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene

# ─────────── constants ────────────────────────────────────────────────
# world→base  (from your long pipeline, unchanged)
W2B = np.array([
    [ 0.986680115832,  0.12442503, -0.10478913,  0.373036 ],
    [ 0.16035478,     -0.8522923,   0.49787968, -0.0069075],
    [-0.02736227,     -0.50805142, -0.86089202,  0.4207423],
    [ 0, 0, 0, 1]
])

# gripper→wristCam  (same constant you used in GaussianSplatRenderer)
gTc = np.array([
    [ 0.55898662, -0.82831147, -0.03786894,  0.04419197],
    [ 0.80713679,  0.55402063, -0.20393957,  0.11418125],
    [ 0.18990565,  0.08343408,  0.97825078, -0.00579795],
    [ 0, 0, 0, 1]
])

# ─────────── helpers ──────────────────────────────────────────────────
def load_intrinsics(path):
    d = np.load(path)
    return float(d["fx"]), float(d["fy"]), float(d["ppx"]), float(d["ppy"])

def project_graphics(p, fx, fy, cx, cy):
    X, Y, Z = p[:,0], p[:,1], p[:,2]
    u =  fx * (X / Z) + cx
    v = -fy * (Y / Z) + cy
    return u, v, Z

def in_img(u, v, z, W, H):
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    return (z > 1e-6) & (ui>=0)&(ui<W)&(vi>=0)&(vi<H)

# ─────────── CLI ──────────────────────────────────────────────────────
def make_args():
    p = argparse.ArgumentParser()
    ModelParams(p); PipelineParams(p); OptimizationParams(p)

    p.set_defaults(model_path="/home/carl_lab/akash/gaussian-splatting/output/franka_outputs",
                   source_path="/home/carl_lab/Riad/graspbility/GraspSplats/franka_data")

    p.add_argument("--pose-file",  default="franka_data/poses/2.npy")   # base→wristCam (4×4)
    p.add_argument("--rgb-file",   default="franka_data/images/2.png")  # to read W,H
    p.add_argument("--intrinsics", default="franka_data/rgb_intrinsics.npz")
    p.add_argument("--point-size", type=float, default=1.4)
    return get_combined_args(p)

# ─────────── MAIN ─────────────────────────────────────────────────────
def main():
    a = make_args()

    # 1 ▸ all splats in “world” ------------------------------------------------
    g = GaussianModel(a.sh_degree, a.distill_feature_dim)
    g.training_setup(a)
    Scene(a, g, load_iteration=-1, shuffle=False)
    xyz_world = g.get_xyz.detach().cpu().numpy()                # (N,3)

    # 2 ▸ world→wristCam transform (with gripper offset) ----------------------
    bTg  = np.load(a.pose_file)                                 # base→gripper
    wTcam = np.linalg.inv( np.linalg.inv(W2B) @ bTg @ gTc )     # world→cam
    R_wc, t_wc = wTcam[:3,:3], wTcam[:3,3]
    xyz_cam0 = (R_wc @ xyz_world.T).T + t_wc

    # 3 ▸ intrinsics & crop in BOTH axes --------------------------------------
    fx, fy, cx, cy = load_intrinsics(a.intrinsics)
    H, W, _ = cv2.imread(a.rgb_file).shape

    W_full, H_full = int(round(2*cx)), int(round(2*cy))
    crop_left = max((W_full - W)//2, 0)   # centre-crop ΔX
    crop_top  = max((H_full - H)//2, 0)   # centre-crop ΔY
    if crop_left or crop_top:
        print(f"[INFO] crop detected  left={crop_left}px  top={crop_top}px")

    cx -= crop_left
    cy -= crop_top

    # 4 ▸ brute-force (flipY, flipZ) handedness search ------------------------
    variants = {
        ( 1,  1):  xyz_cam0,
        ( 1, -1):  xyz_cam0 * [1,-1, 1],
        (-1,  1):  xyz_cam0 * [1, 1,-1],
        (-1, -1):  xyz_cam0 * [1,-1,-1],
    }
    best_cnt, best_key, best_xyz = -1, None, None
    for flips, pts in variants.items():
        u,v,z = project_graphics(pts, fx, fy, cx, cy)
        u -= crop_left;  v -= crop_top
        cnt  = in_img(u, v, z, W, H).sum()
        if cnt > best_cnt:
            best_cnt, best_key, best_xyz = cnt, flips, pts
    print(f"[INFO] flips chosen  Y={best_key[0]}  Z={best_key[1]}   → {best_cnt} visible")

    # 5 ▸ final mask & Open3D clouds -----------------------------------------
    u,v,z = project_graphics(best_xyz, fx, fy, cx, cy)
    u-=crop_left; v-=crop_top
    mask = in_img(u,v,z,W,H)
    xyz_vis = xyz_world[mask]

    pc_all = o3d.geometry.PointCloud()
    pc_all.points = o3d.utility.Vector3dVector(xyz_world)
    pc_all.paint_uniform_color([0.45,0.45,0.45])

    pc_vis = o3d.geometry.PointCloud()
    pc_vis.points = o3d.utility.Vector3dVector(xyz_vis)
    pc_vis.paint_uniform_color([1,0,0])

    vis = o3d.visualization.Visualizer()
    vis.create_window("all = grey  |  visible = red")
    vis.add_geometry(pc_all); vis.add_geometry(pc_vis)
    opt = vis.get_render_option(); opt.point_size = a.point_size; opt.background_color=[0,0,0]
    vis.run(); vis.destroy_window()

if __name__ == "__main__":
    main()
