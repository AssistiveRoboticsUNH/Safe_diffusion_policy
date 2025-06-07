# https://github.com/graphdeco-inria/gaussian-splatting/issues/506
# Problem is different axis scaling of the frames
import os
import json
import numpy as np
import torch
from argparse import ArgumentParser
from typing import NamedTuple, Optional
from scipy.spatial.transform import Rotation as R
from PIL import Image
from matplotlib import pyplot as plt
import sys
sys.path.append("feature-splatting-inria")
sys.path.append(".")
from scene import Scene
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
from scene.colmap_loader import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    qvec2rotmat,
)
from scene.cameras import Camera
import torchvision
from scipy.linalg import sqrtm
from camera_view_extractor import SplatCameraInfo
import cv2

class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    depth_params: Optional[dict]
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class DummyCamera:
    def __init__(self, R, T, FoVx, FoVy, W, H):
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        self.R = R
        self.T = T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0,0,0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = W
        self.image_height = H
        self.FoVx = FoVx
        self.FoVy = FoVy

def read_data(base_dir):
    rgb_folder = os.path.join(base_dir, 'images')
    depth_folder = os.path.join(base_dir, 'depth')
    pose_folder = os.path.join(base_dir, 'poses')

    print(rgb_folder)

    rgb_list, depth_list, pose_list = None, None, None

    # Check if RGB folder exists
    if os.path.exists(rgb_folder):
        # Read RGB images
        rgb_files = [f for f in os.listdir(rgb_folder) if f.endswith('.png')]
        rgb_files.sort()
        print(rgb_files)
        rgb_list = []
        for f in rgb_files:
            img = cv2.imread(os.path.join(rgb_folder, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_list.append(img)

    # Check if depth folder exists
    if os.path.exists(depth_folder):
        # Read depth images
        depth_files = [f for f in os.listdir(depth_folder) if f.endswith('.npy')]
        depth_files.sort()
        print(depth_files)
        depth_list = [np.load(os.path.join(depth_folder, f)) for f in depth_files]

    # Check if pose folder exists
    if os.path.exists(pose_folder):
        # Read poses
        pose_files = [f for f in os.listdir(pose_folder) if f.endswith('.npy')]
        pose_files.sort()
        print(pose_files)
        pose_list = [np.load(os.path.join(pose_folder, f)) for f in pose_files]

    # Check if camera parameters exist
    rgb_params_file = os.path.join(base_dir, 'rgb_intrinsics.npz')
    if os.path.exists(rgb_params_file):
        # Load the intrinsic parameters
        camera_params = np.load(rgb_params_file)
        fx = camera_params['fx']
        fy = camera_params['fy']
        ppx = camera_params['ppx']
        ppy = camera_params['ppy']
        rgb_coeffs = camera_params['coeffs']
        rgb_intrinsics = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
    else:
        rgb_intrinsics, rgb_coeffs = None, None

    depth_params_file = os.path.join(base_dir, 'depth_intrinsics.npz')
    if os.path.exists(depth_params_file):
        # Load the intrinsic parameters
        camera_params = np.load(depth_params_file)
        fx = camera_params['fx']
        fy = camera_params['fy']
        ppx = camera_params['ppx']
        ppy = camera_params['ppy']
        depth_coeffs = camera_params['coeffs']
        depth_intrinsics = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        depth_scale = camera_params['depth_scale']
    else:
        depth_intrinsics, depth_coeffs, depth_scale = None, None, None

    return rgb_list, depth_list, pose_list, rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs, depth_scale


class GaussianSplatRenderer:
    """
    Encapsulates loading of model, scene, and allows rendering from any
    camera pose expressed in the robot-base frame.
    """

    def __init__(self,model_path: str):
       
        self.bTw = np.array([
            [0.9866801158315599, 0.12442503082301001, -0.10478912504316718, 0.3730359970326709],
            [0.16035477915117624, -0.8522923025192382, 0.49787968011341105, -0.006907504865351453],
            [-0.027362270117755733, -0.5080514174482224, -0.8608920161105309, 0.4207423250737201],
            [0.0, 0.0, 0.0, 1.0],
            ])
        self.wTb = np.linalg.inv(self.bTw)
        self.gTc = np.array([
            [ 0.55898662, -0.82831147, -0.03786894,  0.04419197],
            [ 0.80713679,  0.55402063, -0.20393957,  0.11418125],
            [ 0.18990565,  0.08343408,  0.97825078, -0.00579795],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
            ])
        # store
        self.model_path  = model_path

        # — argparse to wire up ModelParams & PipelineParams —
        parser = ArgumentParser(add_help=False)
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--quiet",     action="store_true")
        parser.set_defaults(model_path=self.model_path)
        
        args = get_combined_args(parser)
        safe_state(args.quiet)

        self.source_path = args.source_path

        print("Model Path: ", self.model_path)
        print("Source Path: ", self.source_path)

        # — dataset & pipeline & scene setup —
        self.dataset   = model.extract(args)
        self.pipeline  = pipeline.extract(args)
        self.separate_sh = False
        try:
            from diff_gaussian_rasterization import SparseGaussianAdam
            self.separate_sh = True
        except ImportError:
            pass
        
        self.gaussians = GaussianModel(self.dataset.sh_degree,args.distill_feature_dim)
        self.scene     = Scene(self.dataset,self.gaussians,load_iteration=args.iteration,shuffle=False,)
        bg_color = [1,1,1] if self.dataset.white_background else [0,0,0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=args.data_device)
        

        
        camera_info = SplatCameraInfo(self.source_path)
        self.views = camera_info.get_camera_info()
        self.available_images_in_sfm = camera_info.available_images_in_sfm

        print(f"\nTotal images in sfm: {len(self.available_images_in_sfm)}")

        '''
        with open(self.source_path+"/franka_camera_info.json", "r") as f:
            self.franka_camera_info = json.load(f)

        # filter out images that are not in the image pose dataset
        self.all_image_keys = list(self.franka_camera_info.keys())
        print(f"Total images in demo: {len(self.all_image_keys)}")
        self.available_image_keys = []

        for image in self.all_image_keys:
            if image in self.available_images_in_sfm:
                self.available_image_keys.append(image)
        print(f"Total images in demo after filtering: {len(self.available_image_keys)}\n")
        '''
    def _ensure_homogeneous(self, mat: np.ndarray) -> np.ndarray:
        H = np.array(mat).reshape(4,4)
        if not np.allclose(H[3], [0,0,0,1]):
            H = H.T
        return H

    def pose7d_to_homogeneous(self, pose):
        """
        pose: iterable of 7 floats [tx, ty, tz, qx, qy, qz, qw],
            where the quaternion is (x,y,z,w).
        returns: 4x4 numpy array [R  t; 0 1]
        """
        tx, ty, tz, qx, qy, qz, qw = pose
        # build rotation
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()   # 3×3
        # assemble homogeneous
        H = np.eye(4)
        H[:3, :3] = rot
        H[:3,  3] = [tx, ty, tz]
        return H
    
    def get_rendered_image(self, gripper_pose_bTg: np.ndarray, pose_type="4x4") -> torch.Tensor:
        """
        Given a 4×4 camera pose in the robot-base frame (bTc),
        return the rendered image tensor.
        """
        
        # if pose_type == "4x4":
        #     bTg = np.array(gripper_pose_bTg).reshape(4, 4)
        #     bTg = self._ensure_homogeneous(gripper_pose_bTg)
        # elif pose_type == "7d":
        #     bTg = self.pose7d_to_homogeneous(gripper_pose_bTg)
            
        # closest_image, min_dist = renderer.find_closest_image(bTc)
        # print("Closest image: ", closest_image)
        # print("Distance: ", min_dist)


        ## ================= Render from splat camera pose==============
        view = self.views["0.png"]
        # print("Expected: ")
        # print(view.R.T)
        # print(view.T)


        world_T_C = np.linalg.inv( self.wTb @ gripper_pose_bTg @ self.gTc)

        R = world_T_C[:3, :3]
        T = world_T_C[:3, 3]

        # print("\nCalculated: ")
        # print(R)
        # print(T)

        myCam = DummyCamera(R.T, T, view.FovX, view.FovY, view.width, view.height)
        print()
        
        # render
        with torch.no_grad():
            out = render(
                myCam,
                self.gaussians,
                self.pipeline,
                self.background,
                # use_trained_exp=False,
                # separate_sh=self.separate_sh,
                )
        return out["render"]


# # renderer = GaussianSplatRenderer(model_path="/home/carl_lab/akash/gaussian-splatting/output/599308b4-a")
# renderer = GaussianSplatRenderer(model_path="/home/carl_lab/akash/gaussian-splatting/output/franka_outputs")



# # pose7d = "0.617899 -0.024531 5.253318 0.998965 -0.041620 -0.016440 -0.008106".split()
# # pose7d = [0.59348536,0.00767812,0.20253659,0.99943456,-0.03000188,-0.01490609,-0.00287291]


# # with open("first_demo_7d_poses.txt", "r") as f:
# #     count = 1
# #     for line in f:
# #         print(count)
# #         pose7d = list(map(float, line.split()))
# #         rendered_tensor = renderer.get_rendered_image(pose7d, pose_type = "7d")

# #         torchvision.utils.save_image(rendered_tensor, f"output_images/myRender{count}.png")
# #         count += 1



# # with open(renderer.source_path+"/franka_camera_info.json", "r") as f:
# #     franka_camera_info = json.load(f)
# #     count = 1
# #     # Camere pose wrt Robot base
# #     images = list(franka_camera_info.keys())
# #     for i in range(0, len(images), 5):
# #         img_name = images[i]
# #         image_pose_bTc = np.array(franka_camera_info[img_name]).reshape(4,4)
# #         image_pose_bTc = image_pose_bTc.T

# #         rendered_tensor = renderer.get_rendered_image(image_pose_bTc, pose_type = "4x4")

# #         torchvision.utils.save_image(rendered_tensor, f"output_images/myRender{count}.png")
# #         count += 1






# # pose_folder = os.path.join(renderer.source_path, 'poses')
# # # save or display:
# # pose_files = [f for f in os.listdir(pose_folder) if f.endswith('.npy')]
# # pose_files.sort()
# # print(pose_files)
# # pose_list = [np.load(os.path.join(pose_folder, f)) for f in pose_files]

# rgb_list, depth_list, arm_pose_list, rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs, depth_scale = read_data(renderer.source_path)

# bTc = arm_pose_list[0]




# T = np.array([
#     [-0.61596241,  0.76216893, -0.19922046,  0.4820782 ],
#     [ 0.65539716,  0.63611089,  0.4072069 , -0.25117048],
#     [ 0.43708675,  0.12025562, -0.89134379,  0.35940646],
#     [ 0.0       ,  0.0       ,  0.0       ,  1.0       ]
# ])


# rendered_tensor = renderer.get_rendered_image(T, pose_type = "4x4") # or pose_type="7d"

# # torchvision.utils.save_image(rendered_tensor, "myRender1.png")

# plt.imshow(rendered_tensor.permute(1, 2, 0).cpu().numpy())  # Ensure it's in (H, W, C) format
# plt.axis("off")
# plt.show()

# print("Done")