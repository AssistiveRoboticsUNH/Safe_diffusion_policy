'''
This script extracts camera view information from a dataset in the format used by Gaussian Splatting.
It reads camera extrinsics and intrinsics from binary files, computes the camera parameters, and stores them in a dictionary.


Author: @mnakash
'''
import sys
sys.path.append("feature-splatting-inria")
sys.path.append(".")
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool



class SplatCameraInfo():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        print("Getting Views from source: ", self.dataset_path)
        try:
            self.cameras_extrinsic_file = os.path.join(self.dataset_path, "sparse/0", "images.bin")
            self.cameras_intrinsic_file = os.path.join(self.dataset_path, "sparse/0", "cameras.bin")
            self.cam_extrinsics = read_extrinsics_binary(self.cameras_extrinsic_file)
            self.cam_intrinsics = read_intrinsics_binary(self.cameras_intrinsic_file)
        except:
            self.cameras_extrinsic_file = os.path.join(self.dataset_path, "sparse/0", "images.txt")
            self.cameras_intrinsic_file = os.path.join(self.dataset_path, "sparse/0", "cameras.txt")
            self.cam_extrinsics = read_extrinsics_text(self.cameras_extrinsic_file)
            self.cam_intrinsics = read_intrinsics_text(self.cameras_intrinsic_file)
    

        # self.cameras_extrinsic_file = os.path.join(self.dataset_path, "sparse", "0", "images.bin")
        # self.cameras_intrinsic_file = os.path.join(self.dataset_path, "sparse", "0", "cameras.bin")
        # self.cam_extrinsics = read_extrinsics_binary(self.cameras_extrinsic_file)
        # self.cam_intrinsics = read_intrinsics_binary(self.cameras_intrinsic_file)   
        self.cam_infos = {}
        self.available_images_in_sfm = []

    def get_camera_info(self):
        for idx, key in enumerate(self.cam_extrinsics):
            
            
            extr = self.cam_extrinsics[key]
            intr = self.cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width

            uid = intr.id
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            n_remove = len(extr.name.split('.')[-1]) + 1
            depth_params = None
            
            image_name = extr.name
            self.available_images_in_sfm.append(image_name) # Store avaialable images in sfm
            # print(f"{key} > {image_name}")
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                                image_path="", image_name=image_name, depth_path="",
                                width=width, height=height, is_test=True)
            
            # if cam_info.image_name == "image_1.png":
            #     print(cam_info)
            #     print("Scale of Rotation Matrix: ", np.linalg.norm(cam_info.R))
            #     break
            self.cam_infos[cam_info.image_name] = cam_info
        return self.cam_infos
    

# source_path = "/home/akash/UNH/safety_research/gaussian-splatting/dataset/gsplat_demo31/"
# camera_info = SplatCameraInfo(source_path)
# views = camera_info.get_camera_info()
# view = views["image_115.png"]
# print(view)

# source_path = "/home/carl_lab/Riad/graspbility/colmap_handeye/example_data"
# camera_info = SplatCameraInfo(source_path)
# views = camera_info.get_camera_info()

# print(views.keys())
# view = views["0.png"]
# print(view)




