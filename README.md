# GraspSplats + Grasp Pose Module

A combined repository featuring the core **GraspSplats** framework and a dedicated **Grasp Pose Module** for headless grasp execution, trajectory generation, Gaussian-based visualization, and image rendering.

---

## 🚀 Quick Setup

### 1. Clone & Switch to Your Branch

```bash
git clone https://github.com/jimazeyu/GraspSplats.git
cd GraspSplats
git checkout -b feature/grasp-pose-module
```

---

## 📦 Installation

1. **Create the environment**

   ```bash
   micromamba create -n grasp_splats python=3.10 -c conda-forge
   micromamba activate grasp_splats
   ```

2. **Install part-level feature splatting**

   ```bash
   git clone --recursive https://github.com/vuer-ai/feature-splatting-inria.git
   cd feature-splatting-inria
   git checkout roger/graspsplats_part
   pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   micromamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
   ```

3. **Install submodule: diff-gaussian-rasterization**

   ```bash
   cd feature-splatting-inria/submodules/diff-gaussian-rasterization
   pip install -e .
   ```

4. **Install submodule: simple-knn**

   ```bash
   cd ../simple-knn
   pip install -e .
   ```

5. **Install remaining Python requirements**

   ```bash
   cd ../../
   pip install -r requirements.txt
   ```

6. **(Optional) Fix compatibility issues**

   ```bash
   pip install numpy==1.23.5        # Downgrade to 'numpy<2' if needed
   pip install setuptools==69.5.1   # Fixes ‘packaging’ import errors
   ```

7. **Install Grasp Pose Detection (GPD)**
   Follow the [GPD installation guide](https://github.com/tu-rbo/GPD) to set up OpenCV, Eigen, and PCL, then:

   ```bash
   cd gpd
   mkdir build && cd build
   cmake ..
   make -j8
   ```

8. **Install additional Python dependencies**

   ```bash
   pip install viser==0.1.10 roboticstoolbox-python transforms3d
   pip install panda_python
   ```

---

## 🎯 Module Usage

### Grasp Pose & Trajectory Tools (`grasp_pose_module/`)

1. **Headless GraspSplats (no GUI)**

   ```bash
   python grasp_pose_module/grasp_splat_headless.py \
     -m /path/to/franka_outputs \
     -s /path/to/franka_data \
     --visualize
   ```

2. **Non-linear trajectory (random init → grasp)**

   ```bash
   python grasp_pose_module/draw_trajectory_scene.py \
     -m /path/to/franka_outputs \
     -s /path/to/franka_data \
     --visualize
   ```

3. **Linear trajectory (camera → grasp)**

   ```bash
   python grasp_pose_module/generate_proper_from_camera_pose_trajectory.py \
     --visualize
   ```

4. **Select visible Gaussians**

   ```bash
   python grasp_pose_module/select_gaussian_from_camera_pose.py
   ```

5. **Detect grasp pose with camera viz**

   ```bash
   python grasp_pose_module/grasp_pose_con_camera_pose.py --camviz
   ```

6. **Visualize Gaussians along trajectory**

   ```bash
   python grasp_pose_module/generate_visualization_camera_pose_from_trajectory.py
   ```

7. **Graspability visualization (Gaussians + grasp)**

   ```bash
   python grasp_pose_module/generate_grasp_pose_vis_camera_pose_trajectory.py
   ```

---

### Image Rendering (`Image_rendering/`)

* **Render single-view image**

  ```bash
  python Image_rendering/render_view.py
  ```

* **Render trajectory sequence**
  *(Requires a pre-generated trajectory)*

  ```bash
  python Image_rendering/render_images_trajectory.py
  ```

---


