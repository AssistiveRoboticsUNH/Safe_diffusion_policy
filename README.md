# Safe Diffusion Policy

This guide provides step-by-step instructions for setting up the environment, installing dependencies, recording demonstrations, and running experiments.

---

## 🚀 Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AssistiveRoboticsUNH/Safe_diffusion_policy
```

### 2️⃣ Create & Activate the Conda Environment
```bash
conda create --name safe_lfd
conda activate safe_lfd
```

### 3️⃣ Clone VIVID123
```bash
git clone https://github.com/ubc-vision/vivid123
```

### 4️⃣ Install PyTorch (Compatible with Your CUDA Version)
Follow the official [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) to install the appropriate version.

Example (for CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5️⃣ Install Required Python Packages
```bash
pip install "diffusers==0.24" transformers accelerate einops kornia imageio[ffmpeg] opencv-python pydantic scikit-image lpips h5py
```

### 6️⃣ Install Additional Dependencies
```bash
conda install conda-forge::matplotlib
```

### 7️⃣ Install Background Removal Package (For Background Removal Task)
```bash
pip install carvekit --extra-index-url https://download.pytorch.org/whl/cu113
```

---

## 📌 Running the Experiment

You can either **record a demonstration using the robot** or **test your installation** with our sample dataset.

### 1️⃣ Download Sample Data
```bash
pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/1KnVeUR2r97q7at0cCsCNG-uVJqIuRdwl/view?usp=drive_link"
```

### 2️⃣ Update the Dataset Path
Modify the following line in `run_vivid_123_experiment_v1.py`:
```python
dataset_path = "demo.hdf5"
```

### 3️⃣ Run the Experiment
```bash
python run_vivid_123_experiment_v1.py
```

---

## 🛠️ Safe Set Generation & Trajectory Synthesis

To generate safe sets and synthesize trajectories, install the required dependencies:

```bash
pip install mpl-tools
conda install conda-forge::scipy
conda install conda-forge::plotly
conda install conda-forge::pyvista
```


## 🛠️ Safe Set Generation & Trajectory Synthesis

To generate safe sets and synthesize trajectories, install the required dependencies:

```bash
git clone https://github.com/real-stanford/diffusion_policy
conda env create -f conda_environment.yaml
```
Copy 
1. train_object_detector_using_visionencoder.ipynb
2. safe_image_franka_image_240_320.yaml
3. safe_train_franka.ipynb

to the diffusion policy folder. 

To train the diffusion policy:
1. edit safe_image_franka_image_240_320.yaml
2. run safe_train_franka.ipynb

To train object detector using the same vision encoder:

Run train_object_detector_using_visionencoder.ipynb

---

## 💡 Notes

- Ensure that you have **CUDA installed and properly configured** to leverage GPU acceleration.
- If you encounter **installation issues**, check the package compatibility with your system.
- For PyTorch version compatibility with CUDA, refer to [PyTorch's official guide](https://pytorch.org/get-started/locally/).

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 👨‍💻 Contributors

- **[Your Name]** – *Maintainer & Developer*
- **[Other Contributors]** – *Contributors*

---

## 📫 Contact

For any issues or queries, feel free to open an issue or reach out at **[riad.ahmed@unh.edu]**.

