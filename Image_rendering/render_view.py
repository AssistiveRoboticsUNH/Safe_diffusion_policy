from matplotlib import pyplot as plt
import numpy as np
from render_for_robot_class_v3_module import GaussianSplatRenderer

renderer = GaussianSplatRenderer(model_path="/home/carl_lab/akash/gaussian-splatting/output/franka_outputs")

# rgb_list, depth_list, arm_pose_list, rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs, depth_scale = read_data(renderer.source_path)

# bTc = arm_pose_list[0]


# T = np.array([
#     [-0.61596241,  0.76216893, -0.19922046,  0.4820782 ],
#     [ 0.65539716,  0.63611089,  0.4072069 , -0.25117048],
#     [ 0.43708675,  0.12025562, -0.89134379,  0.35940646],
#     [ 0.0       ,  0.0       ,  0.0       ,  1.0       ]
# ])



pose20 = np.array([
    [ 0.852638,  0.311121,  0.419777, -0.110382],
    [ 0.362981, -0.930582, -0.047566, -0.182613],
    [ 0.375838,  0.192928, -0.90638 ,  0.429924],
    [ 0.0     ,  0.0     ,  0.0     ,  1.0     ]
])

W2B = np.array([
            [0.9866801158315599, 0.12442503082301001, -0.10478912504316718, 0.3730359970326709],
            [0.16035477915117624, -0.8522923025192382, 0.49787968011341105, -0.006907504865351453],
            [-0.027362270117755733, -0.5080514174482224, -0.8608920161105309, 0.4207423250737201],
            [0.0, 0.0, 0.0, 1.0],
            ])

pose20 = np.linalg.inv(pose20)
# pose20 = W2B @ pose20
rendered_tensor = renderer.get_rendered_image(pose20, pose_type = "4x4") # or pose_type="7d"

# torchvision.utils.save_image(rendered_tensor, "myRender1.png")

plt.imshow(rendered_tensor.permute(1, 2, 0).cpu().numpy())  # Ensure it's in (H, W, C) format
plt.axis("off")
plt.show()

print("Done")