import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Generate a synthetic trajectory (Replace this with actual robot trajectory data)
num_points = 50
t = np.linspace(0, 2 * np.pi, num_points)

# Example trajectory: A helical path with orientation changes
x = np.cos(t)
y = np.sin(t)
z = t / (2 * np.pi)
roll = np.sin(t) * np.pi / 4
pitch = np.cos(t) * np.pi / 4
yaw = t

# Convert roll, pitch, yaw to rotation matrices
poses = np.column_stack((x, y, z, roll, pitch, yaw))

# Plot the trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], 'b-', label="Trajectory")

# Plot orientation frames at selected points
for i in range(0, num_points, 5):  # Plot every 5th frame for clarity
    pos = poses[i, :3]
    rpy = poses[i, 3:]
    rot_matrix = R.from_euler('xyz', rpy).as_matrix()

    # Define frame axes
    scale = 0.1
    for j, color in enumerate(['r', 'g', 'b']):  # X=Red, Y=Green, Z=Blue
        axis = rot_matrix[:, j] * scale
        ax.quiver(pos[0], pos[1], pos[2], axis[0], axis[1], axis[2], color=color)

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Trajectory with Orientation Frames")
ax.legend()
plt.show()
