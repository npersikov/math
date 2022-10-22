# Navigation Design Challenge for SpaceRyde GNC Engineer Application

# Used imports
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import random

# Initial states of relevant objects. Column vectors.
robot_state_bodyFromOrigin_BodyFrame = np.array([[0], [0], [0], [0], [0], [np.pi/2], [0], [0], [0], [0], [0], [0]]);
radio_beacon_position_inertial = np.array([[5], [0], [0]]);

# Robot shape for plotting
width   = 0.6;
length  = 1;
height  = 0.2;

# Robot parameters
a_mps2          = 10
v_max_mps       = 5
alpha_radps2    = 2
omega_max_radps = 1

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(x, y, z, 50, cmap='binary')

# Plot util vars
window_size = 10

# Simulation variables
time_step = 1.0/25.0 # 25 Hz frequency
time_index = 0

# plot the initial scene
ax = plt.axes()
ax.set_xlim(-window_size,window_size)
ax.set_ylim(-window_size,window_size)
beacon = plt.Circle((0, 5), 0.1, fc='blue',ec="red")
robot = plt.Rectangle((-length/2, -width/2), length, width, fc='blue',ec="red")
plt.gca().add_patch(robot)
plt.gca().add_patch(beacon)
plt.axis('scaled')
fig = plt.figure("animation")
im = plt.imshow(robot_state_bodyFromOrigin_BodyFrame[0:3])

# Test Scenario Simulation
# Step 1: accelerate until robot reaches its max speed in the body x axis
# NOTE: this can be integrated as an x_dot = Ax + Bu system, but A is a sparse
# matrix, so computational power is saved by doing the simulation this way.
state_history = np.array([[0], [0], [0]])
time_history = np.array([0])
current_time_s = 0.0;
while np.linalg.norm(robot_state_bodyFromOrigin_BodyFrame[6:9]) < v_max_mps : # while the norm of the translational velocity is below the max

    # Integrate the equations of motion
    robot_state_bodyFromOrigin_BodyFrame[0:3] = robot_state_bodyFromOrigin_BodyFrame[0:3] + np.multiply(robot_state_bodyFromOrigin_BodyFrame[6:9], time_step)
    robot_state_bodyFromOrigin_BodyFrame[6:9] = robot_state_bodyFromOrigin_BodyFrame[6:9] + np.multiply(np.array([[a_mps2], [0], [0]]), time_step)

    # Save and update time
    current_time_s = current_time_s + time_step


    # Add state vector at time step to a variable for the animation
    state_history = np.c_[state_history, robot_state_bodyFromOrigin_BodyFrame[0:3]]

    print(np.linalg.norm(robot_state_bodyFromOrigin_BodyFrame[6:9]))

def init():
    ax.add_patch(robot)
    return robot,
def animate(i):
    patch = patches.Rectangle((state_history[0,i], state_history[1,i]), 1.2, 1.0, fc ='y',angle = -np.rad2deg(yaw[i]))
    return patch,
anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=360,
                               interval=1,
                               blit=True)
plt.show()
