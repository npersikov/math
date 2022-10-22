# Navigation Design Challenge for SpaceRyde GNC Engineer Application

# Used imports
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import random
import matplotlib.patches as patches
import matplotlib.animation as animation

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

# Plot util vars
window_size = 10

# Simulation variables
time_step = 1.0/25.0 # 25 Hz frequency
time_index = 0

# plot the initial scene
fig = plt.figure()
ax = plt.axes()
ax.set_xlim(-2,25)
ax.set_ylim(-2,7)
ax.set_aspect('equal', adjustable='box')
beacon = plt.Circle((0, 5), 0.1, fc='blue',ec="red")
robot = patches.Rectangle((0, 0), 0, 0, fc='y')

# ======================== Test Scenario Simulation ===========================

# Step 1: accelerate until robot reaches its max speed in the body x axis
# NOTE: this can be integrated as an x_dot = Ax + Bu system, but A is a sparse
# matrix, so computational power is saved by doing the simulation this way.
state_history   = robot_state_bodyFromOrigin_BodyFrame
time_history    = np.array([0.0])
current_time_s  = 0.0;

# while the norm of the translational velocity is below the max
while np.linalg.norm(robot_state_bodyFromOrigin_BodyFrame[6:9]) < v_max_mps :

    # Integrate the equations of motion
    robot_state_bodyFromOrigin_BodyFrame[0:3] = robot_state_bodyFromOrigin_BodyFrame[0:3] + np.multiply(robot_state_bodyFromOrigin_BodyFrame[6:9], time_step)
    robot_state_bodyFromOrigin_BodyFrame[6:9] = robot_state_bodyFromOrigin_BodyFrame[6:9] + np.multiply(np.array([[a_mps2], [0], [0]]), time_step)

    # If the simulation reached a point where the velocity is greater than the
    # maximum possible value, limit the velocity to the max value but keep the
    # same direction vector
    if np.linalg.norm(robot_state_bodyFromOrigin_BodyFrame[6:9]) > v_max_mps :
        robot_state_bodyFromOrigin_BodyFrame[6:9] = robot_state_bodyFromOrigin_BodyFrame[6:9] / np.linalg.norm(robot_state_bodyFromOrigin_BodyFrame[6:9]) * v_max_mps

    # Save and update time
    current_time_s = current_time_s + time_step
    time_history = np.append(time_history, current_time_s)

    # Add state vector at time step to a variable for the animation
    state_history = np.c_[state_history, robot_state_bodyFromOrigin_BodyFrame]

    # print(np.linalg.norm(robot_state_bodyFromOrigin_BodyFrame[6:9]))
    # print(robot_state_bodyFromOrigin_BodyFrame[5,0])



additional_time_s = 5
time_at_max_speed = current_time_s
for t in range(int(additional_time_s/time_step)) :

    # Integrate the equation of motion
    robot_state_bodyFromOrigin_BodyFrame[0:3] = robot_state_bodyFromOrigin_BodyFrame[0:3] + np.multiply(robot_state_bodyFromOrigin_BodyFrame[6:9], time_step)

    # Save and update time
    current_time_s = current_time_s + time_step
    time_history = np.append(time_history, current_time_s)

    # Add state vector at time step to a variable for the animation
    state_history = np.c_[state_history, robot_state_bodyFromOrigin_BodyFrame]

def init():
    ax.add_patch(robot)
    return robot,

def animate(i):
    robot.set_width(width)
    robot.set_height(length)
    robot.set_xy([state_history[0,i], state_history[1,i]])
    robot.angle = np.rad2deg(state_history[5, i]) # angle rotated about z axis
    print('speed: ' + str(np.linalg.norm(state_history[6:9, i])))
    print('time: ' + str(time_history[i]))
    return robot,

anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(state_history[0]), interval = 500, blit = True, repeat = False)
plt.show()
