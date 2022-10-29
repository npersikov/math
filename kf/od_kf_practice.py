# This script will simulate a simple orbit, simulate sensed data, design a 
# Kalman Filter, and use it to determine the orbit.

import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import random

# Physical constants
mu = 4e5

# Initial orbit state
x = np.array([[9000], [0], [0]])
v = np.array([[0], [5], [0]])

def orbit_determination(x, v, period, dt) :
    # The list of real state vectors
    x_list = x
    v_list = v

    # Standard deviations of position and velocity measurments
    sigma_x = 10 # 10 km
    sigma_v = 0.01 # 10 meters

    # The list of measured state vectors
    #TODO change the nav data to give range and range-rate information
    x_nav_list = np.add(x, random.normal(0, sigma_x, size=(3,1)))
    v_nav_list = np.add(v, random.normal(0, sigma_v, size=(3,1)))

    times = np.arange(0, period, dt)
    for time in times :
        ddr = -mu*x/np.linalg.norm(x)**3
        v = v + ddr*dt
        x = x + v*dt

        # Save real state
        x_list = np.c_[x_list, x]
        v_list = np.c_[v_list, v]

        # Save measured state
        x_nav_list = np.c_[x_nav_list, np.add(x, random.normal(0, sigma_x, size=(3,1)))]
        v_nav_list = np.c_[v_nav_list, np.add(v, random.normal(0, sigma_v, size=(3,1)))]

    plt.plot(x_nav_list[0,:], x_nav_list[1,:])
    plt.show()

def predict_state(x_nav_list, v_nav_list)
    # Make a state vector array for easy manipulation
    states = np.vstack(x_nav_list, v_nav_list)

    # Set up the matrices for kalman filter setup
    B = np.zeros(6,1) # Convert to new predicted state from input vector
    F = []
    H = []
    P = []
    Q = np.zeros(6,1) # additional environment uncertainty constant matrix
    R = []
    K = []

    # Save estimated states
    state_best_guesses = []

    # Loop through state measurment time history to generate 
    # kalman filtered estimation data
    for state in state :
        # Predict next best guess

        # Create next best guess using prediction and measurment data
        next_best_guess = M*P*Q #TODO change this its not this!!!!!!

        state_best_guesses.append(next_best_guess)

    return state_best_guesses


def plot_orbit(x, v, period, dt) :
    x_list = x
    times = np.arange(0, period, dt)
    for time in times :
        ddr = -mu*x/np.linalg.norm(x)**3
        v = v + ddr*dt
        x = x + v*dt
        x_list = np.c_[x_list, x]
        v_list = np.c_[v_list, v]
    
    plt.plot(x_list[0,:], x_list[1,:])
    plt.show()


def get_sma(x, v) :
    return 1 / -(np.linalg.norm(v)**2 / mu - 2/np.linalg.norm(x))

def get_period(x, v, mu) :
    sma = get_sma(x, v)
    return 2*math.pi*math.sqrt(sma**3 / mu)

orbit_determination(x, v, get_period(x, v, mu), 0.1)