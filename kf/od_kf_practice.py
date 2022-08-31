# This script will simulate a simple orbit, simulate sensed data, design a 
# Kalman Filter, and use it to determine the orbit.

import matplotlib.pyplot as plt
import numpy as np
import math

# Physical constants
mu = 4e5

# Initial orbit state
x = np.array([[9000], [0], [0]])
v = np.array([[0], [5], [0]])

def plot_orbit(x, v, period, dt) :
    x_list = x
    times = np.arange(0, period, dt)
    for time in times :
        ddr = -mu*x/np.linalg.norm(x)**3
        v = v + ddr*dt
        x = x + v*dt
        x_list = np.c_[x_list, x]
    
    plt.plot(x_list[0,:], x_list[1,:])
    plt.show()


def get_sma(x, v) :
    return 1 / -(np.linalg.norm(v)**2 / mu - 2/np.linalg.norm(x))

def get_period(x, v, mu) :
    sma = get_sma(x, v)
    return 2*math.pi*math.sqrt(sma**3 / mu)


plot_orbit(x, v, get_period(x, v, mu), 0.1)