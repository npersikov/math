# This script will simulate a simple orbit, simulate sensed data, design a
# Kalman Filter, and use it to determine the orbit.

import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import random
from numpy import linalg

# Physical constants
mu = 4e5

# Initial orbit state
x = np.array([[9000], [0], [0]])
v = np.array([[0], [5], [0]])

# This function integrates the orbit and plots it
def orbit_determination(x, v, period, dt) :
    # The list of real state vectors
    x_list = x
    v_list = v

    # Standard deviations of position and velocity state uncertainty
    sigma_x = 1 # 1 km
    sigma_v = 0.01 # 10 mps

    # The list of measured state vectors
    #TODO change the nav data to give range and range-rate information
    x_nav_list = np.add(x, random.normal(0, sigma_x, size=(3,1)))
    v_nav_list = np.add(v, random.normal(0, sigma_v, size=(3,1)))

    # =========== Initial Values for State Estimation ===========
    x_0 = np.vstack(x, v)
    P_0 = np.diag([sigma_x**2, sigma_x**2, sigma_x**2, sigma_v**2, sigma_v**2, sigma_v**2])
    H   = get_H(x_0)
    Q   = np.zeros((len(x_0), len(x_0)))

    # This one has output x output dimensions. Measurment noise covariance. The
    # assumed measurment variables are cartesian velocity components, range, and
    # range-rate.
    # NOTE: consider updating for accelerometer data.
    R   = np.diag([sigma_v, sigma_v, sigma_v, 3*(sigma_x**2), 3*(sigma_v**2)])

    # ===========================================================

    # Simulate orbit for a period
    times = np.arange(0, period, dt)
    x_vec = x_0
    x_vec_u = x_0
    P_u = P_0
    for time in times :

        # ============ State Estimation ===============

        # Predict
        stm = get_stm()
        x_p = predict_next_state(x_vec_u, stm)
        P_p = predict_next_covariance(P_u, stm, Q)

        # Update


        # =============================================

        # Incremental orbit physics integration
        ddr = -mu*x/np.linalg.norm(x)**3
        v   = v + ddr*dt
        x   = x + v*dt
        x_vec = np.vstack(x, v)

        # Save real state
        x_list = np.c_[x_list, x]
        v_list = np.c_[v_list, v]

        # Save measured state
        x_nav_list = np.c_[x_nav_list, np.add(x, random.normal(0, sigma_x, size=(3,1)))]
        v_nav_list = np.c_[v_nav_list, np.add(v, random.normal(0, sigma_v, size=(3,1)))]

    # Plot the period
    plt.plot(x_nav_list[0,:], x_nav_list[1,:])
    plt.show()

def get_stm(state) :


def predict_state(x_nav_list, v_nav_list) :
    # Make a state vector array for easy manipulation
    states = np.vstack(x_nav_list, v_nav_list)

    # Set up the matrices for kalman filter setup
    B = np.zeros(6,1) # Convert to new predicted state from input vector
    F = get_stm() # The STM from one orbit state to the next
    H = []
    P = np.eye(6) # Covariance matrix of orbit state TODO check what this should be
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
        ddr     = -mu*x/np.linalg.norm(x)**3
        v       = v + ddr*dt
        x       = x + v*dt
        x_list  = np.c_[x_list, x]
        v_list  = np.c_[v_list, v]

    plt.plot(x_list[0,:], x_list[1,:])
    plt.show()

# Get semi major axis
def get_sma(x, v) :
    return 1 / -(np.linalg.norm(v)**2 / mu - 2/np.linalg.norm(x))

# Get orbit period
def get_period(x, v, mu) :
    sma = get_sma(x, v)
    return 2*math.pi*math.sqrt(sma**3 / mu)

# Calculate the Kalman gain matrix once all required components are known.
# NOTE: the subscript p means "predict." Any variable with this subscript is at
# its state before it has been updated. "Updated" variables will have the
# subscript u.
#
# Interesting: the Kalman gain does vary since P_p varies, but in a linear case,
# the gain should converge. However, H varies too, so in this case K will
# probably never converge. Would be interesting to plot it.
#
#   P_p:    Predicted covariance matrix of the next step
#   H:      Partial derivative of mapping function h (from state to measurment)
#               wrt the state vector x
#   R:      The measurment noise covariance matrix
def get_kalman_gain(P_p, H, R):
    term_1  = np.matmul(P_p, np.transpose(H))
    term_2  = np.matmul(np.matmul(H, P_p), np.transpose(H))
    term_3  = np.inv(np.add(term_2, R))
    K       = np.matmul(term_1, term_3)
    return K

# Predict the next state using a state transition matrix and previous known
# state.
def predict_next_state(x_u_old, stm):
    x_p = np.matmul(stm, x_u_old)
    return x_p

# Update the predicted state with the filtered value.
def update_state_estimate(x_p, K, y, z_p):
    diff_term = np.subtract(y, z_p)
    x_u = np.add(x_p, np.matmul(K, diff_term))
    return x_u

# Predict the next state's covariance matrix using old one and process noise Q.
def predict_next_covariance(P_u_old, stm, Q):
    P_p = np.add(np.matmul(np.matmul(stm, P_u_old), np.transpose(stm)), Q)

# Update the predicted covariance matrix.
def update_covariance(K, H, P_p):
    mat_prod = np.matmul(np.matmul(K, H), P_p)
    P_u = np.add(P_p, mat_prod)
    return P_u

# H is the partial derivative of mapping function h (from state to measurment)
# wrt the state vector x. In this case, the measurment vector z was assumed to
# be [v_x; v_y; v_z; r; r_dot], in other words the velocity from an IMU, range,
# and range-rate.
#
# z can take on many forms, for example including IMU accelerometer data, and
# this would require redefinition of h and rederivation of H.
#
# x is assumed to be a 6 element vector of position and velocity cartesian
# components.
def get_H(x):
    # TODO figure out how to split up this line nicely.
    H = np.array([[0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [2*x(1), 2*x(2), 2*x(3), 0, 0, 0], [0, 0, 0, 2*x(4), 2*x(5), 2*x(6)]])

# Get the matrix needed to integrate the state transition matrix from one time
# to the next in an orbit.
#
# NOTE: if you want the STM from one state to another, you integrate the
# stm rate matrix from time 0 to time t_f. The stm rate matrix is the output of
# this function (dgdx) times the previous stm, which, at t = 0, is an identity
# matrix. So, if you want the stm from one time step to the next, you integrate
# from t = 0 to t = dt, so you just multiply dgdx by eye(6) and by dt to get the
# stm.
def get_stm_jacobian(r, mu):

    zeros   = np.zeros(len(r), len(r))
    eye     = np.identity(len(r))
    partial = np.identity(len(r))
    rows    = len(r)
    cols    = len(r)

    # The dgdx matrix is 6x6, with a zeros(3) anx eye(3) on the top, and a 3x3
    # matrix with analytically determined derivatives and zeros(3) on the
    # bottom. This for loop calculates the 3x3 matrix with derivatives.
    for row in range(rows) :
        for col in range(cols) :
            if row == col :
                partial(row, col) = (-mu*(np.linalg.norm(r)**2 - 3*r(row)**2)) / (np.linalg.norm(r)**5)
            else :
                partial(row, col) = 3*mu*r(row)*r(col)/(np.linalg.norm(r)**5)

    # Concatenate the four sub-matrices to get the stm stm (yes the stm stm).
    top     = np.hstack(zeros, eye)
    bot     = np.hstack(partial, zeros)
    dgdx    = np.vstack(top, bot)

    # Finally, return the stm stm
    return dgdx

orbit_determination(x, v, get_period(x, v, mu), 0.1)
