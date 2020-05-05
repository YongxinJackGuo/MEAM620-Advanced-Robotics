#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE

    # Update the nomial states according to Newton's Law of Motion
    R = q.as_matrix()
    new_p = p + v * dt + 0.5 * (R @ (a_m - a_b) + g) * dt**2
    new_v = v + (R @ (a_m - a_b) + g) * dt
    rotvec = (w_m - w_b) * dt
    new_q = q * Rotation.from_rotvec(rotvec.flatten())  # Expressed in Rotation Object
    # Biases and gravity keep the same throughout the course
    a_b = a_b
    w_b = w_b
    g = g

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    # Update the error state covariance matrix using P = FxPFx.t + FiQiFi.t

    # 1. Construct Fx Matrix
    I, Z = np.eye(3), np.zeros((3,3))
    R = q.as_matrix()
    a = a_m - a_b
    a_ss = np.array([[0, -a[2, 0], a[1, 0]],
                     [a[2, 0], 0, -a[0, 0]],
                     [-a[1, 0], a[0, 0], 0]])
    w = (w_m - w_b) * dt
    R_w = Rotation.from_rotvec(w.flatten()).as_matrix()
    Fx = np.block([[I, I * dt, Z, Z, Z, Z],
                   [Z, I, -R @ a_ss * dt, -R * dt, Z, I * dt],
                   [Z, Z, R_w.T, Z, -I*dt, Z],
                   [Z, Z, Z, I, Z, Z],
                   [Z, Z, Z, Z, I, Z],
                   [Z, Z, Z, Z, Z, I]])
    # 2. Construct Fi Matrix
    Fi = np.block([[Z, Z, Z, Z],
                   [I, Z, Z, Z],
                   [Z, I, Z, Z],
                   [Z, Z, I, Z],
                   [Z, Z, Z, I],
                   [Z, Z, Z, Z]])
    # 3. Construct Qi Matrix
    Vi = accelerometer_noise_density**2 * dt**2 * I
    Theta_i = gyroscope_noise_density**2 * dt**2 * I
    Ai = accelerometer_random_walk**2 * dt * I
    Omega_i = gyroscope_random_walk**2 * dt * I
    Qi = np.block([[Vi, Z, Z, Z],
                   [Z, Theta_i, Z, Z],
                   [Z, Z, Ai, Z],
                   [Z, Z, Z, Omega_i]])

    # Compute new P
    new_error_state_covariance = Fx @ error_state_covariance @ Fx.T + Fi @ Qi @ Fi.T

    # return an 18x18 covariance matrix
    return new_error_state_covariance


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    R = q.as_matrix()
    Pc = R.T @ (Pw - p)
    nominal_image_pos = np.array([Pc[0], Pc[1]]) / Pc[2]
    innovation = uv - nominal_image_pos
    # Determine if the point is a outlier, return origin value if yes
    if norm(innovation) > error_threshold:
        return nominal_state, error_state_covariance, innovation
    # If it is inlier, continue the updates
    # Compute the Jacobian of measurement w.r.t the error states. Shape 2-by-18
    # dz/dPc. Derivative of measurment w.r.t camera coordinates
    dz_dPc = (1 / Pc[2, 0]) * np.array([[1, 0, -Pc[0, 0] / Pc[2, 0]],
                                     [0, 1, -Pc[1, 0] / Pc[2, 0]]])
    # dz/dtheta = dz/dPc * dPc/dtheta. Derivative of measurment w.r.t the orientation error
    dz_dtheta = dz_dPc @ np.array([[0, -Pc[2, 0], Pc[1, 0]],
                                   [Pc[2, 0], 0, -Pc[0, 0]],
                                   [-Pc[1, 0], Pc[0, 0], 0]])

    # dz/dp = dz/dPc * dPc/dp. Derivative of measurment w.r.t the position error
    dz_dp = dz_dPc @ -R.T
    # Construct H_t
    Z = np.zeros((2, 3))
    H_t = np.block([dz_dp, Z, dz_dtheta, Z, Z, Z])
    # Extended Kalman Filter (EKF) update section

    # 1. Compute Kalman Gain, K
    #print((H_t @ error_state_covariance @ H_t.T + Q))
    K_t = error_state_covariance @ H_t.T @ inv(H_t @ error_state_covariance @ H_t.T + Q)
    # 2. Update error state
    new_error_state = K_t @ innovation
    # 3. Update error state covariance
    I_18 = np.eye(18)
    error_state_covariance = (I_18 - K_t @ H_t) @ error_state_covariance @ (I_18 - K_t @ H_t).T + K_t @ Q @ K_t.T

    # update the nomial states by adding new_error_state to it
    p = p + new_error_state[0: 3]
    v = v + new_error_state[3: 6]
    q = q * Rotation.from_rotvec(new_error_state[6:9].flatten())
    a_b = a_b + new_error_state[9: 12]
    w_b = w_b + new_error_state[12: 15]
    g = g + new_error_state[15:18]


    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
