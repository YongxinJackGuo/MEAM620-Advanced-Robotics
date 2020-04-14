# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


# %%

def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """

    #TODO Your code here - replace the return value with one you compute

    # First normalize the linear acceleration by gravitational constant g of 9.81 m/s^2
    linear_acceleration = linear_acceleration / 9.81

    # Compute estimated rotation after dt
    #print(Rotation.from_rotvec(angular_velocity * dt).as_matrix())
    R_est = initial_rotation.as_matrix() @ Rotation.from_rotvec(angular_velocity * dt).as_matrix()

    # Get the linear acceleration vector expressed in initial frame
    g_est = R_est @ linear_acceleration
    # normalize
    g_est = ( g_est / norm(g_est) ).reshape(3,1)

    # compute rotation correction R_acc in quaternion
    R_acc = np.array([0,
                      g_est[2]/np.sqrt(2 * (1 + g_est[0])),
                      -g_est[1]/np.sqrt(2 * (1 + g_est[0])),
                      np.sqrt((1 + g_est[0]) / 2) ])

    # Compute sensed acceleration error e_m and then compute alpha for adaptive gain compensation
    e_m = abs(norm(linear_acceleration) - 1)

    if e_m < 0.1:
        alpha = 1
    elif e_m < 0.2:
        alpha = -10 * e_m + 2
    else:
        alpha = 0

    # Update the R_acc and re-normalize to unit quaternion
    R_acc = (1 - alpha) * np.array([0, 0, 0, 1]) + alpha * R_acc
    R_acc = R_acc / norm(R_acc)  # covert to unit quaternion
    R_acc = Rotation.from_quat(R_acc).as_matrix()

    # Compute final correct rotation
    R = Rotation.from_matrix(R_acc @ R_est)

    return R
