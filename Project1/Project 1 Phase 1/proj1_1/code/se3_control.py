import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2


        # STUDENT CODE HERE

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        # Controller using Geometric Nonlinear Controller
        # initialize the control parameters Kd, Kp, Kr, and Kw
        # Kd = np.diag(np.array([12, 12, 400]))
        # Kp = np.diag(np.array([20, 20, 2000]))
        # KW = np.diag(np.array([100, 100, 6.5]))
        # KR = np.diag(np.array([1000, 1000, 40]))

        Kd = np.diag(np.array([15, 15, 20])) # z: 20
        Kp = np.diag(np.array([20, 20, 5])) # z: 5
        KW = np.diag(np.array([500, 500, 100]))  # 10 ，
        KR = np.diag(np.array([8000, 8000, 1000])) # 15，
        # First of all, compute desire rddot (acceleration)
        rddot_des = flat_output['x_ddot'] - Kd @ (state['v'] - flat_output['x_dot']) - Kp @ (state['x'] - flat_output['x'])
        # use desired rddot to compute the total commanded desired force F expressed in the inertia frame
        F_des = self.mass * rddot_des + np.array([0, 0, self.mass * self.g])
        F_des = F_des.transpose()
        # Get the rotation matrix from the quaternion
        R = Rotation.from_quat(state['q'])
        R = R.as_matrix()
        # calculate b3 unit vector axis expressed in the inertia frame
        b3 = R @ (np.array([0, 0, 1]).transpose())
        # Then calculate control input u1
        u1 = b3.transpose() @ F_des

        #------Start calculating desired rotation matrix R_des------
        # first compute the unit vector b3_des
        b3_des = F_des / np.linalg.norm(F_des)
        # then compute the unit vector b2_des
        # Yaw heading vector a_psi
        a_psi = np.array([np.cos(flat_output['yaw']), np.sin(flat_output['yaw']), 0]).transpose()
        b2_des = np.cross(b3_des, a_psi) / np.linalg.norm(np.cross(b3_des, a_psi))
        # compute the unit vector b1_des
        b1_des = np.cross(b2_des, b3_des)
        # concatenate R_des
        R_des = np.concatenate([b1_des.reshape(3,1), b2_des.reshape(3,1), b3_des.reshape(3,1)], axis = 1)
        #R_des = np.array([b1_des, b2_des, b3_des])
        # Compute rotation error matrix eR
        eR = (R_des.transpose() @ R - R.transpose() @ R_des)
        eR = 0.5 * (np.array([eR[2, 1], eR[0, 2], eR[1, 0]]).transpose())
        # Compute angular velocity error matrix eW
        w_des = np.array([0, 0, 0]).transpose()
        eW = state['w'] - w_des
        # Compute control input u2
        # I = np.array([[self.Ixx, 0, 0], [0, self.Iyy, 0], [0, 0, self.Izz]])
        u2 = self.inertia @ (-1 * KR @ eR - KW @ eW)

        # concatenate control inputs u1 and u2 into u
        u = np.append(u1, u2)
        gamma = self.k_drag / self.k_thrust
        # Define A matrix in equation u = AF
        A = np.array([[1, 1, 1, 1],
                     [0, self.arm_length, 0, -self.arm_length],
                     [-self.arm_length, 0, self.arm_length, 0],
                     [gamma, -gamma, gamma, -gamma]])
        # solve for F1, F2, F3, and F4
        F = np.linalg.inv(A) @ u

        # prevent the thrust being negative
        for index, f in enumerate(F):
            if f < 0:
                F[index] = 0


        # convert F into motor speeds
        cmd_motor_speeds = np.sqrt(F / self.k_thrust)


        #dir_vector = F / abs(F)
        #cmd_motor_speeds = cmd_motor_speeds * dir_vector
        print('Force: ')
        print(F)
        print('---------------')
        print(cmd_motor_speeds)







        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
