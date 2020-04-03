import numpy as np

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """
        # STUDENT CODE HERE
        self.T = 2 # 2 is original
        self.points = points
        self.maxVel = 2
    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE

        # return the goal
        if t == np.inf:
            x = self.points[-1, :]
        else:
            # check how many times t wraps around
            count = int(t // self.T)
            if count >= len(self.points) - 1:
                count = len(self.points) - 2
                # when it is the end, first point is just the second point
                first_pt = self.points[count + 1, :]
            else:
                # first point before the end
                first_pt = self.points[count, :]

            t = t % self.T
            # second point
            second_pt = self.points[count + 1, :]
            # calculate the unit velocity vector
            velocity_vec = (second_pt - first_pt) / self.T
            x_dot = velocity_vec
            # update the point
            x = first_pt + velocity_vec * t


        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
