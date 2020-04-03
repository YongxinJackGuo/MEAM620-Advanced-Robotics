import numpy as np

from proj1_3.code.graph_search import graph_search
from proj1_3.code.occupancy_map import OccupancyMap
import time

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        self.world = world
        self.max_sparse_dist = 1 # 1.8
        self.max_speed = 2.3 # 2.5
        self.points = self.get_sparse_points(self.path)
        self.time_param = self.get_time_segment(self.points)
        self.coefficients = self.get_spline_coeff(self.points, self.time_param[0])

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
        if t == np.inf:
            x = self.points[-1, :]
        # Find out which segment the drone should be now at t
        for seg_num in range(len(self.time_param[1])):
            if t <= self.time_param[1][seg_num]:
                cur_seg = seg_num
                break

        if t > self.time_param[1][-1]:
            cur_seg = len(self.time_param[1]) - 1

        if t is not np.inf:
            # Scale the time to 0 start
            temp_time_cum = np.insert(self.time_param[1], 0, 0)
            t = t - temp_time_cum[cur_seg]
            # define the current coefficient matrix
            cur_coeff_mat = np.array([[t**5, t**4, t**3, t**2, t, 1],
                                      [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0],
                                      [20*t**3, 12*t**2, 6*t, 2, 0, 0],
                                      [60*t**2, 24*t, 6, 0, 0, 0],
                                      [120*t, 24, 0, 0, 0, 0] ])

            # compute the states using the coefficients. Note: 6 coefficients per segment
            start_index = cur_seg * 6
            # x-component
            state_x = cur_coeff_mat @ self.coefficients[0][start_index: start_index + 6]
            # y-component
            state_y = cur_coeff_mat @ self.coefficients[1][start_index: start_index + 6]
            # z-component
            state_z = cur_coeff_mat @ self.coefficients[2][start_index: start_index + 6]

            # combine all 3 axes states
            all_state = np.concatenate((state_x, state_y, state_z), axis = 1)
            x = all_state[0, :]
            x_dot = all_state[1, :]
            x_ddot = all_state[2, :]
            x_dddot = all_state[3, :]
            x_ddddot = all_state[4, :]

        if t >= self.time_param[1][-1] - 1.5:
            x = self.points[-1, :]
            x_dot = np.zeros((3,))
            x_ddot = np.zeros((3,))
            x_dddot = np.zeros((3,))
            x_ddddot = np.zeros((3,))





        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output

    def get_sparse_points(self, path):
        """

        :param path: a path returned by astar algorithms
        :return: a trimmed path consists of sparse waypoints
        """
        sparse_points = []
        segment = len(self.path) - 1 # get segment number
        start_pt = self.path[0, :]
        goal_pt = self.path[-1, :]
        sparse_points.append(start_pt.tolist())
        occ_map = OccupancyMap(self.world, self.resolution, self.margin)

        for cur_seg_num in range(segment):
            dist = np.sqrt( sum((self.path[cur_seg_num + 1, :] - start_pt)**2) )
            if self.has_line_collided(start_pt, self.path[cur_seg_num + 1, :], occ_map):
                start_pt = self.path[cur_seg_num, :]
                sparse_points.append(start_pt.tolist())
                goal_pt_exist = False # since we only append the previous pt
            else:
                if (dist > self.max_sparse_dist) or ( (cur_seg_num + 1) is segment): # end pt reached or max dist exceeded
                    start_pt = self.path[cur_seg_num + 1, :]
                    sparse_points.append(start_pt.tolist())
                goal_pt_exist = True # if not collided, goal pt must be appended

        if goal_pt_exist is False:
            sparse_points.append(goal_pt.tolist())
        # remove the points near the goal
        # del sparse_points[-2]
        # del sparse_points[-3]
        # del sparse_points[-4]
        sparse_points = np.asarray(sparse_points)
        return sparse_points

    def has_line_collided(self, start_pt, end_pt, occ_map):
        # set collision detecting parameter
        step_length = 1/10
        dist = np.sqrt(sum((end_pt - start_pt) ** 2))  # get distance
        num_steps = (np.floor(dist / step_length)).astype('int32')
        line_resolution = 1 / num_steps
        unit_vec = (end_pt - start_pt) / dist # get unit vector direction
        has_collided = False

        for step in range(num_steps):
            cur_mid_pt = start_pt + (step * line_resolution) * unit_vec # incrementally update the mid_pt
            if occ_map.is_occupied_metric(tuple(cur_mid_pt)):
                has_collided = True
                break

        return has_collided


    def get_spline_coeff(self, sparse_pts, t_segment):
        """
        :param sparse_pts: an array of sparse points extracted from the astar path along the trajectory
                t_segment: an array of time duration for each segment
        :return: an array of coefficients of spline trajectory
        """
        # # Implementation of Minimum Jerk Trajectory. 5th degree polynomial with 6 unknowns for each segments
        seg_num = len(sparse_pts) - 1
        mid_pt_num = seg_num - 1
        col_num = row_num = 6 * seg_num # 6 unknowns per segment
        A = np.zeros((row_num, col_num))
        #constraint_matrix = np.zeros((row_num, col_num))
        bx = np.zeros((row_num, 1))
        by = np.zeros((row_num, 1))
        bz = np.zeros((row_num, 1))
        mid_pt = sparse_pts[1:-1, :] # get the mid pt
        for mid_pt_index in range(mid_pt_num):
            t = t_segment[mid_pt_index]
            row_start = 3 + mid_pt_index * 6
            row_end = row_start + 6
            col_start = mid_pt_index * 6
            col_end = col_start + 12
            # A[row_start: row_end, col_start: col_end] = \
            #     np.array([ [t**5, t**4, t**3, t**2, t, 1, 0, 0, 0, 0, 0, 0],
            #              [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0, 0, 0, 0, 0, -1, 0],
            #              [20*t**3, 12*t**2, 6*t, 2, 0, 0, 0, 0, 0, -2, 0, 0],
            #              [60*t**2, 24*t, 6, 0, 0, 0, 0, 0, -6, 0, 0, 0],
            #              [120*t, 24, 0, 0, 0, 0, 0, -24, 0, 0, 0, 0],
            #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] ])
            A[row_start: row_end, col_start: col_end] = \
                np.array([[t ** 5, t ** 4, t ** 3, t ** 2, t, 1, 0, 0, 0, 0, 0, 0],
                          [5 * t ** 4, 4 * t ** 3, 3 * t ** 2, 2 * t, 1, 0, 0, 0, 0, 0, -1, 0],
                          [20 * t ** 3, 12 * t ** 2, 6 * t, 2, 0, 0, 0, 0, 0, -2, 0, 0],
                          [60 * t ** 2, 24 * t, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

            # specify x y z axis b column array
            bx[row_start: row_end] = np.array([mid_pt[mid_pt_index, 0], 0, 0, 0, 0, mid_pt[mid_pt_index, 0]]).reshape(6, 1)
            by[row_start: row_end] = np.array([mid_pt[mid_pt_index, 1], 0, 0, 0, 0, mid_pt[mid_pt_index, 1]]).reshape(6, 1)
            bz[row_start: row_end] = np.array([mid_pt[mid_pt_index, 2], 0, 0, 0, 0, mid_pt[mid_pt_index, 2]]).reshape(6, 1)


        # Assign elements to start and goal pts for A, bx, by, and bz
        t_f = t_segment[-1]
        A[0: 3, 0: 6] = np.array([ [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 2, 0, 0] ])
        A[-3:, -6:] = np.array( [ [t_f**5, t_f**4, t_f**3, t_f**2, t_f, 1],
                                  [5*t_f**4, 4*t_f**3, 3*t_f**2, 2*t_f, 1, 0],
                                  [20*t_f**3, 12*t_f**2, 6*t_f, 2, 0, 0] ] )
        bx[0:3] = np.array([sparse_pts[0, 0], 0, 0]).reshape(3, 1)
        bx[-3:] = np.array([sparse_pts[-1, 0], 0, 0]).reshape(3, 1)
        by[0:3] = np.array([sparse_pts[0, 1], 0, 0]).reshape(3, 1)
        by[-3:] = np.array([sparse_pts[-1, 1], 0, 0]).reshape(3, 1)
        bz[0:3] = np.array([sparse_pts[0, 2], 0, 0]).reshape(3, 1)
        bz[-3:] = np.array([sparse_pts[-1, 2], 0, 0]).reshape(3, 1)
        coeff_x = np.linalg.inv(A) @ bx
        coeff_y = np.linalg.inv(A) @ by
        coeff_z = np.linalg.inv(A) @ bz

        return (coeff_x, coeff_y, coeff_z)

    def get_time_segment(self, sparse_pts):
        """
        :param sparse_pts: a array of sparse points extracted from the astar path along the trajectory
        :return: a tuple consisting 2 elements: 1). time required by each segment. 2). cumulative time array for segment
        targeting
        """
        seg_num = len(sparse_pts) - 1
        dist_seg = np.zeros((seg_num, 3))
        # calculate the distance from each sparse pts(except the first pt) to the first pt
        for m in range(len(sparse_pts) - 1):
            dist_seg[m, :] = sparse_pts[m + 1, :] - sparse_pts[m, :]

        dist = np.sqrt((np.sum(dist_seg ** 2, axis = 1)).reshape(seg_num, 1))
        t_segment = dist / self.max_speed # time segment
        t_segment[-1] = 6 * t_segment[-1] # double the last segment time
        t_segment[-2] = 4 * t_segment[-2]  # double the last segment time
        t_segment[-3] = 4 * t_segment[-3]
        t_segment[-4] = 2 * t_segment[-4]
        t_cum = np.cumsum(t_segment).reshape(seg_num, 1) # cumulative time array
        # check if the any time segment is near 0
        t_segment = np.where(t_segment < 0.001, 0.5, t_segment)
        return (t_segment, t_cum)



