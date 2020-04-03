from heapq import heappush, heappop  # Recommended.
import numpy as np

from flightsim.world import World
from proj1_2.code.occupancy_map import OccupancyMap # Recommended.

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    occ_map.create_map_from_world()
    all_cost = np.zeros(occ_map.map.shape) + 10000 # add a dummy large number for representing inf.
    all_cost[start_index] = 0 # add 0 cost for the start
    # heapq stores the tuple with cost and index
    Q = [] # heapq storing all the neighbor points
    if astar is True:
        h0 = np.sqrt((resolution[0] * (start_index[0] - goal_index[0])) ** 2
                                             + (resolution[1] * (start_index[1] - goal_index[1])) ** 2
                                             + (resolution[2] * (start_index[2] - goal_index[2])) ** 2)
    else:
        h0 = 0
    heappush(Q, (0 + h0, start_index)) # push the start point
    parent = dict() # initialize a empty dict for storing parent nodes


    # finding the path
    while Q:
        cur_node = heappop(Q)
        # if cur_node[1] is goal_index:
        #     print('Goal cost is: ', cur_node[0])
        #     break
        for neighbors in find_neighbors(cur_node[1]):
            if occ_map.is_valid_index(neighbors) and not occ_map.is_occupied_index(neighbors):
                if astar is True:
                    h = np.sqrt((resolution[0] * (neighbors[0] - goal_index[0])) ** 2
                                             + (resolution[1] * (neighbors[1] - goal_index[1])) ** 2
                                             + (resolution[2] * (neighbors[2] - goal_index[2])) ** 2)
                else:
                    h = 0
                cost = all_cost[cur_node[1]] + np.sqrt((resolution[0] * (neighbors[0] - cur_node[1][0])) ** 2
                                             + (resolution[1] * (neighbors[1] - cur_node[1][1])) ** 2
                                             + (resolution[2] * (neighbors[2] - cur_node[1][2])) ** 2)
                if cost < all_cost[neighbors]:
                    heappush(Q, (cost + h, neighbors))  # child found, push into heap
                    all_cost[neighbors] = cost  # update the cost matrix
                    parent[neighbors] = cur_node[1]
    # print('Is goal still in dict? ', goal_index in parent.keys())
    # check if the path exists
    if goal_index in parent.keys():
        pass
    else:
        return None
    # traverse the parent nodes
    parent_key = goal_index # initialize the parent key as goal_index
    path = list() # initialize the path list
    while True: #goal_index in parent.keys():
        parent_key = parent[parent_key] # update the next parent key
        if parent_key is start_index:
            break
        parent_in_metric = occ_map.index_to_metric_center(parent_key)
        path.append(tuple(parent_in_metric))

    path.reverse() # reverse the order back
    path.insert(0, start)
    path.append(goal)
    path = np.asarray(path)
    # print('the path is: ', path)
    return path


def find_neighbors(point):
    """
    :param point: a point index, a tuple type
    :return: all of its neighbors, a tuple type consisting of all the neighbors in tuple type
    """
    step = (-1, 1, 0) # step size for three axes

    neighbors = []
    for i in step:
        for j in step:
            for k in step:
                neighbors.append(tuple((point[0] + i, point[1] + j, point[2] + k)))
    # in this case, the last element in the neighbors will be point itself, so do not return it

    return neighbors[:-1]
