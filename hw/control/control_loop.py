import numpy as np
from abc import ABC, abstractmethod

'''
Abstract ControlLoop Class
Handles updating the current waypoint as the controller traverses a trajectory.
Input:
    waypoints: list of numpy arrays, each with [x,y]
    dt: time between each iteration of the control loop
    threshold: distance that is "close enough" to the goal, when to consider the next waypoint
'''
class ControlLoop(ABC):

    def __init__(self, waypoints, dt, threshold=0.1, debug=False):
        self.dt = dt
        self.threshold = threshold
        self.debug = debug
        self.is_finished = False

        self.set_waypoint(waypoints[0])
        self.remaining_waypoints = waypoints[1:]

    # Check if car is within self.threshold distance of the waypoint
    def reached_waypoint(self, state):
        cur_pos = state[:2]
        dist = np.linalg.norm(cur_pos - self.cur_waypoint)
        return dist <= self.threshold

    # Update to the next waypoint, unless trajectory is finished.
    def next_waypoint(self):
        if len(self.remaining_waypoints) == 0:
            self.is_finished = True # Outer loop should check for is_finished
        else:
            self.set_waypoint(self.remaining_waypoints.pop(0))

    # Set cur_waypoint. Overriden by child classes
    def set_waypoint(self, waypoint):
        self.cur_waypoint = waypoint

    # Wraps calc_action by first updating the waypoint if necessary.
    def get_action(self, state):
        if self.reached_waypoint(state):
            if self.debug:
                print("Arrived at {}".format(self.cur_waypoint))

            self.next_waypoint()

        return self.calc_action(state)

    # Computation to determine an action from state and self.cur_waypoint.
    # Implemented in child classes
    @abstractmethod
    def calc_action(self, state):
        return NotImplemented
