import json
import os
import numpy as np
import time
import sys

sys.path.append('../')

'''
Keypoint Tracker class
Used for maintaining the list of braking target locations,
as well as calculating the distance to the nearest in the
direction of motion.
'''
class Keypoint_Tracker():
    def __init__(self):
        #File path to where the keypoints are stored
        self.fp = os.path.join(self.bd_fp(), "keypoint_locations.json")

        #If the file doesn't exist yet start from scratch
        if (os.path.exists(self.fp)):
            self.keypoint_locations = json.load(open(self.fp, "r"))
        else:
            self.keypoint_locations = []

        #Distance Conversion Constant for standard duckietown map.
        self.DISTANCE_CONVERSION = 0.516

    '''
    Returns distance to nearest keypoint in the direction of motion
    '''
    def dist_to_nearest(self, x, y, angle):

        #Converts angles from degrees to radians and rotates
        angle_rad = 2 * np.pi - (angle * np.pi / 180)

        nearest_dist_to = None

        '''
        k_x = keypoint x location
        k_y = keypoint y location
        k_angle = direction car was facing when logged keypoint.

        For each keypoint we check if it is valid, and finally
        select the closest of all valid points.
        '''
        for i, (k_x, k_y, k_angle) in enumerate(self.keypoint_locations):

            '''
            Heading should be within 15 degrees of the heading of the
            car when the keypoint was marked.
            '''
            if (abs(angle - k_angle) > 15):
                continue

            '''
            We define the target line as the one that crosses through
            the keypoint and is perpendicular to the car's heading when
            the keypoint was logged.
            dist_to calculates the distance of the car to this line, by
            taking the projection of the point displacement on the normal
            vector.
            '''
            displacement = [k_x - x, k_y - y]
            normal = [np.cos(angle_rad), np.sin(angle_rad)]
            dist_to = np.dot(displacement, normal)

            '''
            If we are more than 0.02 units past the target line than
            this is invalid.
            The 0.02 is to allow a little buffer, generally the car
            will be in front of the target line.
            '''
            if (dist_to < -0.02):
                continue

            '''
            Offset is the horizontal component of the displacement.
            That is whatever part of the displacement is not normal
            to the target line.
            '''
            offset = np.sqrt(displacement[0] ** 2 + displacement[1] ** 2 - dist_to ** 2)

            '''
            If this offset is greater than 0.1 units we deem the
            target location invalid.
            '''
            if (offset > 0.1):
                continue

            '''
            Update nearest_dist so that it allows has the distance
            to closest valid target point if any.
            '''
            if (nearest_dist_to is None):
                nearest_dist_to = dist_to
            else:
                nearest_dist_to = min(nearest_dist_to, dist_to)

        '''
        Divide by the distance conversion constant to convert
        into our units.
        '''
        if (nearest_dist_to is None):
            return None
        else:
            return nearest_dist_to / self.DISTANCE_CONVERSION

    '''
    Adds a keypoint at the desired position and heading (angle).
    Pauses the simulation for a second so you don't add multiple
    times.
    '''
    def add_keypoint(self, x, y, angle):
        print("Adding keypoint...")
        time.sleep(1)
        self.keypoint_locations.append((x, y, angle))
        json.dump(self.keypoint_locations, open(self.fp, "w"))

    #File path to Braking Distance folder
    def bd_fp(self):
        import settings

        cur_dir = os.path.dirname(__file__)
        internal_path = os.path.join(cur_dir, '../../internal_only/braking_distance/')
        if settings.use_internal_if_available and os.path.exists(internal_path):
            return internal_path
        return cur_dir

'''
If this file is run directly the following code is run.
This code allows you to manage your keypoints if any.
Specifically you may delete an erroneously added keypoint
if you would like.
If you would like to start over you may delete all keypoints
by trashing the keypoint_locations.json file.
'''
if (__name__ == "__main__"):
    if (not os.path.exists("keypoint_locations.json")):
        print("You do not currently have any keypoints saved")
    else:
        tracker = Keypoint_Tracker()
        print("Keypoints:")
        for i, (x, y, angle) in enumerate(tracker.keypoint_locations):
            print(f"{i + 1}. (x: {x:.2f}, y: {y:.2f}); angle: {angle}")
        delete = input("Would you like to delete a keypoint? (y or n) ")
        if (delete == 'y'):
            to_delete = input(f"Which one would you like to delete (1 to {len(tracker.keypoint_locations)}) ")
            try:
                index = int(to_delete) - 1
                x, y, angle = tracker.keypoint_locations[index]
                print(f"Deleting (x: {x:.2f}, y: {y:.2f}); angle: {angle}")
                tracker.keypoint_locations.pop(index)
                json.dump(tracker.keypoint_locations, open(tracker.fp, "w"))
            except Exception:
                print("Invalid deletion position")
