#!/usr/bin/env python
# coding: utf-8

import cv2
import math
import numpy as np
import pdb
import time
from utils import *
np.random.seed(0)
#STEPS is the amount of iterations the algorithm will run for 
#given that it has found a solution by the time that many iterations pass.
#Try increasing this to see what happens, if you have time.
STEPS = 4000




# The find_path method takes in 5 parameters:
# * `start` point
# * `target` point
# * `edges`, an image which has the boundaries of the track
# * `point_type` of both of these points (either 'map' or 'image')
# * a `radius` of safety (it must maintain that pixel distance from walls)
def find_path(start, target, edges, radius=CAR_BUFFER, point_type='map'):
    start.cost = 0
    #We need to do this process on our image, so we convert the points to
    #image-coordinates if they aren't already in it
    if point_type == 'map':
        start, target = mapToImage(start), mapToImage(target)
    if not (checkCircle(edges, start) and checkCircle(edges, target)):
        print("invalid endpoints")
        sys.exit(0)
    limit = 50

    end = None
    nodes = [start]
    counter = 0
    while not end or counter < STEPS:
        #TODO: we need to update all children of a node if the node's cost changes
        #print status updates every 10 stpes
        if counter % 100 == 0:
            print("Iteration: {} Nodes: {} ".format(counter, len(nodes)))
        counter += 1
        #create a random point within the tree (should check if it is already in the tree)
        random_point = Node(np.random.randint(0, edges.shape[1]),
                            np.random.randint(0, edges.shape[0]))

        ######################################
        #   FIRST THING YOU HAVE TO FILL-IN   #
        ######################################
        # In the list of nodes, find the node which is closest to random_point,
        #and find the distance from this closest point to the random_point
        #Put these in minNode and dist, respectively
        minNode=None #FIXME
        dist=None #FIXME

        #if we can't connect the random_point and the node closest to it in our tree,
        #then we assume that the random_point is out of our search space and then exit
        if not checkLine(edges, random_point, minNode):
            continue

        #if the closest node to the random point is still too far away, we scale the random point so that
        #it is closer to the minimum node
        if dist > limit:
            weight = dist / limit - 1
            total = weight + 1
            random_point = Node(
                int((random_point.x + weight * minNode.x) / total),
                int((random_point.y + weight * minNode.y) / total))

        #the random point's parent will be the node closest to it (for now) and its cost will
        #be the length of the path from the closest node to the random point,
        #added to the length of the shortest path to the closest node
        random_point.parent = minNode
        random_point.cost = dist + minNode.cost
        #we now see all the points closest to this random point that we have just added to the tree
        X_n = nearest(nodes, random_point, NEIGHBOR_RADIUS, edges)

        ######################################
        #   SECOND THING YOU HAVE TO FILL-IN   #
        ######################################
        #You need to check, for each of the nodes in X_n, whether that node would
        #be a better parent for random_point than the node closest to it.
        #A node N is a better parent for the random_point if random_point's cost,
        #if N was its parent, is lower than its current cost.
        #If N is a better parent for the random_point, then we update random_point's parent and cost

        for node in X_n:
            pass #FIXME


        ######################################
        #   THIRD THING YOU HAVE TO FILL-IN   #
        ######################################
        #You need to check, for each of the nodes in X_n, whether random_point
        # is a better parent for each node than that node's current parent. 
        #If random_point is, update the node's parent and cost accordingly

        for node in X_n:
            pass #FIXME

        #finally, we add the random_point to the list of nodes in our search space
        nodes.append(random_point)
        #if we can get directly to the target from the random_point we're at, and
        #our cost is lower than the current cost of the target, then we update
        #the target's cost and parent accordingly
        if checkLine(edges, random_point,
                     target) and (not end or random_point.cost +
                                  random_point.distance(target) < target.cost):
            end = target
            target.parent = random_point
            target.cost = random_point.cost + random_point.distance(target)
    #We follow the path of parents back from the end in order to draw the final path to target.
    while end.parent:
        drawLine(edges, end, end.parent, radius=1)
        end = end.parent
    print(len(nodes))
    cv2.imshow("Final Frame", edges)
    cv2.waitKey(0)
    cv2.imwrite("last_trajectory_rrtstar.png", edges)


# Finally, if you've filled out everything else correctly above,
#when you run the file, you should be able to find a working path between
#the top left and bottom right corner in the board:


def main():
    edges = cv2.imread('boundary_map.png', 0)
    start = Node(528, 468)
    start.cost = 0
    target = Node(184, 120)
    find_path(start, target, edges, point_type="image")


if __name__ == "__main__":
    main()
