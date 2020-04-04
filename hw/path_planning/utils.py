import cv2
import math
import numpy as np
import pdb
import time
NEIGHBOR_RADIUS=100
CAR_BUFFER=25



# For our algorithm to work, we need some image of the external environment that makes it easy to check whether any path hits an obstacle. This is easier to do on a picture of the environment than in the actual environment itself. Thus, we need to constantly convert coordinates from the actual coordinates (that the car measures its location in) to the pixel on the image that the car is centered at. The above methods allow us to do that.




def mapToImage(point):
	new_point=Node()
	new_point.x=int((672-123)/12 *point.x+123)
	new_point.y=int((529-72)/10 *point.y+72)
	return new_point
def imageToMap(point):
	new_point=Node()
	new_point.x=12*(point.x-123)/(672-123)
	new_point.y=10*(point.y-72)/(529-72)


# The nearest function returns all the nodes that are within a certain radius of a node, and are connectable by a straight line to the node.




def nearest(nodes,node,radius,img):
	return [x for x in nodes if x.distance(node)<radius and checkLine(img,x,node)]


# drawLine connects two different points on an image using Bresenham's line algorithm. The implementation given here also draws the line with a width of "radius" pixels. 




def drawLine(img,point1,point2,radius=5):
	deltax=point2.x-point1.x
	deltay=point2.y-point1.y
	x=point1.x
	if deltax==0:
		x_increment=0
	else:
		x_increment=deltax//abs(deltax)
	error=0
	y=point1.y
	if deltay==0:
		y_increment=0
	else:
		y_increment=deltay//abs(deltay)
	
	if abs(deltax)>=abs(deltay):
		slope=deltay/deltax
		
		while x!=point2.x+x_increment:
			
			for y_temp in range(y-radius,y+radius):
				if y_temp>=0 and y_temp<img.shape[0]:
					img[y_temp][x]=255
			error+=abs(slope)
			if error>=0.5:
				error-=1
				y+=y_increment
			x+=x_increment
	else:
		slope=deltax/deltay
		
		while y!=point2.y+y_increment:
			for x_temp in range(x-radius,x+radius):

					img[y][x_temp]=255
			error+=abs(slope)
			if error>=0.5:
				error-=1
				x+=x_increment
			y+=y_increment


# The checkLine function also uses the Bresenham line algorithm, but this time it is being used to check whether the line between the two points is free fo any obstacles. 




def checkLine(img,point1,point2,radius=CAR_BUFFER):

	deltax=point2.x-point1.x
	deltay=point2.y-point1.y
	if deltax==0 and deltay==0:
		return True
	x=point1.x
	if deltax==0:
		x_increment=0
	else:
		x_increment=deltax//abs(deltax)
	error=0
	y=point1.y
	if deltay==0:
		y_increment=0
	else:
		y_increment=deltay//abs(deltay)
	if abs(deltax)>=abs(deltay):

		slope=deltay/deltax
	
		while x!=point2.x+x_increment:

			for y_temp in range(y-radius,y+radius):
				if y_temp>=0 and y_temp<img.shape[0] and img[y_temp][x]>200:
					return False
			error+=abs(slope)
			if error>=0.5:
				error-=1
				y+=y_increment
			x+=x_increment
	else:
		slope=deltax/deltay
	
		while y!=point2.y+y_increment:
			for x_temp in range(x-radius,x+radius):
				if x_temp>=0 and x_temp <img.shape[1] and img[y][x_temp]>200:
					return False
			error+=abs(slope)
			if error>=0.5:
				error-=1
				x+=x_increment
			y+=y_increment
	return True


# The following method is used at the start and end to check whether there is a clear enough circle for the starting and ending point to be valid (e.g. if the point is not itself invalid).




def checkCircle(img,point,radius=CAR_BUFFER):
	for x_diff in range(-radius,radius+1):
		for y_diff in range(-(radius-abs(x_diff)),radius-abs(x_diff)+1):
			if point.y+y_diff>=0 and point.y+y_diff<img.shape[0] and point.x+x_diff>=0 and point.x+x_diff<img.shape[1]:
				if img[point.y+y_diff][point.x+x_diff]>200:
					return False
	return True


# The node class we define here represents a single node in our expanding tree. The parent field will be the "parent" of the node in the tree. 
#Note that in this case, we never actually have to keep track of the children of a node. Cost is the distance of the shortest path from the root node to this node. 




class Node:
	def __init__(self,x=0,y=0,parent=None,mode='image'):
		if mode=='image':
			self.x=int(x)
			self.y=int(y)
		else:
			self.x=x
			self.y=y
		self.parent=parent
		self.cost=0
	def distance(self,other):
		return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2)
	def __str__(self):
		return "{} {}".format(self.x,self.y)


# The edges variable is a numpy array that reads the boundary_map picture in grayscale, so each entry is an integer between 0 and 255 inclusive.
# 










