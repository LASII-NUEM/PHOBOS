import math

# Classes are data structures that belong to the realm of OBJECT-ORIENTED PROGRAMMING (OOP).
# A class represents an object (instance)and the operations that can be performed within the context of this object.
# Values of a class are known as 'attributes' and functions within the class are 'methods'.

# 0. class structure and constructor
class Point:
    #This is called a constructor, meaning it 'constructs' the Point object
    def __init__(self, x, y):
        self.x = x #x attribute
        self.y = y #y attribute
        self.x_rot = None #x with rotation
        self.y_rot = None #y with rotation

    #a method (function) of the Point class
    def rotate(self, theta):
        '''
        :param theta: rotation angle in radians
        '''
        self.x_rot = self.x*math.cos(theta)-self.y*math.sin(theta)
        self.y_rot = self.x*math.sin(theta)+self.y*math.cos(theta)
        # Note that self represents "this", the context of the class. Hence, x and y can be accessed from within the class.
        # Also, since we are updating an attribute of Point, rotate does not necessarily need to return a value!

point_obj = Point(1,-2) #create an object of the class 'Point'
print(f'attributes of Point before rotating 60°:')
print(f'x = {point_obj.x}')
print(f'y = {point_obj.y}')
print(f'x rotate = {point_obj.x_rot}')
print(f'y rotate = {point_obj.y_rot}')
print()

point_obj.rotate(math.radians(60)) #rotate 60 degrees
print(f'attributes of Point after rotation:')
print(f'x = {point_obj.x}')
print(f'y = {point_obj.y}')
print(f'x rotated = {point_obj.x_rot}')
print(f'y rotated = {point_obj.y_rot}')
