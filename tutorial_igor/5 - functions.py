import math

# 0. Python packages
# Everytime we import a package, we also import its methods (functions). To call a function from a package simply:
# <package_name>.<function_name>(arguments)

# 1. Functions
# Another *very* important feature of programming in general (this is not exclusive to Python!) is FUNCTIONS.
# Defined using the keyword "def", followed by the function name, the function parameters inside the parentheses, and a colon
# They allow you to compute the same operation multiple times, subject to changes in the input arguments.
# In practice, this means only one line of code each time you need to repeat the computation.
# For example, if we wanted to create a function that computes the distance between to 2D-points (tuples), it would be something like:

def euclid_distance(p1:tuple, p2:tuple):
    '''
    :param p1: point 1 (tuple by default)
    :param p2: point 2 (tuple by default)
    :return: the Euclidean distance between point 1 and 2
    '''

    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

x1 = (2,-2)
x2 = (-2,6)
dist = euclid_distance(x1, x2)
print(f'dist(x1,x2) = {dist}')

#compare with the built-in distance function of the math package
dist_math = math.dist(x1, x2)
print(f'math.dist(x1,x2) = {dist}')

# Note: variables inside a function usually does not exist outside the function
# (exceptions if they are shadowed or the function is a method of a class, more on classes later!)
# i.e., the Rosenbrock function is a very famous function in numerical optimization
# it is known as a benchmark function, given the difficulty to converge at its global minimum (1,1)

def rosenbrock(x):
    first_half = 100*(x[1]-x[0]**2)**2
    print(f'[inside the function] first_half = {first_half}')
    second_half = (1-x[0])**2
    print(f'[inside the function] second_half = {second_half}')
    return first_half + second_half

rosen_min = (-1.2,1)
cost = rosenbrock(rosen_min)

# 2. Try and except
# Try to print the values computed in first_half and second_half outside the function
try:
    print(f'[outside the function] first_half = {first_half}')
    print(f'[outside the function] second_half = {second_half}')
except:
    print(f'[outside the function] first_half and second_half do not exist outside rosenbrock()')

# 3. Default arguments
# Functions can have default arguments when defined.
# If the user does not provide them when calling the function, it defaults to the value provided in the definition.
def power_of_n(x, n=2):
    return x**n

x3 = 2
p_default = power_of_n(x3)
print(f'[power_of_n] default n = {p_default}')
p_argument = power_of_n(x3, n=3)
print(f'[power_of_n] "n=3" = {p_argument}')
print()

#NOTE: functions in Python can return one or multiple values, but also nothing!
#i.e., a function that prints all the characters in a name isn't required to return the input string
print('Function with no return value:')
def print_char(name):
    for i in name:
        print(i)
    print()

print_char('igor')

