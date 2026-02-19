# First, let's learn some basics of Python programming

# 0. Variables
# A way to store values computed or assigned in your script.

var1 = 1 #assign a value to variable
var2 = max([1,2,3,4,5]) #compute a value and assign to a variable

# 1. Data types
# Any valid "value" in Python belongs to a data type.
# These types can span from simple numbers (integers, floats, complex values), character sequences (strings), and logic values (boolean) to more complex structures like classes.

x = 420 #integer
print(f'type(x) = {type(x)}')

x1 = 420.5 #float
print(f'type(x1) = {type(x1)}')

x2 = 420.-1j
print(f'type(x2) = {type(x2)}')
#OBS: complex values store real and imaginary parts as attributes
x2_real = x2.real
x2_imag = x2.imag

x3 = True #boolean
print(f'type(x3) = {type(x3)}')

x4 = 'Igor'
print(f'type(x4) = {type(x4)}')

# 1.1 Type casting
# You can "convert" a value from a type to another, this is called "casting"
x = float(x)
print(f'type(x) w/ float casting = {type(x)}')

x1 = int(x1) #OBS: integer casting performs a floor operation
print(f'type(x1) w/ integer casting  = {type(x1)}')

x3 = int(x3)
print(f'type(x3) w/ integer casting  = {type(x3)}')

# 2. Operators
# Operators are straightforward and work as one would expect:

# - sum: +
# - subtraction: -
# - division: /
# - multiplication: *
# - power of n: **n

#However, Python has a feature that allows operations over pre-assigned variables. It's not a unique feature; for instance, C/C++ also has this!
var3 = 42 #assign a value to var3
var3 *= 10 #multiply the last assigned value by 10
print(f'var3*=10 = {var3}')

# This can be done with all basic operations:
# - sum: +=
# - subtract: -=
# - division: /=
# - multiplication: *=
# - power: **=

#Heads up: all division operations output a float value!
var4 = 1
var5 = 420
print(f'[int() / int()] a division of two integers results in: {type(var5/var4)}')

#Other special operators include:
# - modulo: %
# - integer division: //
