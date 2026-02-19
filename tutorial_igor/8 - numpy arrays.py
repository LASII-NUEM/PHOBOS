# Numpy (NUMerical PYthon) contains multidimensional arrays (vectors, matrices, and tensors) and multiple functions to efficiently operate on them!
# In synthesis, arrays are like lists, but with greater dimensions and MUCH more efficient.
# learn more about the library at: https://numpy.org/doc/stable/user/absolute_beginners.html

import numpy as np

# 0. Array creation
A = np.array([[1,2], [3,4]])
print(f'A = {A}')
print(f'shape(A) = {np.shape(A)}') #or A.shape also works
print(f'A[0,1] = {A[0,1]}') #indexing
A[0,0] = 9 #array are mutable (just like lists)!
print(f'A = {A}')
print()
# By convention, the first index of a matrix represents the lines, and the second the columns.
# For portuguese speakers, remember: A[i,j] -> i stands for 'inha' (linha), j stands for 'joluna' (coluna) :P

B = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(f'B = {B}')
print(f'shape(B) = {np.shape(B)}') #or A.shape also works
print(f'B[1,1] = {B[1,1]}') #indexing

# Numpy also provides many functions to create different arrays
C = np.zeros(shape=(2,2)) #a 2-by-2 matrix filled with zeros
print(f'C = {C}')
print()

D = np.ones((1,2)) #an array filled with ones
print(f'D = {D}')
print()

E = np.eye(5) #a 5-by-5 identity matrix
print(f'E = {E}')
print()

# 1. Element wise operations
# One of the major advantages of numpy arrays is that arithmetic operations correspond to element wise operations.
# Meaning, when computing matrix operations, it is not necessarily required to iterate over the elements
B_eye = np.eye(3) # a 3-by-3 identity with the size of B
hadamard_prod_B = B*B_eye #compute the hadamard product (element-wise multiplication) between B and B_I
print(f'B ⊙ B_I = {hadamard_prod_B}')
print()
# Note that '*' in numpy does represent matrix multiplication like in MATLAB. Matrix multiplication uses both "@" and "np.dot" operators!

matmul_B = B@B_eye #matrix multiplication between B and the 3x3 identity
print(f'B @ B_I = {matmul_B}')
print()

dot_B = np.dot(B, B_eye) #matrix multiplication between B and the 3x3 identity
print(f'dot(B,B_I) = {matmul_B}')
print()

D2 = np.array([5,10]) #an 1-by-2 array
ew_sum_D = D+D2
print(f'D + D2 = {ew_sum_D}')
print()

#All the available operation with numpy arrays are listed in the documentation: https://numpy.org/doc/stable/reference/routines.math.html

# 2. Slicing
# Numpy uses pass-by-reference semantics so it creates views into the existing array, without implicit copying.
# This is particularly helpful with very large arrays because copying can be slow.
#i.e.,
F = np.array([1,2,3,4,5,6])
print(f'F = {F}')
print()

G = F[0:4] #sliced up to the 4th element
print(f'G = {G}')
print()

#If we change an element in G:
G[0] = 15
#F is also affected, because of the pass-by-reference
print(f'F = {F}')
print(f'G = {G}')
print()

#To only change G, copy the whole array F into a new variable
G_copy = np.copy(F)
G_copy[2] = 999
print(f'F = {F}')
print(f'G_copy = {G_copy}')
print()

# 3. Indexing
# Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array:
H = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# subarray consisting of the first 2 rows and columns 0 and 1; b is the following array of shape (2, 2)
print(f'H[:2, 1:3] = {H[:2, 1:3]}')
print()

# Boolean array indexing lets you pick out arbitrary elements of an array.
# Frequently this type of indexing is used fo thresholding.
# This is also known as 'masking' an array.
fs = 1e3
freqs = np.linspace(-fs/2, fs/2, 1000) #create 1000 evenly-spaced elements from -fs/2 up to fs/2
print(f'Array before masking: {freqs}')
print()
freqs_mask = freqs>0 #let's say we want only the positive values
freqs = freqs[freqs_mask]
print(f'Array after masking: {freqs}')
print()

# 4. Broadcasting
# i.e., suppose that we want to add a constant vector to each row of a matrix.
I = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]]) # 4-by-3 matrix
J = np.array([1, 0, 1]) #1-by-3 array
K = np.zeros_like(I) #4-by-3 matrix

# the naive solution: for loop
for i in range(len(K)):
    K[i,:] = I[i,:] + J
print(f'K = {K}')

# this works. However, when the matrix I is very large, computing an explicit loop in Python could be slow.
# adding the vector J to each row of the matrix I is equivalent to forming a matrix J' by stacking multiple copies of J vertically
# a more efficient way of broadcasting J into I would be:
J_tile = np.tile(J, (4,1))
K_tile = I + J_tile
print(f"K' = {K_tile}")
# other broadcasting operations include: np.reshape(), .T -> transpose, and etc...