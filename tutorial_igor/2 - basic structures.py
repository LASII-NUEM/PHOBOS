import numpy as np

# Following the last scripts on data types. Let's learn about some useful data structures that will help you in scientific computing tasks!

# 0. Indexing
# Python uses 0-based indexing (like most popular programming languages: C, C++, Java,...) with square bracketing ("[]").
# This means that the n-th element of an array is accessed using the index "[n]",
# i.e., given the time array "t" with length N, the first element is t[0] and the last is t[N-1] (or t[-1]). Both should match the input variables "t_init" and "t_end".
# However, if you work with MATLAB, indexing is 1-based and uses parentheses. Hence, for the same "t" array, we would have t(1) for the first element, and t(N) for the last.
# Lastly, indexable components can also be sliced, indexed with a fixed step, masked, ...

# 1. Strings
# Sometimes, you may need to process standard ASCII code data (i.e., letters).
# These 'letters' are usually known as characters (char).
# However, in Python, all letters (and sequences of them) belong to the same class: Strings. Think of strings as a vector of characters (which is the usual interpretation for compiled languages such as C or C++).
# Since strings are a sequence of characters, one can extract individual characters from the string with indexing:

str1 = "Hello world"
print(f'First character: {str1[0]}')
print(f'Last character: {str1[-1]}')
#NOTE: you can also extract the last element based on the length of the string
print(f'Last character with len(str): {str1[len(str1)-1]}')
print(f'Sliced string: {str1[:5]}')
print(f'Indexed with a step: {str1[::2]}')

# 2. Lists
# Lists are also a sequence, but the elements can be of any type
l_int = [1,2,3,4,5]
print(f'type(l_int): {type(l_int)}')
#OBS: lists can store elements of multiple types as well
l_mult = [1,'2',3.,True,1+1j]

#The same slicing techniques can be used to manipulate strings
print(f'First element: {l_mult[0]}')
print(f'Last character: {l_mult[-1]}')
print(f'Sliced list: {l_mult[:3]}')
print(f'Indexed with a step: {l_mult[::2]}')

# 3. Tuples
# Behave like lists, but once created they cannot be modified. This behavior is knows as immutability!
tuple1 = (1,2)
print(f'type(tuple1): {type(tuple1)}')

# The values of the tuple can be unpacked by a comma-separated list of variables
t1_a, t1_b = tuple1
print(f'First element: {t1_a}')
print(f'Second element: {t1_b}')

# 4. Dictionary
# Also known as hash tables, these structures map a key to a value (or multiple values).
dict1 = {"key1": 1,
         "key2": 2,
         "key3": 3}

print(f'dict["key1"] = {dict1["key1"]}')
print(f'dict["key2"] = {dict1["key2"]}')
print(f'dict["key3"] = {dict1["key3"]}')

# A value from a key can be updated
dict1["key1"] = 6
print(f'dict["key1"] = {dict1["key1"]}')

# A new entry can be added by assigning to a new key
dict1["new_key"] = 4
print(f'dict["new_key"] = {dict1["new_key"]}')

