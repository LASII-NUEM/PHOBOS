# Loops are control structures that run until a condition is met. The two loop structures available in Python are: "for" and "while" loops.

# 0. "for" loops
# The for loop iterates over the elements of the "condition", executing the code block inside it once for each element.
print('Basic for loop:')
for i in range(0,5):
    print(f'[for loop] {i}')

print()

# For loops can also be nested (one loop inside another).
# However, tread lightly when nesting too many loops, because for each element of the outer loop, the nested loop runs all its elements.
# Meaning, the task grows exponentially with the addition of another loop (curse of dimensionality)!
print('Nested for loop:')
for i in range(0,5):
    print(f'[outer loop] i = {i}')
    for j in range(0,5):
        print(f'[inner loop] j = {j}')

print()

# Lists can be iterated over their elements and also over their indices!
random_list = [0,1,2,3,4]

#iterating over the indices
print(f'Iterating {random_list} over the indices:')
for i in range(len(random_list)): #by default range starts at 0
    print(random_list[i])

print()

#iterating over the elements
print(f'Iterating {random_list} over the elements:')
for elem in random_list:
    print(elem)

print()

# 1. "while" loops
# The while loop runs indefinitely until the condition verified inside the loop is met.
# Don't forget to validate the condition properly, or your loop could run forever!
print(f'Simple while loop')
i = 0
while i<10:
    i += 1
print(f'exited while loop with i = {i}')
print()

#Uncomment to run the infinite loop:
# i = 0
# while True:
#     if i > 0:
#         break
#
#     print('running...')

# 2. "break" and "continue" statements
# Once a condition is met to stop a loop, "break" can be used. In cases where a condition is used to skip the current iteration, use "continue".
print(f'break statement')
i = 0
while i<10:
    i += 1
    if i == 4:
        break
    print(f'i = {i}')

print()

print(f'continue statement')
i = 0
while i<10:
    i += 1
    if i == 4:
        continue
    print(f'i = {i}')

print()
