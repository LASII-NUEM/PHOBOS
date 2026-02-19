# 0. Indentation
# When dealing with control flow structures, indentation ('tab' with respect to the closest structure) is required!
# If done incorrectly, it will raise errors from the interpreter!

#Try:
# if type(1) == int:
# print('teste')

# 1. Comparisons
# Control flow is basically the conditions you impose on your code that make it behave accordingly.
# What defines the conditions are known as "comparisons":
# - equal: ==
# - different: !=
# - logical 'and': and (bitwise 'and' is &)
# - logical 'or': or (bitwise 'or' is |)
# Booleans, by default, are the result of a comparison (True or False), which means that they can be defined as a condition without a comparison operator!

# Case 1: when the statements are booleans
statement1 = False
if statement1:
    print('Statement 1 is True')
else:
    print('Statement 1 is False')

statement2 = False
if not statement2:
    print('Statement 2 is False')
else:
    print('Statement 2 is True')

if not statement1 and not statement2:
    print(f'Statements 1 and 2 are False')
else:
    if not statement1:
        print(f'Statements 1 is False and 2 is True')
    else:
        print(f'Statements 1 is True and 2 is False')

# Case 2: when the statements are comparisons
def check_igor(your_name):
    if your_name == 'igor':
        print("name is igor")
    else:
        print("name is not igor")

name = "Igor"
check_igor(name)

# Note that string comparisons are also case-sensitive, in this case "Igor" is not the same as "igor" because I != i
# Quick tip: convert the name string to lower case with
name = name.lower()
check_igor(name)

