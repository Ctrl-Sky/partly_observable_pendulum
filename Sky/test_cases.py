import fractions

# user defined funcitons
def conditional(input1, input2):
    if input1 < input2:
        return -input1
    else: return input1

def if_(input, output1, output2):
    if input: return output1
    else: return output2

def limit(input, minimum, maximum):
    if input < minimum:
        return minimum
    elif input > maximum:
        return maximum
    else:
        return input
    
def protectedDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1

def h(h):
    if h < 0:
        return 0
    elif h == 0:
        return 0
    else: return 2

vel = [-0.07, -0.03, 0.00, 0.03, 0.07]
arr = []

i = fractions.Fraction(-7, 100)
while i < fractions.Fraction(7, 100):
    print(i)
    a = limit(i, 0, 1)
    b = if_(a, 1, -1)   
    if b < 0: b =0
    else: b = 2
    arr.append(b)
    i += fractions.Fraction(1, 100)

print(arr)

arr2 = []
i = fractions.Fraction(-7, 100)
while i < fractions.Fraction(7, 100):
    a = h(i)
    arr2.append(a)
    i += fractions.Fraction(1, 100)

print(arr2)
