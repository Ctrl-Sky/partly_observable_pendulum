import math

def conditional(input1, input2):
    if input1 < input2:
        return -input1
    else: return input1

def limit(input, minimum, maximum):
    if input < minimum:
        return minimum
    elif input > maximum:
        return maximum
    else:
        return input
    
def protectedDiv(left, right):
    try: return truncate(left, 8) / truncate(right, 8)
    except ZeroDivisionError: return 1

def truncate(number, decimals=0):
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def vel(y2, y1, x2, x1):
    try: y = truncate((y2-y1), 8)
    except: 
        y = 1
    try: x = truncate((x2-x1), 8)
    except:
        x = 1
    v = protectedDiv(y, x)
    return v

def ang_vel(y2, y1, x2, x1):
    top = atan(y2, x2) - atan(y1, x1)
    bottom = y2 - y1
    return protectedDiv(top, bottom)

def acos(x, y):
    if protectedDiv(x, y) < 1 and protectedDiv(x, y) > -1:
        return math.acos(x/y)
    elif protectedDiv(y, x) < 1 and protectedDiv(y, x) > -1:
        return math.acos(y/x)
    else:
        return x

def asin(x, y):
    if protectedDiv(y, x) < 1 and protectedDiv(y, x) > -1:
        return math.asin(y/x)
    elif protectedDiv(y, x) < 1 and protectedDiv(y, x) > -1:
        return math.asin(y/x)
    else:
        return x

def atan(x, y):
    return math.atan(protectedDiv(x, y))

def vel(y2, y1, x2, x1):
    top = y2-y1
    bottom = x2-x1
    return protectedDiv(top, bottom)