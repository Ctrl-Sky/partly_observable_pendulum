import math
def hello():
    sqerrors = (x**2 for x in range(5))
    return math.fsum(sqerrors),

print(hello())