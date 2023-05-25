import random
from deap import gp

# Create two example individuals
ind1 = gp.PrimitiveTree.from_string("add(sub(a, b), mul(c, d))")
ind2 = gp.PrimitiveTree.from_string("div(add(a, b), sub(c, d))")

# Perform one-point crossover
offspring1, offspring2 = gp.cxOnePoint(ind1, ind2)

# Display the resulting offspring
print(offspring1)
print(offspring2)