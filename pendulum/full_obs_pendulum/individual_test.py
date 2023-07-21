import math
import operator

from deap import gp

# Import modules from different directory
import sys
# sys.path.append('/deap_experiments/pendulum')

from modules.prim_functions import *
from modules.output_functions import *
from modules.full_obs_eval_individual import fullObsEvalIndividual

# Set up primitives and terminals
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(conditional, 2)

pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')
pset.renameArguments(ARG2='vel')

def testGravity(s):
    gravity = [1, 2, 3, 4, 5, 6, 7, 8, 9.81, 11, 12, 13, 14, 15, 16, 17]
    for i in gravity:
        add_to_excel = []
        total = 0
        add_to_excel.append(i)

        for j in range(5):
            fit = fullObsEvalIndividual(s, pset, i, test=False)[0]
            total += fit

        add_to_excel.append(round(total/5, 2))
        write_to_excel(add_to_excel, path="/Users/sky/Documents/pendulum_gravity.xlsx")

str = 'conditional(add(x, y), add(vel, y))'
s = gp.PrimitiveTree.from_string(str, pset)
# nodes, edges, labels = gp.graph(s)
print(fullObsEvalIndividual(s, pset, 9.81, True))

# plot_as_tree(nodes, edges, labels, 12)
# testGravity(s)