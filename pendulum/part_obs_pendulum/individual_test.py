import math
import operator

from deap import gp

# Import modules from different directory
import sys
sys.path.append('/Users/sky/Documents/Work Info/Research Assistant/deap_experiments/pendulum')


from modules.prim_functions import *
from modules.output_functions import *
from modules.part_obs_eval_individual import partObsEvalIndividual

# Set up primitives and terminals
pset = gp.PrimitiveSet("MAIN", 6)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(conditional, 2)
pset.addPrimitive(ang_vel, 4)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(asin, 2)
pset.addPrimitive(acos, 2)
pset.addPrimitive(atan, 2)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.tan, 1)
pset.addPrimitive(max, 2)
pset.addPrimitive(limit, 3)

pset.renameArguments(ARG0='y1')
pset.renameArguments(ARG1='x1')
pset.renameArguments(ARG2='y2')
pset.renameArguments(ARG3='x2')
pset.renameArguments(ARG4='y3')
pset.renameArguments(ARG5='x3')

def testGravity(s):
    gravity = [1, 2, 3, 4, 5, 6, 7, 8, 9.81, 11, 12, 13, 14, 15, 16, 17]
    for i in gravity:
        add_to_excel = []
        total = 0
        add_to_excel.append(i)

        for j in range(5):
            fit = partObsEvalIndividual(s, pset, i, test=False)[0]
            total += fit

        add_to_excel.append(round(total/5, 2))
        write_to_excel(add_to_excel, path="/Users/sky/Documents/pendulum_gravity.xlsx")

str = 'ang_vel(limit(asin(protectedDiv(y3, y2), acos(y3, x2)), conditional(x1, conditional(y3, x3)), tan(y3)), cos(sin(x1)), cos(x2), x2)'
s = gp.PrimitiveTree.from_string(str, pset)
# nodes, edges, labels = gp.graph(s)
# print(partObsEvalIndividual(s, pset, 9.81, True))

# plot_as_tree(nodes, edges, labels, 12)
# testGravity(s)