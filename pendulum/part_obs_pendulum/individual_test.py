# genetic programming for Gymnasium Pendulum task
# https://www.gymlibrary.dev/environments/classic_control/pendulum/ 

import math
import operator

from deap import gp

# Import modules from different directory
import os
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from modules.prim_functions import *
from modules.output_functions import *
from modules.eval_individual import partObsEvalIndividual

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

def partObsTestGravity(ind, pset, string, path):
    gravity = [1, 2, 3, 4, 5, 6, 7, 8, 9.81, 11, 12, 13, 14, 15, 16, 17]
    add_to_excel = [string]
    for i in gravity:
        total = 0

        for j in range(3):
            fit = partObsEvalIndividual(ind, pset, i, test=False)[0]
            total += fit
        
        add_to_excel.append(round(total/3, 2))
    write_to_excel(add_to_excel, path)

# Replace value of str to an individuals tree in string form to test it
# Can simply print the indivudual to output the ind's tree in string form
# in string form and just copy and paste it here
l=['add(ang_vel(limit(max(x1, y1), add(y1, y2), conditional(y3, atan(y2, y3))), x3, y3, x3), ang_vel(atan(limit(y2, x1, x2), atan(y2, y3)), x3, ang_vel(acos(x2, x2), x3, x3, asin(sub(y1, y2), add(y3, y1))), x3))']

for i in l:
    string=i
    ind=gp.PrimitiveTree.from_string(string, pset)

    # Creates an env and displays the individual being tested and
    # then prints out it's fitness score
    # print(partObsEvalIndividual(ind, pset, 14, True))

    # Plots the graph of the ind in a more falttering way and
    # # saves it to a png to view
    nodes, edges, labels = gp.graph(ind)
    plot_as_tree(nodes, edges, labels, 13)

    # Test the ind at different gravity values and then
    # writes the fitness score at each gravity to part_obs_grav.xlsx
    # partObsTestGravity(ind, pset, string, path=os.path.dirname(os.path.abspath(__file__)) + "/excel_sheets/part_obs_grav.xlsx")

