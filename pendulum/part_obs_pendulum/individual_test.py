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

def partObsTestGravity(inds, pset, trained_grav, path_to_excel):
    gravity = [1, 2, 3, 4, 5, 6, 7, 8, 9.81, 11, 12, 13, 14, 15, 16, 17]

    for i in inds:
        add_to_excel = [i]
        ind = gp.PrimitiveTree.from_string(i, pset)

        for j in gravity:
            total = 0

            for k in range(10):
                fit = partObsEvalIndividual(ind, pset, j, test=False)[0]
                total += fit
            
            add_to_excel.append(round(total/10, 2))
        write_to_excel(add_to_excel, trained_grav, path_to_excel)

path_to_read=os.path.dirname(os.path.abspath(__file__))+'/part_obs_raw_data.xlsx'
path_to_write=os.path.dirname(os.path.abspath(__file__))+'/part_obs_grav.xlsx'
GRAV='12'
inds = get_one_column(path_to_read, GRAV, 'A')
partObsTestGravity(inds, pset, GRAV, path_to_write)


# Replace value of str to an individuals tree in string form to test it
# Can simply print the indivudual to output the ind's tree in string form
# in string form and just copy and paste it here
# l=['protectedDiv(ang_vel(ang_vel(sin(add(x1, x3)), sin(y1), protectedDiv(y2, y2), acos(y3, y1)), y3, sub(sin(x2), sin(x1)), x2), acos(limit(add(y3, y1), tan(y3), limit(y3, limit(x1, conditional(max(limit(x1, y2, y2), asin(x2, x2)), protectedDiv(y3, y1)), y2), x3)), y3))']

# for i in l:
#     string=i
#     ind=gp.PrimitiveTree.from_string(string, pset)

    # Creates an env and displays the individual being tested and
    # then prints out it's fitness score
    # print(partObsEvalIndividual(ind, pset, 14, True))

    # Plots the graph of the ind in a more falttering way and
    # # saves it to a png to view
    # nodes, edges, labels = gp.graph(ind)
    # plot_as_tree(nodes, edges, labels, 12)

    # Test the ind at different gravity values and then
    # writes the fitness score at each gravity to part_obs_grav.xlsx
    # partObsTestGravity(ind, pset, string, path=os.path.dirname(os.path.abspath(__file__)) + "/excel_sheets/part_obs_grav.xlsx")

