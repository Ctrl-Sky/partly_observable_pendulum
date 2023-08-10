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
from modules.eval_individual import indexMemEvalIndividual

# Set up primitives and terminals
pset = gp.PrimitiveSetTyped("main", [list, float, float], float)

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(protectedLog, [float], float)
pset.addPrimitive(conditional, [float, float], float)
pset.addPrimitive(limit, [float, float, float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(operator.abs, [float], float)
pset.addTerminal(0, float)

pset.addPrimitive(read, [list, float], float)
pset.addPrimitive(write, [list, float, float], float)

pset.renameArguments(ARG0="a0")
pset.renameArguments(ARG1="a1")
pset.renameArguments(ARG2="a2")

def indexMemTestGravity(inds, pset, trained_grav, path):
    gravity = [1, 2, 3, 4, 5, 6, 7, 8, 9.81, 11, 12, 13, 14, 15, 16, 17]

    for i in inds:
        add_to_excel = [i]
        ind = gp.PrimitiveTree.from_string(i, pset)

        for j in gravity:
            total = 0

            for k in range(3):
                fit = indexMemEvalIndividual(ind, pset, j, test=False)[0]
                total += fit
            
            add_to_excel.append(round(total/3, 2))
        write_to_excel(add_to_excel, path, sheet_name=trained_grav)

path_to_read='/Users/sky/Documents/Work Info/Research Assistant/deap_experiments/pendulum/work_on_memory/memory_raw_data.xlsx'
path_to_write='/Users/sky/Documents/Work Info/Research Assistant/deap_experiments/pendulum/work_on_memory/memory_grav.xlsx'
GRAV='9.81'
# inds = get_one_column(path_to_read, 'A')
# indexMemTestGravity(inds, pset, GRAV, path_to_write)



# Replace value of str to an individuals tree in string form to test it
# Can simply print the indivudual to output the ind's tree in string form
# in string form and just copy and paste it here
# inds=['protectedDiv(ang_vel(ang_vel(sin(add(x1, x3)), sin(y1), protectedDiv(y2, y2), acos(y3, y1)), y3, sub(sin(x2), sin(x1)), x2), acos(limit(add(y3, y1), tan(y3), limit(y3, limit(x1, conditional(max(limit(x1, y2, y2), asin(x2, x2)), protectedDiv(y3, y1)), y2), x3)), y3))']

# Creates an env and displays the individual being tested and
# then prints out it's fitness score
# print(indexMemEvalIndividual(ind, pset, 14, True))

# Plots the graph of the ind in a more falttering way and
# saves it to a png to view
# nodes, edges, labels = gp.graph(ind)
# plot_as_tree(nodes, edges, labels, 12)

# Test the ind at different gravity values and then
# writes the fitness score at each gravity to part_obs_grav.xlsx
# indexMemEvalIndividual(ind, pset, string, path=os.path.dirname(os.path.abspath(__file__)) + "/excel_sheets/part_obs_grav.xlsx")

