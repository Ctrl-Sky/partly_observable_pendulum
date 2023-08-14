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
pset.addPrimitive(memProtectedDiv, [float, float], float)
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

def indexMemTestGravity(inds, pset, trained_grav, path_to_excel):
    gravity = [1, 2, 3, 4, 5, 6, 7, 8, 9.81, 11, 12, 13, 14, 15, 16, 17]

    for i in inds:
        add_to_excel = [i]
        ind = gp.PrimitiveTree.from_string(i, pset)

        for j in gravity:
            total = 0

            for k in range(10):
                fit = indexMemEvalIndividual(ind, pset, j, test=False)[0]
                total += fit
            
            add_to_excel.append(round(total/10, 2))
        write_to_excel(add_to_excel, trained_grav, path_to_excel)

path_to_read=os.path.dirname(os.path.abspath(__file__))+'/memory_raw_data.xlsx'
path_to_write=os.path.dirname(os.path.abspath(__file__))+'/memory_grav.xlsx'
GRAV='13'
inds = get_one_column(path_to_read, GRAV, 'A')
indexMemTestGravity(inds, pset, GRAV, path_to_write)



# Replace value of str to an individuals tree in string form to test it
# Can simply print the indivudual to output the ind's tree in string form
# in string form and just copy and paste it here
# inds=['sub(sub(read(a0, sin(conditional(0, sin(a1)))), read(a0, write(a0, protectedLog(conditional(sub(a1, 0), 0)), write(a0, a2, protectedDiv(protectedLog(abs(protectedLog(cos(protectedLog(abs(a1)))))), a2))))), protectedDiv(a1, a2))']

# Creates an env and displays the individual being tested and
# then prints out it's fitness score
# for i in inds:
#     ind=gp.PrimitiveTree.from_string(i, pset)
#     print(indexMemEvalIndividual(ind, pset, 9.81, True))

# Plots the graph of the ind in a more falttering way and
# saves it to a png to view
# nodes, edges, labels = gp.graph(ind)
# plot_as_tree(nodes, edges, labels, 12)

# Test the ind at different gravity values and then
# writes the fitness score at each gravity to part_obs_grav.xlsx
# indexMemTestGravity(inds, pset, '9.81', path_to_write)

