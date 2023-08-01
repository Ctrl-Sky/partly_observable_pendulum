# genetic programming for Gymnasium Pendulum task
# https://www.gymlibrary.dev/environments/classic_control/pendulum/ 

import operator

from deap import gp

# Import modules from different directory
import os
import sys
PATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PATH)

from modules.prim_functions import *
from modules.output_functions import *
from modules.test_gravity import fullObsTestGravity

# Set up primitives and terminals
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(conditional, 2)

pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')
pset.renameArguments(ARG2='vel')

# Replace value of str to an individuals tree in string form to test it
# Can simply print the indivudual to output the ind's tree in string form
# in string form and just copy and paste it here
str='conditional(add(x, y), add(vel, y))'
ind=gp.PrimitiveTree.from_string(str, pset)

# Creates an env and displays the individual being tested and
# then prints out it's fitness score
# print(fullObsEvalIndividual(ind, pset, 9.81, True))

# Plots the graph of the ind in a more falttering way and
# saves it to a png to view
# nodes, edges, labels = gp.graph(s)
# plot_as_tree(nodes, edges, labels, 12)

# Test the ind at different gravity values and then
# writes the fitness score at each gravity to full_obs_grav.xlsx
fullObsTestGravity(ind, pset, path=os.path.dirname(os.path.abspath(__file__)) + "/full_obs_grav.xlsx")
