import operator

from deap import gp

# Import modules from different directory
import os
import sys
PATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PATH)

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

# Test individual at different gravities and takes the average fitness
# of 5 and writes it to an excel sheet
def testGravity(ind, path):
    gravity = [1, 2, 3, 4, 5, 6, 7, 8, 9.81, 11, 12, 13, 14, 15, 16, 17]
    for i in gravity:
        add_to_excel = []
        total = 0
        add_to_excel.append(i)

        for j in range(5):
            fit = fullObsEvalIndividual(ind, pset, i, test=False)[0]
            total += fit

        add_to_excel.append(round(total/5, 2))
        write_to_excel(add_to_excel, path)

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
testGravity(ind, path=os.path.dirname(os.path.abspath(__file__)) + "/full_obs_grav.xlsx")
