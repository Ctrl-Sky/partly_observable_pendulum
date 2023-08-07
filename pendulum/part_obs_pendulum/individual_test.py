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

        for j in range(1):
            fit = partObsEvalIndividual(ind, pset, i, test=False)[0]
            total += fit
        
        add_to_excel.append(round(total/5, 2))
    write_to_excel(add_to_excel, path)

# Replace value of str to an individuals tree in string form to test it
# Can simply print the indivudual to output the ind's tree in string form
# in string form and just copy and paste it here
# l = [
#      'ang_vel(conditional(sub(ang_vel(x3, y2, x1, y2), y3), y3), conditional(y1, y1), max(x1, x1), sub(x3, x1))',
#      'ang_vel(protectedDiv(y3, y1), limit(ang_vel(y2, y1, ang_vel(add(y1, y3), protectedDiv(x3, y1), conditional(x3, x3), add(asin(acos(y1, y3), max(x2, x2)), x2)), x2), ang_vel(y2, y2, asin(x1, y3), x3), limit(x1, y2, y3)), acos(ang_vel(y1, x2, atan(x1, x3), x1), y2), acos(x2, x2))',
#      'ang_vel(atan(max(conditional(y3, ang_vel(protectedDiv(x3, y2), atan(y3, y2), protectedDiv(x1, y2), tan(x3))), cos(atan(y3, y1))), y2), ang_vel(x1, y2, x1, sin(sub(x2, sub(x2, y1)))), sub(x3, x2), sin(x2))',
#      'ang_vel(asin(y3, x3), cos(x1), acos(sin(x2), atan(x1, protectedDiv(acos(y3, y3), cos(x3)))), sub(asin(x2, cos(y1)), x1))',
#      'ang_vel(y2, sub(y3, x1), sub(y2, y3), acos(max(cos(conditional(x3, add(add(atan(y3, x2), y1), y2))), y1), y2))',
#      'protectedDiv(conditional(ang_vel(limit(y3, y2, protectedDiv(sin(x2), add(x1, y2))), protectedDiv(x1, y2), limit(y3, y2, y2), sin(x3)), tan(ang_vel(y1, x2, x3, y2))), ang_vel(x3, y3, x2, y2))',
#      'ang_vel(tan(x2), asin(y1, x1), sin(x3), max(y3, conditional(sub(y2, y3), conditional(x3, y2))))',
#      'add(sub(add(x3, x3), conditional(x3, asin(x3, x1))), limit(conditional(acos(cos(y1), x3), add(atan(max(cos(x3), atan(x1, x1)), conditional(x3, conditional(y1, y2))), y2)), asin(x3, x3), cos(y1)))',
#      'asin(protectedDiv(tan(tan(x2)), conditional(x2, x1)), acos(max(y3, acos(protectedDiv(y3, y2), sub(sin(x3), y2))), sub(x2, max(y2, x2))))',
#      'ang_vel(protectedDiv(add(y1, y2), y2), atan(y1, y3), ang_vel(protectedDiv(x2, y3), ang_vel(x1, y1, protectedDiv(x2, y3), x3), sin(x3), tan(y3)), sub(tan(x2), x3))',
#      'tan(ang_vel(sub(x2, x3), limit(y1, y2, conditional(y2, y3)), sin(x3), limit(limit(sin(limit(sin(y1), protectedDiv(y3, asin(x2, asin(asin(sub(x2, y1), x2), protectedDiv(y2, y1)))), tan(y2))), x1, conditional(x2, y2)), x2, y1)))',
#      'add(conditional(x2, tan(acos(sin(y3), acos(x3, y2)))), tan(protectedDiv(x3, protectedDiv(x3, limit(y1, acos(conditional(x2, tan(x1)), max(x1, x1)), y2)))))',
#      'sub(ang_vel(limit(protectedDiv(acos(y3, y2), y2), x2, x2), x1, x2, add(y3, ang_vel(y1, x3, limit(sin(y2), limit(x3, x3, x2), sub(x2, x1)), y3))), sub(acos(y3, y2), conditional(x1, x3)))',
#      'ang_vel(asin(x3, acos(y3, x2)), asin(cos(x1), tan(x1)), limit(asin(x2, max(sub(x3, x2), add(y3, y3))), max(x3, x2), x2), atan(protectedDiv(x1, y2), max(x1, ang_vel(limit(max(x1, x1), protectedDiv(x1, x3), limit(max(x1, x3), x3, y3)), sin(add(y2, y1)), x2, x2))))',
#      'sub(limit(x1, x3, x3), atan(ang_vel(conditional(x2, cos(asin(y3, y3))), x3, asin(y3, y1), conditional(y3, y1)), x3))',
#      'ang_vel(asin(acos(ang_vel(x1, x3, limit(asin(max(limit(x1, x3, y3), sub(atan(x2, x2), conditional(y3, y3))), y1), max(x2, y1), sin(y3)), x3), x1), conditional(x3, x3)), x1, sin(sin(x3)), y3)',
#      'add(ang_vel(limit(max(x1, y1), add(y1, y2), conditional(y3, atan(y2, y3))), x3, y3, x3), ang_vel(atan(limit(y2, x1, x2), atan(y2, y3)), x3, ang_vel(acos(x2, x2), x3, x3, asin(sub(y1, y2), add(y3, y1))), x3))',
#      'add(add(y3, x3), asin(x2, sin(x3)))',
#      'ang_vel(acos(sin(y1), asin(y3, y3)), conditional(limit(y3, x3, add(asin(x1, y1), add(y2, x3))), add(x1, y3)), asin(ang_vel(x1, x3, y1, x3), cos(y1)), atan(atan(sub(protectedDiv(x1, x3), add(x1, y2)), cos(y3)), acos(x2, y1)))',
#      'ang_vel(cos(limit(x1, limit(sin(protectedDiv(y1, x2)), conditional(x2, y2), limit(x1, x2, y2)), x1)), max(sub(y2, y2), protectedDiv(x1, x2)), limit(y1, x1, x1), add(x1, atan(y2, x2)))',
#      'tan(ang_vel(x1, conditional(limit(y2, conditional(limit(y2, y2, y2), sub(asin(x1, y2), y1)), protectedDiv(y1, x1)), sub(x1, y1)), tan(y1), x3))']


l = ['add(ang_vel(limit(max(x1, y1), add(y1, y2), conditional(y3, atan(y2, y3))), x3, y3, x3), ang_vel(atan(limit(y2, x1, x2), atan(y2, y3)), x3, ang_vel(acos(x2, x2), x3, x3, asin(sub(y1, y2), add(y3, y1))), x3))']
for i in l:
    string=i
    ind=gp.PrimitiveTree.from_string(string, pset)

    # Creates an env and displays the individual being tested and
    # then prints out it's fitness score
    # print(partObsEvalIndividual(ind, pset, 9.81, True))

    # Plots the graph of the ind in a more falttering way and
    # saves it to a png to view
    nodes, edges, labels = gp.graph(ind)
    plot_as_tree(nodes, edges, labels, 12)

    # Test the ind at different gravity values and then
    # writes the fitness score at each gravity to part_obs_grav.xlsx
    # partObsTestGravity(ind, pset, string, path=os.path.dirname(os.path.abspath(__file__)) + "/excel_sheets/part_obs_grav.xlsx")


