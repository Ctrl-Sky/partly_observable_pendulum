# genetic programming for Gymnasium Pendulum task
# https://www.gymlibrary.dev/environments/classic_control/pendulum/ 

import numpy
import random
import gymnasium as gym
import operator
import matplotlib.pyplot as plt
import math
from openpyxl import load_workbook
import sys
import inspect
from inspect import isclass
from operator import attrgetter

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import pygraphviz as pgv

# parallel
import multiprocessing

# Import modules from different directory
import os
import sys
PATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PATH)

from modules.prim_functions import conditional, truncate
from modules.output_functions import *
from modules.eval_individual import fullObsEvalIndividual

GRAV=17
POP=500
PROCESSES=96 #set to 16 if using compute canada
GENS=450
PATH_TO_WRITE='full_obs_raw_data.xlsx'

# Set up primitives and terminals
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(conditional, 2)

pset.renameArguments(ARG0='y')
pset.renameArguments(ARG1='x')
pset.renameArguments(ARG2='vel')

# Prepare individual and mountain car
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register functions in the toolbox needed for evolution
toolbox.register("evaluate", fullObsEvalIndividual, pset=pset, grav=GRAV)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    # Initialize the population
    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pool = multiprocessing.Pool(processes=PROCESSES) # parllel (Process Pool of 16 workers)
    toolbox.register("map", pool.map) # parallel

    pop, log = algorithms.eaSimple(pop, toolbox, 0.2, 0.5, GENS, stats=mstats, halloffame=hof, verbose=True)

    pool.close()

    gen = log.select("gen") 
    fit_mins = log.chapters["fitness"].select("max")
    best_fit = truncate(hof[0].fitness.values[0], 0)
    # nodes, edges, labels = gp.graph(hof[0])

    create_sheet(['ind', 'fitness'], str(GRAV), PATH_TO_WRITE)
    append_to_excel=[]
    append_to_excel.append(str(hof[0]))
    append_to_excel.append(best_fit)
    write_to_excel(append_to_excel, str(GRAV), PATH_TO_WRITE)

    # Prints the fitness score of the best individual
    # print(best_fit)

    # Prints the individual's tree in string form
    # print(hof[0])

    # Graphs the fitness score of every ind over the generations and displays it
    # plot_onto_graph(gen, fit_mins, best_fit)

    # Creates an env and displays the best ind being tested in the env
    # fullObsEvalIndividual(hof[0], pset, 9.81, True)

    return pop, log, hof

if __name__ == "__main__":
    main()
