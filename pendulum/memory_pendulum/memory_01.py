# genetic programming for Gymnasium Cart Pole task
# https://gymnasium.farama.org/environments/classic_control/mountain_car/

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

# Import modules from different directory
import os
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from modules.prim_functions import memProtectedDiv, protectedLog, conditional, limit, truncate, read, write
from modules.output_functions import *
from modules.eval_individual import indexMemEvalIndividual, varAnd, eaSimple_early_stop, generate_typed_safe

# parallel
import multiprocessing

GRAV=18
POP=2
PROCESSES=2
GENS=2
PATH_TO_WRITE='memory_raw_data.xlsx'

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

# Prepare individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", generate_typed_safe, pset=pset, min_=2, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register functions in the toolbox needed for evolution
toolbox.register("evaluate", indexMemEvalIndividual, pset=pset, grav=GRAV)
toolbox.register(
    "select",
    tools.selDoubleTournament,
    fitness_size=3,
    parsimony_size=1.3,
    fitness_first=True
)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    seed = sys.argv[1]  # do args better
    random.seed(seed)
    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pool = multiprocessing.Pool(processes=PROCESSES)  # parllel
    toolbox.register("map", pool.map)  # parallel

    # pop, log = eaSimple_early_stop(
    #     pop,
    #     toolbox,
    #     0.2,
    #     0.75,
    #     1000000,
    #     stats=mstats,
    #     halloffame=hof,
    #     verbose=True,
    #     sufficient_fitness=-200,
    # )

    pop, log = algorithms.eaSimple(pop, toolbox, 0.2, 0.75, GENS, stats=mstats, halloffame=hof, verbose=True)

    pool.close()  # parallel

    gen = log.select("gen")
    best_fits = log.chapters["fitness"].select("max")
    best_fit = truncate(hof[0].fitness.values[0], 0)

    nodes, edges, labels = gp.graph(hof[0])

    save_graph(seed, gen, best_fits, best_fit)
    # indexMemEvalIndividual(hof[0], pset, grav=GRAV, True) # visualize
    plot_as_tree(nodes, edges, labels, best_fit)
    
    create_sheet(['inds', 'fitness'], str(GRAV), PATH_TO_WRITE)
    append_to_excel=[]
    append_to_excel.append(str(hof[0]))
    append_to_excel.append(best_fit)
    write_to_excel(append_to_excel, str(GRAV), PATH_TO_WRITE)

    return pop, log, hof

if __name__ == "__main__":
    main()
