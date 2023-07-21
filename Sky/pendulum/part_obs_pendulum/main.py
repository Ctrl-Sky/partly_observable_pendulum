# genetic programming for Gymnasium Cart Pole task
# https://gymnasium.farama.org/environments/classic_control/mountain_car/

import numpy
import operator
import math

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#parallel
import multiprocessing

# Import modules from different directory
import sys
sys.path.append('/Users/sky/Documents/Work Info/Research Assistant/deap_experiments/Sky/pendulum')

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

# Prepare individual and pendulum
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register functions in the toolbox needed for evolution
toolbox.register("evaluate", partObsEvalIndividual, pset=pset, grav=9.81)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pool = multiprocessing.Pool(processes=2) # parllel (Process Pool of 16 workers)
    toolbox.register("map", pool.map) # parallel

    pop, log = algorithms.eaSimple(pop, toolbox, 0.2, 0.5, 1, stats=mstats, halloffame=hof, verbose=True)

    pool.close() # parallel
    
    gen = log.select("gen") 
    fit_mins = log.chapters["fitness"].select("max")
    best_fit = truncate(hof[0].fitness.values[0], 0)
    nodes, edges, labels = gp.graph(hof[0])

    print(best_fit)
    print(hof[0])
    plot_onto_graph(gen, fit_mins, best_fit)
    partObsEvalIndividual(hof[0], pset, 9.81, True)


    fit_mins = best_ind_info(fit_mins, best_fit, hof, labels, True)
    write_to_excel(fit_mins, path="/Users/sky/Documents/Book1.xlsx") # Write to specific excel sheet, comment out if not using Sky's Mac

    return pop, log, hof

if __name__ == "__main__":
    main()
