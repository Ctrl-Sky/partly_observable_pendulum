# THIS CODE IS NOT WORKING BECAUSE 
# FOR SOME REASON algorithms.GenerationalAlgorithm
# "does not exist". EVEN THOUGH IT DOES

import operator
import numpy
import random
import matplotlib.pyplot as plt
import pygraphviz as pgv
from deap import gp
from deap import tools
from deap import creator
from deap import base
from deap import algorithms

def if_then_else(input, output1, output2):
    return output1 if input else output2

def displayBest(hof):
    nodes, edges, labels = gp.graph(hof[0])

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("best_tree.pdf")

# Initialize multiplexer problem input and output vectors

MUX_SELECT_LINES = 3
MUX_IN_LINES = 2 ** MUX_SELECT_LINES
MUX_TOTAL_LINES = MUX_SELECT_LINES + MUX_IN_LINES

# input : [A0 A1 A2 D0 D1 D2 ... D7] for a 8-3 mux
inputs = [[0] * MUX_TOTAL_LINES for i in range(2 ** MUX_TOTAL_LINES)] # Initializes a 2D array
outputs = [None] * (2**MUX_TOTAL_LINES) # Initializes a 1D array

for i in range(2 ** MUX_TOTAL_LINES):
    value = i
    divisor = 2 ** MUX_TOTAL_LINES

    # Fill the input bits
    for j in range(MUX_TOTAL_LINES):
        divisor /= 2
        if value >= divisor:
            inputs[i][j] = 1
            value -= divisor

    # Determine Corresponding output
    indexOutput = MUX_SELECT_LINES
    for j, k in enumerate(inputs[i][:MUX_SELECT_LINES]): # : is a slice, so it goes through the first 3 elements in the array
        indexOutput += k * 2**j # j is the index and k the value
    outputs[i] = inputs[i][indexOutput]


# inputs = [
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  Binary rep of 0
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  Binary rep of 1
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  Binary rep of 2
#     ...                                 ...
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   Binary rep of 2047
# ]
# Inputs will look like this where the first 3 columns represent A0 A1 A2
# the other columns are D0 D1 D2 ... D7
# Based on the binary rep of A0 A1 and A2 it will select the respective D#
# Example: A0 = 1, A1 = 0, A2 = 1 (101 = 5) so it will output D5 in the same row




pset = gp.PrimitiveSet("MAIN", MUX_TOTAL_LINES, "IN")
pset.addPrimitive(operator.and_, 2)
pset.addPrimitive(operator.or_, 2)
pset.addPrimitive(operator.not_, 1)
pset.addPrimitive(if_then_else, 3)
pset.addTerminal(1)
pset.addTerminal(0)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalMultiplexer(individual):
    func = toolbox.compile(expr=individual)
    return sum(func(*in_) == out for in_, out in zip(inputs, outputs)),

# Evaluates Fitness by compiling the individual and saving it to func
# looks kinda like and(and(1, x), or(x, 1))
# Generates a list of 1 and 0 by repeating func(*in_) == out (the asterisk unpacks the list)
# *in_ replaces x
# then sums it to get its fitness score

toolbox.register("evaluate", evalMultiplexer)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main(verbose=True, seed=None):
    random.seed(seed)

    NGEN = 40
    CXPB = 0.8
    MUTPB = 0.1

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # Even though toolbox.mate, mutate, select, and expr_mut are never
    # called they are used in algorithms.eaSimple as a process for evolution
    pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=mstats, halloffame=hof, verbose=True)

    gen = log.select("gen") 
    fit_mins = log.chapters["fitness"].select("max")
    size_avgs = log.chapters["size"].select("avg")
    # Simply change the lines in quottation above to change the values you want to graph

    fig, ax1 = plt.subplots() # Allows you to create multiple plots in one figure
    line1 = ax1.plot(gen, fit_mins, "b-", label="Maximum Fitness") # Plots using gen as x value and fit_mins as y, both are list
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels(): # Changes colour of ticks and numbers on axis
        tl.set_color("b")

    ax2 = ax1.twinx() # Creates ax2 that shares the same x axis and ax1
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2 # lns is a list containing both lines [line1, line2]
    labs = [l.get_label() for l in lns] # labs contains the labels of each line (Minimum Fitness and Average Size)
    ax1.legend(lns, labs, loc="center right") # Adds then a legend

    plt.show()


    # THIS SECTION DOES NOT WORK BECAUSE IT CAN NOT RECGONIZE algorithms.GenerationalAlgorithm
    # -----------------------------------------------------------------------------
    # algo = algorithms.GenerationalAlgorithm(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)

    # for gen, state in enumerate(algo):
    #     hof.update(state.population)

    #     record = stats.compile(state.population)
    #     logbook.record(gen=gen, nevals=state.nevals, **record)
    #     if verbose:
    #         print(logbook.stream)

    #     if gen >= NGEN:
    #         break

    # return pop, stats, hof
    # -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
