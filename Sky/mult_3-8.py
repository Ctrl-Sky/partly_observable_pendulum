# import operator
# import numpy
# import random
# from deap import gp
# from deap import tools
# from deap import creator
# from deap import base
# from deap import algorithms

# def if_then_else(input, output1, output2):
#     return output1 if input else output2

# # Initialize multiplexer problem input and output vectors

# MUX_SELECT_LINES = 3
# MUX_IN_LINES = 2 ** MUX_SELECT_LINES
# MUX_TOTAL_LINES = MUX_SELECT_LINES + MUX_IN_LINES

# # input : [A0 A1 A2 D0 D1 D2 ... D7] for a 8-3 mux
# inputs = [[0] * MUX_TOTAL_LINES for i in range(2 ** MUX_TOTAL_LINES)] # Initializes a 2D array
# outputs = [None] * (2**MUX_TOTAL_LINES) # Initializes a 1D array

# for i in range(2 ** MUX_TOTAL_LINES):
#     value = i
#     divisor = 2 ** MUX_TOTAL_LINES

#     # Fill the input bits
#     for j in range(MUX_TOTAL_LINES):
#         divisor /= 2
#         if value >= divisor:
#             inputs[i][j] = 1
#             value -= divisor

#     # Determine Corresponding output
#     indexOutput = MUX_SELECT_LINES
#     for j, k in enumerate(inputs[i][:MUX_SELECT_LINES]): # : is a slice, so it goes through the first 3 elements in the array
#         indexOutput += k * 2**j # j is the index and k the value
#     outputs[i] = inputs[i][indexOutput]


# # inputs = [
# #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  Binary rep of 0
# #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  Binary rep of 1
# #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  Binary rep of 2
# #     ...                                 ...
# #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   Binary rep of 2047
# # ]
# # Inputs will look like this where the first 3 columns represent A0 A1 A2
# # the other columns are D0 D1 D2 ... D7
# # Based on the binary rep of A0 A1 and A2 it will select the respective D#
# # Example: A0 = 1, A1 = 0, A2 = 1 (101 = 5) so it will output D5 in the same row




# pset = gp.PrimitiveSet("MAIN", MUX_TOTAL_LINES, "IN")
# pset.addPrimitive(operator.and_, 2)
# pset.addPrimitive(operator.or_, 2)
# pset.addPrimitive(operator.not_, 1)
# pset.addPrimitive(if_then_else, 3)
# pset.addTerminal(1)
# pset.addTerminal(0)

# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# toolbox = base.Toolbox()
# toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=4)
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("compile", gp.compile, pset=pset)

# def evalMultiplexer(individual):
#     func = toolbox.compile(expr=individual)
#     return sum(func(*in_) == out for in_, out in zip(inputs, outputs)),

# toolbox.register("evaluate", evalMultiplexer)
# toolbox.register("select", tools.selTournament, tournsize=7)
# toolbox.register("mate", gp.cxOnePoint)
# toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# def main(verbose=True, seed=None):
#     random.seed(seed)

#     NGEN = 40
#     CXPB = 0.8
#     MUTPB = 0.1

#     pop = toolbox.population(n=40)
#     hof = tools.HallOfFame(1)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", numpy.mean)
#     stats.register("std", numpy.std)
#     stats.register("min", numpy.min)
#     stats.register("max", numpy.max)

#     logbook = tools.Logbook()
#     logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

#     algo = algorithms.GenerationalAlgorithm(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
#     for gen, state in enumerate(algo):
#         hof.update(state.population)

#         record = stats.compile(state.population)
#         logbook.record(gen=gen, nevals=state.nevals, **record)
#         if verbose:
#             print(logbook.stream)

#         if gen >= NGEN:
#             break

#     return pop, stats, hof

# if __name__ == "__main__":
#     main()

import random
import operator

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

def if_then_else(condition, out1, out2):
    return out1 if condition else out2

# Initialize Multiplexer problem input and output vectors

MUX_SELECT_LINES = 3
MUX_IN_LINES = 2 ** MUX_SELECT_LINES
MUX_TOTAL_LINES = MUX_SELECT_LINES + MUX_IN_LINES

# input : [A0 A1 A2 D0 D1 D2 D3 D4 D5 D6 D7] for a 8-3 mux
inputs = [[0] * MUX_TOTAL_LINES for i in range(2 ** MUX_TOTAL_LINES)]
outputs = [None] * (2 ** MUX_TOTAL_LINES)

for i in range(2 ** MUX_TOTAL_LINES):
    value = i
    divisor = 2 ** MUX_TOTAL_LINES
    # Fill the input bits
    for j in range(MUX_TOTAL_LINES):
        divisor /= 2
        if value >= divisor:
            inputs[i][j] = 1
            value -= divisor

    # Determine the corresponding output
    indexOutput = MUX_SELECT_LINES
    for j, k in enumerate(inputs[i][:MUX_SELECT_LINES]):
        indexOutput += k * 2**j
    outputs[i] = inputs[i][indexOutput]

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

    pop = toolbox.population(n=40)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    algo = algorithms.GenerationalAlgorithm(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
    for gen, state in enumerate(algo):
        hof.update(state.population)

        record = stats.compile(state.population)
        logbook.record(gen=gen, nevals=state.nevals, **record)
        if verbose:
            print(logbook.stream)

        if gen >= NGEN:
            break

    return pop, stats, hof

if __name__ == "__main__":
    main()