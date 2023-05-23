import operator, random, numpy
from deap import gp, creator, tools, base, algorithms
import pygraphviz as pgv
import matplotlib.pyplot as plt


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


PARITY_FANIN_M = 6
PARITY_SIZE_M = 2**PARITY_FANIN_M

# Initialize Problem by setting up matrices
# ----------------------------------------------

inputs = [None] * PARITY_SIZE_M
outputs = [None] * PARITY_SIZE_M

for i in range(PARITY_SIZE_M):
    inputs[i] = [None] * PARITY_FANIN_M
    value = i
    dividor = PARITY_SIZE_M
    parity = 1
    for j in range(PARITY_FANIN_M):
        dividor /= 2
        if value >= dividor:
            inputs[i][j] = 1
            parity = int(not parity)
            value -= dividor
        else:
            inputs[i][j] = 0
    outputs[i] = parity

# ----------------------------------------------
# This snippet creates two list, inputs which is a 2D matrix and holds
# binary string and outputs that is a 1D list that contains 0 (true) or
# 1 (false) whether the binary string contains an even or odd number of 1s 
# inputs = [
#     [0, 0, 0, 0],  Binary string: 0000, Parity: 0
#     [0, 0, 0, 1],  Binary string: 0001, Parity: 1
#     [0, 0, 1, 0],  Binary string: 0010, Parity: 1
#     [0, 0, 1, 1],  Binary string: 0011, Parity: 0
#     [0, 1, 0, 0],  Binary string: 0100, Parity: 1
#     [0, 1, 0, 1],  Binary string: 0101, Parity: 0
#     [0, 1, 1, 0],  Binary string: 0110, Parity: 0
#     [0, 1, 1, 1]   Binary string: 0111, Parity: 1
# ]
# outputs = [0, 1, 1, 0, 1, 0, 0, 1]


pset = gp.PrimitiveSet("MAIN", PARITY_FANIN_M, "IN")
pset.addPrimitive(operator.and_, 2)
pset.addPrimitive(operator.or_, 2)
pset.addPrimitive(operator.xor, 2)
pset.addPrimitive(operator.not_, 1)
pset.addTerminal(1)
pset.addTerminal(0)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=3, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# -------------------------------------------
def evalParity(individual):
    func = toolbox.compile(expr=individual)
    return sum(func(*in_) == out for in_, out in zip(inputs, outputs)), # Notice the comma

toolbox.register("evaluate", evalParity)
# -------------------------------------------
# Calculates the fitness

# -------------------------------------------
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
# -------------------------------------------
# Used in algorithms.eaSimple for process of evolution

def main():
    random.seed(318)

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
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True)

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

if __name__ == "__main__":
    main()
