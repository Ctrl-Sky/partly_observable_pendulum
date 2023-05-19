import math, operator, random
import numpy
import matplotlib.pyplot as plt
from deap import gp
from deap import creator
from deap import base
from deap import tools
from deap import algorithms


def protectedDiv(left, right):
    try:
        return left/right
    except ZeroDivisionError:
        return 1
    
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0="x")

# Create Fitness and Genotype
creator.create("Fitness", base.Fitness, weights=(-1.0, ))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

# Defining the tools in the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2) # makes pset into a tree
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) # Using the newly made tree, create an individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Makes a list and fills it with individuals
toolbox.register("compile", gp.compile, pset=pset) # compiles pset so it can be solved


def evalSymbReg(individual, points):
    func = toolbox.compile(expr=individual)

    # Evaluate the mean squred error between the expression
    # and the real function : x^4 + x^3 + x^3 + x
    # From the list points, it uses values from points as x
    # and evaluates the expression and adds it to a list. Fsum then
    # adds them all togethe and gets the mean, this value is the fitness
    # The lower the mean the better. Notice it returns a tuple 
    # because fitness is a tuple
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points),

toolbox.register("evaluate", evalSymbReg, points=[x/10.0 for x in range(-10,10)])

# selects the best among 3 randomly selected individuals
# tools.selTournament(individuals, k, tournsize, fit_attr='fitness')
toolbox.register("select", tools.selTournament, tournsize=3)

# Executes a one point crossover on the input sequence individuals. 
# The two individuals are modified in place. The resulting individuals 
# will respectively have the length of the other.
# tools.cxOnePoint(ind1, ind2)
toolbox.register("mate", gp.cxOnePoint)

# A uniform probability which may append a new full subtree to a node
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Limits the height of the mate and mutate to max depth of 17
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# STATS
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

    return pop, log, hof

if __name__ == "__main__":
    main()
