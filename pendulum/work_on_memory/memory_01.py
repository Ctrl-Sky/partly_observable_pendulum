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

# parallel
import multiprocessing


def varAnd(population, toolbox, cxpb, mutpb):
    r"""Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(
                offspring[i - 1], offspring[i]
            )
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            (offspring[i],) = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def eaSimple_early_stop(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    sufficient_fitness=0.0,
):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    gen = 0
    stop = False
    while gen <= ngen and stop == False:
        gen += 1
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            if fit[0] >= sufficient_fitness:
                stop = True

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def generate_typed_safe(pset, min_, max_, type_=None):
    """Generate a tree as a list of primitives and terminals in a depth-first
    order. The tree is built from the root to the leaves, and it stops growing
    the current branch when the *condition* is fulfilled: in which case, it
    back-tracks, then tries to grow another branch until the *condition* is
    fulfilled again, and so on. The returned list can then be passed to the
    constructor of the class *PrimitiveTree* to build an actual tree object.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """

    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if (
            len(pset.primitives[type_]) == 0
            or height == depth
            or random.random() < 0.5
        ):  # condition
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The generate function tried to add "
                    "a terminal of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The generate function tried to add "
                    "a primitive of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


# user defined funcitons


def read(memory, index):
    if math.isinf(index) or math.isnan(index):
        idx = 0
    idx = int(abs(index))
    return memory[idx % len(memory)]


def write(memory, index, data):
    if math.isinf(index) or math.isnan(index):
        idx = 0
    else:
        idx = int(abs(index))
    memory[idx % len(memory)] = data
    return memory[idx % len(memory)]


def limit(input, minimum, maximum):
    if input < minimum:
        return minimum
    elif input > maximum:
        return maximum
    else:
        return input


def protectedLog(input):
    if input <= 0:
        return 0
    else:
        return math.log(input)


def protectedDiv(left, right):
    if (
        math.isinf(right)
        or math.isnan(right)
        or right == 0
        or math.isinf(left)
        or math.isnan(left)
        or left == 0
    ):
        return 0
    try:
        return truncate(left, 8) / truncate(right, 8)
    except ZeroDivisionError:
        return 0


def truncate(number, decimals=0):
    if math.isinf(number) or math.isnan(number):
        return 0
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0**decimals
    num = number * factor
    if math.isinf(num) or math.isnan(num):
        return 0
    return math.trunc(num) / factor


def conditional(input1, input2):
    if input1 < input2:
        return -input1
    else:
        return input1


# make a revered copy of input (a list) and return it
def listReverse(input):
    rev = input
    rev.reverse()
    return rev


# def listLength(input):
#     return len(input)


def acos(x, y):
    if protectedDiv(x, y) < 1 and protectedDiv(x, y) > -1:
        return math.acos(x / y)
    elif protectedDiv(y, x) < 1 and protectedDiv(y, x) > -1:
        return math.acos(y / x)
    else:
        return x


def asin(x, y):
    if protectedDiv(y, x) < 1 and protectedDiv(y, x) > -1:
        return math.asin(y / x)
    elif protectedDiv(y, x) < 1 and protectedDiv(y, x) > -1:
        return math.asin(y / x)
    else:
        return x


# Set up primitives and terminals
pset = gp.PrimitiveSetTyped("main", [list, float, float], float)

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(protectedLog, [float], float)
pset.addPrimitive(conditional, [float, float], float)
pset.addPrimitive(limit, [float, float, float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin, [float], float)
# pset.addPrimitive(math.acos, [float], float)
# pset.addPrimitive(math.asin, [float], float)
# pset.addPrimitive(math.exp, [float], float)
# pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(operator.abs, [float], float)
# pset.addPrimitive(listReverse, [list], list)
pset.addTerminal(0, float)
# pset.addTerminal(1, float)
# pset.addTerminal(2, float)
# pset.addTerminal(3, float)

pset.addPrimitive(read, [list, float], float)
pset.addPrimitive(write, [list, float, float], float)

pset.renameArguments(ARG0="a0")
pset.renameArguments(ARG1="a1")
pset.renameArguments(ARG2="a2")
# pset.renameArguments(ARG3="a3")

# Prepare individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", generate_typed_safe, pset=pset, min_=2, max_=5)

toolbox.register(
    "individual", tools.initIterate, creator.Individual, toolbox.expr
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

env_train = gym.make("Pendulum-v1", g=9.81)  # For training
env_test = gym.make(
    "Pendulum-v1", g=9.81, render_mode="human"
)  # For rendering the best one


# Takes an individual and makes a tree graph and saves it into trees file
def plot_as_tree(nodes, edges, labels, best_fit):
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw("./fit_" + str(best_fit) + ".png")


# Append the fitness information to an excel sheet
def write_to_excel(fit, path):
    workbook = load_workbook(filename=path)
    sheet = workbook.active

    sheet.append(fit)

    workbook.save(filename=path)


# Creates and shows the graph of the fitness for then entire population
def plot_onto_graph(seed, gen, fit_mins, best_fit):
    colours = ["r-", "g-", "b-", "c-", "m-", "k-"]

    # Simply change the lines in quottation above to change the values you want to graph
    # Allows you to create multiple plots in one figure

    (fig, ax1) = plt.subplots()
    # Plots using gen as x value and fit_mins as y, both are list
    line1 = ax1.plot(
        gen, fit_mins, random.choice(colours), label="Maximum Fitness"
    )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for (
        tl
    ) in ax1.get_yticklabels():  # Changes colour of ticks and numbers on axis
        tl.set_color("b")

    lns = line1  # lns is a list containing both lines [line1, line2]
    # labs contains the labels of each line (Minimum Fitness and Average Size)
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="lower right")  # Adds then a legend

    plt.axis([min(gen), max(gen), min(fit_mins), 0])
    # plt.show()
    plt.savefig(str(seed) + "_fit_curve.png")


# evaluates the fitness of an individual
def evalIndividual(individual, test=False):
    env = env_train
    num_episode = 20
    if test:
        env = env_test
        num_episode = 3

    # Transform the tree expression to functional Python code
    get_action = gp.compile(individual, pset)
    fitness = 0
    failed = False
    for x in range(0, num_episode):
        done = False
        truncated = False
        observation = env.reset()
        observation = observation[0]
        episode_reward = 0
        num_steps = 0
        max_steps = 300
        timeout = False
        memory = [0.0]
        while not (done or timeout):
            if failed:
                action = 0
            else:
                # use the tree to compute action
                action = get_action(memory, observation[0], observation[1])
                action = (action,)
            try:
                # returns the new observation, reward, done, truncated, info
                observation, reward, done, truncated, info = env.step(action)
            except:
                failed = True
                observation, reward, done, truncated, info = env.step(0)
            episode_reward += reward

            num_steps += 1
            if num_steps >= max_steps:
                timeout = True

        fitness += episode_reward
    fitness = fitness / num_episode
    return (0,) if failed else (fitness,)

# Register functions in the toolbox needed for evolution
toolbox.register("evaluate", evalIndividual)
toolbox.register(
    "select",
    tools.selDoubleTournament,
    fitness_size=3,
    parsimony_size=1.3,
    fitness_first=True
)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

toolbox.decorate(
    "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)
toolbox.decorate(
    "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)


def main():
    seed = sys.argv[1]  # do args better
    random.seed(seed)
    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pool = multiprocessing.Pool(processes=96)  # parllel
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

    pop, log = algorithms.eaSimple(pop, toolbox, 0.2, 0.5, 400, stats=mstats, halloffame=hof, verbose=True)


    pool.close()  # parallel

    gen = log.select("gen")
    best_fits = log.chapters["fitness"].select("max")
    best_fit = truncate(hof[0].fitness.values[0], 0)

    nodes, edges, labels = gp.graph(hof[0])

    print(best_fit)
    print(hof[0])
    plot_onto_graph(seed, gen, best_fits, best_fit)
    # evalIndividual(hof[0], True) # visualize
    plot_as_tree(nodes, edges, labels, best_fit)
    # unused, used = find_unused_functions(labels)
    
    append_to_excel=[]
    append_to_excel.append(str(hof[0]))
    append_to_excel.append(best_fit)
    write_to_excel(append_to_excel, 'memory_raw_data.xlsx')

    # inp = input("Pass or fail?: ")
    # notes = input("notes: ")
    # best_fits.append(best_fit)
    # fit_mins.append(inp)
    # if inp == 'passed':
    #     fit_mins.append(str(hof[0]))
    # else:
    #     fit_mins.append(' ')
    # fit_mins.append(unused)
    # fit_mins.append(used)
    # fit_mins.append(notes)

    # write_to_excel(fit_mins)

    return pop, log, hof


if __name__ == "__main__":
    main()
