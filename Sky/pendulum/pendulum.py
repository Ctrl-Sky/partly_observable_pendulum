# genetic programming for Gymnasium Cart Pole task
# https://gymnasium.farama.org/environments/classic_control/mountain_car/

import numpy
import random
import gymnasium as gym
import operator
import matplotlib.pyplot as plt
import math

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import pygraphviz as pgv

# user defined funcitons
def conditional(input1, input2):
    if input1 < input2:
        return -input1
    else: return input1

def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

def limit(input, minimum, maximum):
    if input < minimum:
        return minimum
    elif input > maximum:
        return maximum
    else:
        return input
    
def protectedDiv(left, right):
    try: return truncate(left, 8) / truncate(right, 8)
    except ZeroDivisionError: return 1

def truncate(number, decimals=0):
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

# Set up primitives and terminals
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(conditional, 2)
# pset.addPrimitive(protectedDiv, 2)
# pset.addPrimitive(operator.sub, 2)
# pset.addPrimitive(limit, 3)
# pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(if_then_else, 3)
# pset.addPrimitive(max, 2)
# pset.addPrimitive(operator.abs, 1)

# pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
# pset.addTerminal(0)
# pset.addTerminal(1)

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

env_train = gym.make('Pendulum-v1', g=17) # For training
env_test = gym.make('Pendulum-v1', g=17, render_mode="human") # For rendering the best one

# Used to graph the best individual and output to out.png
def graph(expr, str):
    nodes, edges, labels = gp.graph(expr)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw(str+".png")

# Function to calculate the fitness of an individual
def evalIndividual(individual, test=False):
    env = env_train
    num_episode = 20 # Basically the amount of simulations ran
    if test:
        env = env_test
        num_episode = 1
    
    # Transform the tree expression to functional Python code
    get_action = gp.compile(individual, pset)
    fitness = 0
    failed = False
    for x in range(0, num_episode):
        done = False
        truncated = False
        observation = env.reset() # Reset the pole to some random location and defines the things in observation
        observation = observation[0]
        episode_reward = 0
        while not (done or truncated):
            if failed:
                action = 0
            else:
                # use the tree to compute action, plugs values of observation into get_action
                action = get_action(observation[0], observation[1], observation[2])

                action = (action, )

            try: observation, reward, done, truncated, info = env.step(action) # env.step will return the new observation, reward, done, truncated, info
            except:
                failed = True
                observation, reward, done, truncated, info = env.step(0)
            episode_reward += reward
        fitness += episode_reward
    fitness = fitness/num_episode        
    return (0,) if failed else (fitness,)

# Register functions in the toolbox needed for evolution
toolbox.register("evaluate", evalIndividual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    pop = toolbox.population(n=150)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.2, 0.5, 25, stats=mstats, halloffame=hof, verbose=True)

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
    ax1.legend(lns, labs, loc="lower right") # Adds then a legend

    plt.show()

    # evaluate best individual with visualization
    evalIndividual(hof[0], True)
    # save graph of best individual

    # graph(hof[0], 'out')
    print(hof[0])
    print(hof[0].fitness.values)
    return pop, log, hof

if __name__ == "__main__":
    main()
