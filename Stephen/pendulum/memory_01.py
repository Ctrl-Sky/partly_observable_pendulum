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

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import pygraphviz as pgv

#parallel
import multiprocessing

# user defined funcitons

def read(memory, index):
    idx = int(abs(index))
    return memory[idx % len(memory)]

def write(memory, index, data):
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
    try: return truncate(left, 8) / truncate(right, 8)
    except ZeroDivisionError: return 0

def truncate(number, decimals=0):
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def conditional(input1, input2):
    if input1 < input2:
        return -input1
    else: return input1

# make a revered copy of input (a list) and return it
def listReverse(input):
    rev = input
    rev.reverse()
    return rev

# def listLength(input):
#     return len(input)

def acos(x, y):
    if protectedDiv(x, y) < 1 and protectedDiv(x, y) > -1:
        return math.acos(x/y)
    elif protectedDiv(y, x) < 1 and protectedDiv(y, x) > -1:
        return math.acos(y/x)
    else:
        return x

def asin(x, y):
    if protectedDiv(y, x) < 1 and protectedDiv(y, x) > -1:
        return math.asin(y/x)
    elif protectedDiv(y, x) < 1 and protectedDiv(y, x) > -1:
        return math.asin(y/x)
    else:
        return x

# Set up primitives and terminals
pset = gp.PrimitiveSetTyped("main", [list, float, float, float], float)

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
pset.addPrimitive(listReverse, [list], list)
pset.addTerminal(0, float)
pset.addTerminal(1, float)
pset.addTerminal(2, float)
pset.addTerminal(3, float)

pset.addPrimitive(read, [list, float], float)
pset.addPrimitive(write, [list, float, float], float)

pset.renameArguments(ARG0='a0')
pset.renameArguments(ARG1='a1')
pset.renameArguments(ARG2='a2')
pset.renameArguments(ARG3='a3')

# Prepare individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=3)
# terminal_types = [list, float]
# toolbox.register("expr", generate_safe, pset=pset, min_=1, max_=10, terminal_types=terminal_types)
toolbox.register("expr", gp.genGrow, pset=pset, min_=2, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

env_train = gym.make('Pendulum-v1', g=9.81) # For training
env_test = gym.make('Pendulum-v1', g=9.81, render_mode="human") # For rendering the best one

# Takes an individual and makes a tree graph and saves it into trees file
def plot_as_tree(nodes, edges, labels, best_fit):
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw('./'+str(best_fit)+".png")

# Append the fitness information to an excel sheet
def write_to_excel(fit):
    workbook = load_workbook(filename="./Book1.xlsx")
    sheet = workbook.active

    sheet.append(fit)

    workbook.save(filename="./Book1.xlsx")

# Creates and shows the graph of the fitness for then entire population
def plot_onto_graph(gen, fit_mins, best_fit):
    colours = ['r-', 'g-', 'b-', 'c-', 'm-', 'k-']

    # Simply change the lines in quottation above to change the values you want to graph

    fig, ax1 = plt.subplots() # Allows you to create multiple plots in one figure
    line1 = ax1.plot(gen, fit_mins, random.choice(colours), label="Maximum Fitness") # Plots using gen as x value and fit_mins as y, both are list
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels(): # Changes colour of ticks and numbers on axis
        tl.set_color("b")

    lns = line1 # lns is a list containing both lines [line1, line2]
    labs = [l.get_label() for l in lns] # labs contains the labels of each line (Minimum Fitness and Average Size)
    ax1.legend(lns, labs, loc="lower right") # Adds then a legend

    plt.axis([min(gen), max(gen), min(fit_mins), 0])
    plt.show()

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
        observation = env.reset() # Reset the pole to some random location and defines the things in observation
        observation = observation[0]
        episode_reward = 0
        num_steps = 0
        max_steps = 300
        timeout = False
        memory = [0.0, 0.0, 0.0, 0.0] # a list of floats that will be persistent throughout each episode
        while not (done or timeout):
            
            if failed:
                action = 0
            else:
                # use the tree to compute action, plugs values of observation into get_action
                action = get_action(memory, observation[0], observation[1], observation[2])
                action = (action, )
            try: observation, reward, done, truncated, info = env.step(action) # env.step will return the new observation, reward, done, truncated, info
            except:
                failed = True
                observation, reward, done, truncated, info = env.step(0)
            episode_reward += reward

            num_steps += 1
            if num_steps >= max_steps:
                timeout = True
            
        fitness += episode_reward
    fitness = fitness/num_episode        
    return (0,) if failed else (fitness,)

def find_unused_functions(labels):
    used_functions = set(list(labels.values()))
    all_functions = {'add', 'conditonal', 'ang_vel', 'sub', 'asin', 'acos', 'sin', 'cos', 'max', 'limit', 'delta', 'protectedDiv', 'y1', 'y2', 'y3', 'x1', 'x2', 'x3'}
    unused_functions = all_functions.difference(used_functions)

    string1 = ''
    for i in unused_functions:
        string1 = string1 + i +', '

    string2 = ''
    for i in used_functions:
        string2 = string2 + i + ', '

    return string1, string2

# Register functions in the toolbox needed for evolution
toolbox.register("evaluate", evalIndividual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pool = multiprocessing.Pool(processes=18) # parllel (Process Pool of 16 workers)
    toolbox.register("map", pool.map) # parallel

    pop, log = algorithms.eaSimple(pop, toolbox, 0.25, 0.5, 100, stats=mstats, halloffame=hof, verbose=True)

    pool.close() # parallel
    
    gen = log.select("gen") 
    best_fits = log.chapters["fitness"].select("max")
    best_fit = truncate(hof[0].fitness.values[0], 0)
    nodes, edges, labels = gp.graph(hof[0])

    print(best_fit)
    print(hof[0])
    # plot_onto_graph(gen, best_fits, best_fit)
    # evalIndividual(hof[0], True) # visualize
    plot_as_tree(nodes, edges, labels, best_fit)
    # unused, used = find_unused_functions(labels)

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
