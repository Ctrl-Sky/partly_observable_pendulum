import numpy
import math
import random
import gymnasium as gym
import operator
import matplotlib.pyplot as plt
from openpyxl import load_workbook

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import pygraphviz as pgv

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

# user defined funcitons
def conditional(input1, input2):
    if input1 < input2:
        return -input1
    else: return input1

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

def limit(input, minimum, maximum):
    if input < minimum:
        return minimum
    elif input > maximum:
        return maximum
    else:
        return input
    

# Set up primitives and terminals
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(conditional, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(limit, 3)

pset.renameArguments(ARG0='y')
pset.renameArguments(ARG1='x')
pset.renameArguments(ARG2='vel')

xs = []
vels = []

def write_to_excel(fit_mins):
    workbook = load_workbook(filename="/Users/sky/Documents/pendulum_gravity.xlsx")
    sheet = workbook.active

    sheet.append(fit_mins)

    workbook.save(filename="/Users/sky/Documents/pendulum_gravity.xlsx")

def evalIndividual(individual, grav, test=False):
    env_train = gym.make('Pendulum-v1', g=grav) # For training
    env_test = gym.make('Pendulum-v1', g=grav, render_mode="human") # For rendering the best one
    env = env_train

    num_episode = 100 # Basically the amount of simulations ran
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
        num_steps = 0

        while not (done or truncated):
            if failed:
                action = 0
            else:
                # use the tree to compute action, plugs values of observation into get_action
                action = get_action(observation[0], observation[1], observation[2])
                action = (action,)

            try: observation, reward, done, truncated, info = env.step(action) # env.step will return the new observation, reward, done, truncated, info
            except:
                failed = True
                observation, reward, done, truncated, info = env.step(0)
            episode_reward += reward

            num_steps += 1

        fitness += episode_reward
    fitness = fitness/num_episode      
    return (0,) if failed else (fitness,)

def testGravity(s):
    gravity = [1, 2, 3, 4, 5, 6, 7, 8, 9.81, 11, 12, 13]
    for i in gravity:
        add_to_excel = []
        total = 0
        add_to_excel.append(i)

        for j in range(5):
            fit = evalIndividual(s, i, test=False)[0]
            total += fit

        add_to_excel.append(round(total/5, 2))
        write_to_excel(add_to_excel)

str = 'protectedDiv(limit(add(x, conditional(limit(x, x, protectedDiv(vel, x)), protectedDiv(vel, vel))), protectedDiv(vel, x), limit(add(x, limit(protectedDiv(vel, x), vel, limit(y, vel, vel))), limit(y, x, x), y)), conditional(conditional(vel, vel), add(vel, y)))'
s = gp.PrimitiveTree.from_string(str, pset)
print(evalIndividual(s, 11, True))

# testGravity(s)

# nodes, edges, labels = gp.graph(s)
# plot_as_tree(nodes, edges, labels, 'a')

# counts = range(len(xs))

# fig, ax1 = plt.subplots() # Allows you to create multiple plots in one figure
# line1 = ax1.plot(counts, xs, "b-", label="x") # Plots using gen as x value and fit_mins as y, both are list
# ax1.set_xlabel("steps")
# ax1.set_ylabel("x", color="b")
# for tl in ax1.get_yticklabels(): # Changes colour of ticks and numbers on axis
#     tl.set_color("b")

# ax2 = ax1.twinx() # Creates ax2 that shares the same x axis and ax1
# line2 = ax2.plot(counts, vels, "r-", label="vel")
# ax2.set_ylabel("vel", color="r")
# for tl in ax2.get_yticklabels():
#     tl.set_color("r")

# lns = line1 + line2 # lns is a list containing both lines [line1, line2]
# labs = [l.get_label() for l in lns] # labs contains the labels of each line (Minimum Fitness and Average Size)
# ax1.legend(lns, labs, loc="lower right") # Adds then a legend

# plt.show()