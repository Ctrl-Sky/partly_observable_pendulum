import numpy
import math
import random
import gymnasium as gym
import operator
import matplotlib.pyplot as plt

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

def vel(y2, y1, x2, x1):
    try: y = truncate((y2-y1), 8)
    except: 
        y = 1
    try: x = truncate((x2-x1), 8)
    except:
        x = 1
    v = protectedDiv(y, x)
    return v

def ang_vel(y2, y1, x2, x1):
    top = atan(y2, x2) - atan(y1, x1)
    bottom = y2 - y1
    return protectedDiv(top, bottom)

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

def atan(x, y):
    return math.atan(protectedDiv(x, y))

def stabilize(x1, x2, y1, y2):

    if (x2 - x1) < 0:
        return -y1
    else:
        return y2
    
    top = acos(x2, y2) - acos(x1, y1)
    bottom = y2 - y1
    return protectedDiv(top, bottom)

def delta(x, y):
    return (x - y)

def con2(input1, input2):
    if input1 < input2:
        return input2
    else:
        return -input2

# Set up primitives and terminals
pset = gp.PrimitiveSet("MAIN", 6)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(operator.abs, 1)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(if_then_else, 3)
pset.addPrimitive(conditional, 2)
pset.addPrimitive(limit, 3)
pset.addPrimitive(vel, 4)
pset.addPrimitive(acos, 2)
pset.addPrimitive(asin, 2)
pset.addPrimitive(atan, 2)
pset.addPrimitive(math.tan, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(stabilize, 2)
pset.addPrimitive(ang_vel, 4)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(delta, 2)
pset.addPrimitive(con2, 2)
pset.addPrimitive(min, 2)

pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.addTerminal(0)
pset.addTerminal(1)

pset.renameArguments(ARG0='y1')
pset.renameArguments(ARG1='x1')
pset.renameArguments(ARG2='y2')
pset.renameArguments(ARG3='x2')
pset.renameArguments(ARG4='y3')
pset.renameArguments(ARG5='x3')

env_train = gym.make('Pendulum-v1', g = 9.81) # For training
env_test = gym.make('Pendulum-v1', g = 9.81, render_mode="human") # For rendering the best one

xs = []
vels = []

def plot_as_tree(nodes, edges, labels, best_fit):
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw('/Users/sky/Documents/Work Info/Research Assistant/deap_experiments/Sky/pendulum/graphs/'+best_fit+".pdf")

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

        num_steps = 0
        prev_y = observation[0]
        prev_x = observation[1]
        last_y = observation[0]
        last_x = observation[1]

        # prev_y = 0
        # prev_x = 0
        # last_y = 0
        # last_x = 0


        while not (done or truncated):
            if failed:
                action = 0
            else:
                # use the tree to compute action, plugs values of observation into get_action
                print(truncate(observation[0], 3), truncate(prev_y, 3), truncate(last_y, 3), truncate(observation[1], 3), truncate(prev_x, 3), truncate(last_x, 3))

                if num_steps == 0:
                    action = get_action(observation[0], observation[1], prev_y, prev_x, last_y, last_x)
                    prev_y = observation[0]
                    prev_x = observation[1]
                else:
                    action = get_action(observation[0], observation[1], prev_y, prev_x, last_y, last_x)
                    temp_y = prev_y
                    temp_x = prev_x
                    prev_y = observation[0]
                    prev_x = observation[1]
                    last_y = temp_y
                    last_x = temp_x


                
                action = (action,)

            try: observation, reward, done, truncated, info = env.step(action) # env.step will return the new observation, reward, done, truncated, info
            except:
                failed = True
                observation, reward, done, truncated, info = env.step(0)
            episode_reward += reward

            num_steps += 1

        fitness += episode_reward
    fitness = fitness/num_episode 
    print(individual)       
    return (0,) if failed else (fitness,)
    
def evalIndividual300(individual, test=False):
    env = env_train
    num_episode = 30 # Basically the amount of simulations ran
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
        max_steps = 400
        timeout = False

        prev_y = observation[0]
        prev_x = observation[1]
        last_y = observation[0]
        last_x = observation[1]

        while not (done or timeout):
            if failed:
                action = 0
            else:
                print(truncate(observation[0], 3), truncate(prev_y, 3), truncate(last_y, 3), truncate(observation[1], 3), truncate(prev_x, 3), truncate(last_x, 3))

                # use the tree to compute action, plugs values of observation into get_action
                
                                    
                if num_steps == 0:
                    action = get_action(observation[0], observation[1], prev_y, prev_x, last_y, last_x)
                    prev_y = observation[0]
                    prev_x = observation[1]
                else:
                    action = get_action(observation[0], observation[1], prev_y, prev_x, last_y, last_x)
                    temp_y = prev_y
                    temp_x = prev_x
                    prev_y = observation[0]
                    prev_x = observation[1]
                    last_y = temp_y
                    last_x = temp_x
                # action = get_action(observation[0], observation[1], observation[2])
                
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


# str = 'vel(sub(x3, y2), sub(conditional(x2, y3), x3), vel(x3, y2, x3, y2), add(vel(add(y2, y1), conditional(x2, x1), vel(y2, x2, x2, x1), conditional(x2, x1)), y1))'
# str = 'vel(neg(x3), x2, vel(x1, y3, x1, y1), conditional(y2, add(x3, x1)))'
# str = 'protectedDiv(asin(ang_vel(sin(ang_vel(y1, x1, y3, y3)), y3, ang_vel(x1, x2, x3, x3), x1), cos(asin(max(conditional(acos(x2, x3), protectedDiv(x1, y1)), y2), x2))), sin(acos(y1, y2)))'


str = 'add(sin(x3), asin(min(ang_vel(y3, x3, y3, x3), min(limit(x2, x2, x1), conditional(x1, limit(limit(x2, x2, min(limit(add(protectedDiv(y3, acos(max(limit(y1, y2, y3), y3), max(y3, if_then_else(y2, x1, x2)))), conditional(x1, conditional(asin(x1, y3), limit(x2, y3, y2)))), x1, x1), acos(min(x2, x1), protectedDiv(y1, y2)))), y3, y1)))), x2))'
s = gp.PrimitiveTree.from_string(str, pset)
print(evalIndividual300(s, True))


graph(s, 'test')
nodes, edges, labels = gp.graph(s)
# plot_as_tree(nodes, edges, labels, 'a')

counts = range(len(xs))

fig, ax1 = plt.subplots() # Allows you to create multiple plots in one figure
line1 = ax1.plot(counts, xs, "b-", label="x") # Plots using gen as x value and fit_mins as y, both are list
ax1.set_xlabel("steps")
ax1.set_ylabel("x", color="b")
for tl in ax1.get_yticklabels(): # Changes colour of ticks and numbers on axis
    tl.set_color("b")

ax2 = ax1.twinx() # Creates ax2 that shares the same x axis and ax1
line2 = ax2.plot(counts, vels, "r-", label="vel")
ax2.set_ylabel("vel", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

lns = line1 + line2 # lns is a list containing both lines [line1, line2]
labs = [l.get_label() for l in lns] # labs contains the labels of each line (Minimum Fitness and Average Size)
ax1.legend(lns, labs, loc="lower right") # Adds then a legend

# plt.show()