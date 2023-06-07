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


# user defined funcitons
def conditional(input1, input2):
    if input1 < input2:
        return -input1
    else: return input1

def if_(input, output1, output2):
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
    try: return left / right
    except ZeroDivisionError: return 1

# Set up primitives and terminals
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
# pset.addPrimitive(max, 2)
# pset.addPrimitive(operator.abs, 1)
pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(if_then_else, 3)
pset.addPrimitive(conditional, 2)
pset.addPrimitive(limit, 3)

pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.addTerminal(0)
pset.addTerminal(1)

env_train = gym.make('Pendulum-v1', g = 9.81) # For training
env_test = gym.make('Pendulum-v1', g = 9.81, render_mode="human") # For rendering the best one

xs = []
vels = []



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
                
                if test:
                    x = round(observation[1], 2)
                    vel = round(observation[2], 2)
                    xs.append(x)
                    vel = -vel
                    vels.append(vel)

                    # print(observation[0])

                # if action < -2:
                #     action = -2
                # elif action > 2:
                #     action = 2

                # because pendulum has ndarray of (1,) for action, action will not be iterable
                # so must turn it into an iterable for env.step(action) that refers to action as action[0]
                action = (action, )

            try: observation, reward, done, truncated, info = env.step(action) # env.step will return the new observation, reward, done, truncated, info
            except:
                failed = True
                observation, reward, done, truncated, info = env.step(0)
            episode_reward += reward
        fitness += episode_reward
    fitness = fitness/num_episode        
    return (0,) if failed else (fitness,)
    
str = "conditional(ARG0, add(ARG1, add(ARG2, ARG0)))"
print(evalIndividual(str, True))

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

plt.show()