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
    try: return left / right
    except ZeroDivisionError: return 1

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def vel(x1, x2, y1, y2):
    try: y = truncate((y2-y1), 8)
    except: 
        y = 1
    try: x = truncate((x2-x1), 8)
    except:
        x = 1
    v = protectedDiv(y, x)
    return v



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

        num_steps = 0
        prev_y = 0
        prev_x = 0
        last_y = 0
        last_x = 0


        while not (done or truncated):
            if failed:
                action = 0
            else:
                # use the tree to compute action, plugs values of observation into get_action
                print(observation[1], prev_x)
                                    
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

                    

                # if test:
                #     x = round(observation[1], 2)
                #     vel = round(observation[2], 2)
                #     xs.append(x)
                #     vel = -vel
                #     vels.append(vel)
                
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
    
str = "vel(conditional(ARG2, ARG0), conditional(ARG0, ARG3), ARG5, add(ARG1, ARG5))"

# str = "vel(vel(vel(ARG5, ARG4, ARG0, ARG4), ARG1, ARG4, ARG0), if_then_else(ARG4, if_then_else(ARG0, ARG3, ARG5), ARG5), add(ARG0, conditional(add(ARG4, ARG3), conditional(ARG3, add(ARG0, ARG3)))), if_then_else(ARG4, ARG3, ARG2))"
# high success rate
# very shaky and wobly at top
# Can not succeed if it does not start with enough velocity
# Can only bring the pendulum up using counter clockwise force

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

# plt.show()