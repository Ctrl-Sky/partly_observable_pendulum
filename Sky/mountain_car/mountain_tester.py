import numpy
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

def h(h):
    if h < 0:
        return -1
    elif h == 0:
        return -1
    else:
        return 2

# Set up primitives and terminals
pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(conditional, 2)
pset.addPrimitive(limit, 3)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(if_, 3)
pset.addPrimitive(conditional, 2)
pset.addPrimitive(h, 1)

pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.addTerminal(0)
pset.addTerminal(1)
pset.addTerminal(2)

env_train = gym.make('MountainCar-v0') # For training
env_test = gym.make('MountainCar-v0', render_mode="human") # For rendering the best one
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
                action = get_action(observation[0], observation[1])

                # Used to limit the action to move left (0) or move right (2)
                if action < 0:
                    action = 0
                else: action =2

                if test: 
                    print(action, observation[1], observation[0])

            try: observation, reward, done, truncated, info = env.step(action) # env.step will return the new observation, reward, don, truncated, info
            except:
                failed = True
                observation, reward, done, truncated, info = env.step(0)
            episode_reward += reward
        fitness += episode_reward
    fitness = fitness/num_episode        
    return (0,) if failed else (fitness,)
    
random.seed(69)
# str = "if_(limit(ARG1, 0, 1), 1, -1)"
str = "h(ARG1)"
print(evalIndividual(str, True))