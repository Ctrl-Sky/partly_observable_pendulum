#    This experiment is compatible with SCOOP. 
#    After installing scoop you can run the program with python -m scoop basic-gp.py
#    This file uses DEAP (https://github.com/DEAP/deap)
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import numpy
# import gym
import gymnasium as gym
from gym import wrappers
import operator

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from scoop import futures

import pygraphviz as pgv


def progn(*args):
    for arg in args:
        arg()


# def if_then_else(condition, out1, out2):
#     out1() if condition() else out2()

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

# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

pset = gp.PrimitiveSet("MAIN", 4)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(limit, 3)
pset.addPrimitive(operator.abs, 1)
pset.addPrimitive(if_then_else, 3)
pset.addTerminal(0)
pset.addTerminal(1)



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=5)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


env = gym.make('CartPole-v1')
# env = wrappers.Monitor(env, './out')

def graph(expr):
    nodes, edges, labels = gp.graph(expr)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw('out.png')


def evalIndividual(individual, render=False):
    # Transform the tree expression to functional Python code
    action = gp.compile(individual, pset)
    # graph(individual)
    # print(action)
    fitness = 0
    failed = False
    for x in range(0, 20):
        done = False
        observation = env.reset()
        # print("o0: " + str(observation[0][0]))
        # print("o1: " + str(observation[1]))
        # print("o2: " + str(observation[2]))
        # print("o3: " + str(observation[3]))
        # print("o4: " + str(observation[4]))
        while not done:
            if failed:
                action_result = 0
            else:
                action_result = env.action_space.sample() #action(observation[0], observation[1], observation[2], observation[3])
            # print(action_result)
            try: observation, reward, done, truncated, info = env.step(action_result)
            except:
                failed = True #throw out any individual that throws any type of exception
                observation, reward, done, truncated, info = env.step(0)
                # return (0,) #If your not recording you can reset the environment early,
            if render:
                env.render()
            # if(fitness > 450):
            #     env.render()
            fitness += reward
    # envInUse[i] = False
    return (0,) if failed else (fitness,)

# toolbox.register("map", futures.map)
toolbox.register("evaluate", evalIndividual)
toolbox.register("select", tools.selTournament, tournsize=7)
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    winner = gp.compile(hof[0], pset)
    graph(hof[0])
    evalIndividual(hof[0], True)

    # print log
    return pop, log, hof


    # pop = toolbox.population(n=300)
    # hof = tools.HallOfFame(1)
    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean)
    # stats.register("std", numpy.std)
    # stats.register("min", numpy.min)
    # stats.register("max", numpy.max)
    #
    # algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, halloffame=hof)
    #
    # return pop, hof, stats


if __name__ == "__main__":
    main()
