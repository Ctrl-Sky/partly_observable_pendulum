# genetic programming for Gymnasium Cart Pole task
# https://gymnasium.farama.org/environments/classic_control/cart_pole/

import numpy
import random
import gymnasium as gym
import operator

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from scoop import futures
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

pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(operator.abs, 1)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(if_then_else, 3)
# pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(conditional, 2)
pset.addPrimitive(limit, 3)

pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.addTerminal(0)
pset.addTerminal(1)

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

env_train = gym.make('MountainCar-v0') # For training
env_test = gym.make('MountainCar-v0', render_mode="human") # For rendering the best one

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
                if test:
                    print(observation[0], observation[1], action)
                
            try: observation, reward, done, truncated, info = env.step(action) # env.step will return the new observation, reward, don, truncated, info
            except:
                
                failed = True
                observation, reward, done, truncated, info = env.step(0)
            
            v = abs(observation[0])
            if v > 0.5:
                reward = -(1 - (v - 0.5))
            if v < 0.5:
                reward = -(1 - (0.5 - v))
            episode_reward += reward
        fitness += episode_reward
    fitness = fitness/num_episode        
    return (0,) if failed else (fitness,)

toolbox.register("evaluate", evalIndividual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),
                                        max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"),
                                          max_value=17))

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

    pop, log = algorithms.eaSimple(pop, toolbox, 0.2, 0.5, 10, stats=mstats,
                                   halloffame=hof, verbose=True)
    # evaluate best individual with visualization
    winner = gp.compile(hof[0], pset)
    evalIndividual(hof[0], True)
    # save graph of best individual
    graph(hof[0], 'out')

    # print log
    return pop, log, hof

if __name__ == "__main__":
    main()

# limit as top node seems best

# changed mutation (0.2 -> 0.5) and cross probabilty (0.5 -> 0.2)
# -0.5 is centre
