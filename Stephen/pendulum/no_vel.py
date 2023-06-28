# genetic programming for Gymnasium Cart Pole task
# https://gymnasium.farama.org/environments/classic_control/mountain_car/

import numpy
import random
import gymnasium as gym
import operator
import matplotlib.pyplot as plt
import math
from openpyxl import load_workbook

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import pygraphviz as pgv

#parallel
import multiprocessing



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
    top = acos(x2, y2) - acos(x1, y1)
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
    
def delta(x, y):
    return (x - y)


    if (x2 - x1) < 0:
        return -y1
    else:
        return y2
    

# Set up primitives and terminals
pset = gp.PrimitiveSet("MAIN", 6)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(conditional, 2)
pset.addPrimitive(ang_vel, 4)
pset.addPrimitive(delta, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.sub, 2)

pset.addPrimitive(asin, 2)
pset.addPrimitive(acos, 2)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
# pset.addPrimitive(math.atan, 1)
# pset.addPrimitive(math.tan, 1)
pset.addPrimitive(max, 2)

pset.addPrimitive(limit, 3)
# pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(if_then_else, 3)
# pset.addPrimitive(operator.abs, 1)

# pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
# pset.addTerminal(0)
# pset.addTerminal(1)

pset.renameArguments(ARG0='y1')
pset.renameArguments(ARG1='x1')
pset.renameArguments(ARG2='y2')
pset.renameArguments(ARG3='x2')
pset.renameArguments(ARG4='y3')
pset.renameArguments(ARG5='x3')

# Prepare individual and mountain car
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=3)
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
    g.draw('/Users/sky/Documents/Work Info/Research Assistant/deap_experiments/Sky/pendulum/graphs/'+str(best_fit)+".pdf")

# Append the fitness information to an excel sheet
def write_to_excel(fit):
    workbook = load_workbook(filename="/Users/sky/Documents/Book1.xlsx")
    sheet = workbook.active

    sheet.append(fit)

    workbook.save(filename="/Users/sky/Documents/Book1.xlsx")

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

    plt.axis([min(gen), max(gen), -1000, 0])
    plt.show()

# evaluates the fitness of an individual
def evalIndividual(individual, test=False):
    env = env_train
    num_episode = 20 # Basically the amount of simulations ran
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

        prev_y = 0
        prev_x = 0
        last_y = 0
        last_x = 0

        # prev_y = observation[0]
        # prev_x = observation[1]
        # last_y = observation[0]
        # last_x = observation[1]

        while not (done or timeout):
            if failed:
                action = 0
            else:
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
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pool = multiprocessing.Pool(processes=16) # parllel (Process Pool of 16 workers)
    toolbox.register("map", pool.map) # parallel

    pop, log = algorithms.eaSimple(pop, toolbox, 0.2, 0.5, 25, stats=mstats, halloffame=hof, verbose=True)

    # pool.close() # parallel
    
    gen = log.select("gen") 
    fit_mins = log.chapters["fitness"].select("max")
    best_fit = truncate(hof[0].fitness.values[0], 0)
    nodes, edges, labels = gp.graph(hof[0])

    print(best_fit)
    print(hof[0])
    plot_onto_graph(gen, fit_mins, best_fit)
    evalIndividual(hof[0], True)
    plot_as_tree(nodes, edges, labels, best_fit)
    # unused, used = find_unused_functions(labels)

    # inp = input("Pass or fail?: ")
    # notes = input("notes: ")
    # fit_mins.append(best_fit)
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
