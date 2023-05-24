import copy
from functools import partial
import random
import matplotlib.pyplot as plt
import numpy
import pygraphviz as pgv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

def displayBest(hof):
    nodes, edges, labels = gp.graph(hof[0])

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("best_tree.pdf")

# *args allows us to pass any number of non-keyword arguments to a Python function
def progn(*args):
    for arg in args:
        arg()

def prog2(out1, out2):
    return partial(progn, out1, out2)

def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)

def if_then_else(condition, out1, out2):
    out1() if condition() else out2()

class AntSimulator(object):
    direction = ["north", "east", "south", "west"]
    dir_row = [1,0,-1,0] # + 1 if facing north and - 1 for south
    dir_col = [0,1,0,-1] # + 1 if facing east and - 1 for west

    def __init__ (self, max_moves):
        self.max_moves = max_moves
        self.moves = 0
        self.eater = 0
        self.routine = None

    def _reset(self):
        self.row = self.row_start
        self.col = self.col_start
        self.dir =  1
        self.moves = 0
        self.eaten = 0
        self.matrix_exc = copy.deepcopy(self.matrix) # deepcopy makes an exact copy of the matrix but any changes does not affect eachother

    # decorator in python that defines the get method for position
    # so you can just call position by something like ant.position
    @property 
    def position(self):
        return (self.row, self.col, self.direction[self.dir])
    
    def turn_left(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.dir = (self.dir - 1) % 4

    def turn_right(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.dir = (self.dir + 1) % 4

    def move_forward(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.row = (self.row + self.dir_row[self.dir]) % self.matrix_row # the modolus is used in case someone walks out of bouds
            self.col = (self.col + self.dir_col[self.dir]) % self.matrix_col # it places them on the other side, similar to pac man
            if self.matrix_exc[self.row][self.col] == "food":
                self.eaten += 1
            self.matrix_exc[self.row][self.col] = "passed"

    def sense_food(self):
        ahead_row = (self.row + self.dir_row[self.dir]) % self.matrix_row
        ahead_col = (self.row + self.dir_col[self.dir]) % self.matrix_col
        return self.matrix_exc[ahead_row][ahead_col] == "food"
    
    def if_food_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_food, out1, out2)
    
    def run(self, routine):
        self._reset()
        while self.moves < self.max_moves:
            routine()

    # goes through 2D array and sets up board by placing food and starting point
    def parse_matrix(self, matrix):
        self.matrix = list()
        for i, line in enumerate(matrix):
            self.matrix.append(list())
            for j, col in enumerate(line):
                if col == "#":
                    self.matrix[-1].append("food")
                elif col == ".":
                    self.matrix[-1].append("empty")
                elif col == "S":
                    self.matrix[-1].append("empty")
                    self.row_start = self.row = i
                    self.col_start = self.col = j
                    self.dir = 1
        self.matrix_row = len(self.matrix)
        self.matrix_col = len(self.matrix[0])
        self.matrix_exc = copy.deepcopy(self.matrix)

ant = AntSimulator(600) # 600 max moves

pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(ant.if_food_ahead, 2)
pset.addPrimitive(prog2, 2)
pset.addPrimitive(prog3, 3)
pset.addTerminal(ant.move_forward)
pset.addTerminal(ant.turn_left)
pset.addTerminal(ant.turn_right)

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=2)

# Structure Initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalArtificialAnt(individual):
    # Transform tree expression to functional python code
    routine = gp.compile(individual, pset)
    ant.run(routine)
    return ant.eaten,

toolbox.register("evaluate", evalArtificialAnt)
# selects the ant with the best fitness score from a population hosting k tournaments size of tournsize
toolbox.register("select", tools.selTournament, tournsize=7)
# performs one point cross over between 2 parent individual and creates new individual
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
# performs uniform mutations on the individual
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main(verbose=True):
    print()
    random.seed(69) # saves the spot of the next random generator

    NGEN = 40
    CXPB = 0.5
    MUTPB = 0.2

    with open("Sky/santafe_trail.txt") as f:
        ant.parse_matrix(f)
        
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=mstats, halloffame=hof, verbose = True)

    displayBest(hof)

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
    ax1.legend(lns, labs, loc="center right") # Adds then a legend

    plt.show()
            
    return pop, log, hof

if __name__ == "__main__":
    main()