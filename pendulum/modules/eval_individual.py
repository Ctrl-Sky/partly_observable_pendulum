from deap import gp
from deap import tools
from inspect import isclass
import gymnasium as gym
import random
import sys

# Evaluate fitness of individual
def fullObsEvalIndividual(individual, pset, grav, test=False):
    # Set up the enviornment and gravity
    env_train = gym.make('Pendulum-v1', g=grav) # For training
    env_test = gym.make('Pendulum-v1', g=grav, render_mode="human") # For rendering
    env = env_train
    num_episode = 30

    if test:
        env = env_test
        num_episode = 1
    
    # Transform the tree expression to functional Python code
    get_action = gp.compile(individual, pset)
    fitness = 0
    failed = False
    for x in range(0, num_episode):
        # Set up the variables for the env
        done = False
        truncated = False
        observation = env.reset()
        observation = observation[0]
        episode_reward = 0
        num_steps = 0
        max_steps=300
        timeout=False

        while not (done or timeout):
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
            if num_steps >= max_steps:
                timeout=True

        fitness += episode_reward
    fitness = fitness/num_episode      
    return (0,) if failed else (fitness,)

def partObsEvalIndividual(individual, pset, grav, test=False):
    env_train = gym.make('Pendulum-v1', g=grav) # For training
    env_test = gym.make('Pendulum-v1', g=grav, render_mode="human") # For rendering the best one
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
        max_steps = 300
        timeout = False

        prev_y = observation[0]
        prev_x = observation[1]
        last_y = observation[0]
        last_x = observation[1]

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


def indexMemEvalIndividual(individual, pset, grav, test=False):
    env_train = gym.make("Pendulum-v1", g=grav)  # For training
    env_test = gym.make("Pendulum-v1", g=grav, render_mode="human")  # For rendering the best one
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
        observation = env.reset()
        observation = observation[0]
        episode_reward = 0
        num_steps = 0
        max_steps = 300
        timeout = False
        memory = [0.0]
        while not (done or timeout):
            if failed:
                action = 0
            else:
                # use the tree to compute action
                action = get_action(memory, observation[0], observation[1])
                action = (action,)
            try:
                # returns the new observation, reward, done, truncated, info
                observation, reward, done, truncated, info = env.step(action)
            except:
                failed = True
                observation, reward, done, truncated, info = env.step(0)
            episode_reward += reward

            num_steps += 1
            if num_steps >= max_steps:
                timeout = True

        fitness += episode_reward
    fitness = fitness / num_episode
    return (0,) if failed else (fitness,)
    
def varAnd(population, toolbox, cxpb, mutpb):
    r"""Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(
                offspring[i - 1], offspring[i]
            )
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            (offspring[i],) = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring

def eaSimple_early_stop(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    sufficient_fitness=0.0,
):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    gen = 0
    stop = False
    while gen <= ngen and stop == False:
        gen += 1
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            if fit[0] >= sufficient_fitness:
                stop = True

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def generate_typed_safe(pset, min_, max_, type_=None):
    """Generate a tree as a list of primitives and terminals in a depth-first
    order. The tree is built from the root to the leaves, and it stops growing
    the current branch when the *condition* is fulfilled: in which case, it
    back-tracks, then tries to grow another branch until the *condition* is
    fulfilled again, and so on. The returned list can then be passed to the
    constructor of the class *PrimitiveTree* to build an actual tree object.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """

    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if (
            len(pset.primitives[type_]) == 0
            or height == depth
            or random.random() < 0.5
        ):  # condition
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The generate function tried to add "
                    "a terminal of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The generate function tried to add "
                    "a primitive of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr