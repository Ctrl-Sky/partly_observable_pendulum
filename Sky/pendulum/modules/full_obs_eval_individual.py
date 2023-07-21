from deap import gp
import gymnasium as gym

def fullObsEvalIndividual(individual, pset, grav, test=False):
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