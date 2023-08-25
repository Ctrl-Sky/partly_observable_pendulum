import gym
import pygame

gym.register(
    id="GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)