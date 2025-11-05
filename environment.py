import gym
from gym import spaces
import numpy as np
import random

class TrafficLightEnv(gym.Env):
  
    def __init__(self):
        #for reference I used this to build out the init 
        # https://gymnasium.farama.org/introduction/create_custom_env/

        # Define State Space
        self.maxCars = 20
        self.nCarArrivalRate = 0.2
        self.sCarArrivalRate = 0.2
        self.eCarArrivalRate = 0.2
        self.wCarArrivalRate = 0.2

        self.nsPedArrivalRate = 0.1
        self.ewPedArrivalRate = 0.1

        # Rewards
        self.rewards = {
            'goal': 10000,
            'combat_win': 100,
            'combat_loss': -10,
            'defeat': -1000,
            'invalid_action': -5,
            'heal': 50,  # Reward for successfully healing (health increases at heal tile)
            'oob': -5   # Small penalty for attempting to move out of bounds
        }
      
        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "cars": np.array.zeros(4),   # [x, y] coordinates
                "ped": [false,false],  # [x, y] coordinates
            }
        )

        # Define what actions are available (either north-south lights are green or east-west lights)
        self.action_space = gym.spaces.Discrete(2)

  
