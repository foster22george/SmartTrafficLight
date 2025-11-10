import gym
from gym import spaces
import numpy as np
import random

class TrafficLightEnv(gym.Env):
    """
    
    State Space: 
        cars: integer array of length 4 to represent cars waiting in line from each direction
              [North, South, East, West]
        peds: boolean array of length 2 to represent if pedestrians are waiting to walk
              [N-S crossing, E-W crossing]

    Action Space: 
        Tuple(Discrete(2), Discrete(20))
        - First element: which light is green (0=N-S, 1=E-W)
        - Second element: duration multiplier (0-19, representing 3, 6, 9, 12, 15, ..., 60, seconds)
    """

  
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.maxCars = 20
        self.nCarArrivalRate = 0.2
        self.sCarArrivalRate = 0.2
        self.eCarArrivalRate = 0.2
        self.wCarArrivalRate = 0.2
        
        self.nsPedArrivalRate = 0.1
        self.ewPedArrivalRate = 0.1
        

        
        self.observation_space = spaces.Dict({
            "cars": spaces.Box(low=0, high=self.maxCars, shape=(4,), dtype=np.int32),
            "peds": spaces.Discrete(2) 
        })
        
        # Define action space
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),   # Which light direction (N-S or E-W)
            spaces.Discrete(20)    # Duration: 0-19 maps to 3, 6.. seconds
        ))
        
        self.state = None
        self.steps = 0
        
    def _get_obs(self):
        return self.state
    
    def _get_info(self):
        return {
            "steps": self.steps,
            "total_cars_waiting": np.sum(self.state["cars"]),
            "total_peds_waiting": np.sum(self.state["peds"])
        }
    
    def reset(self):

        # set random num of cars and set peds
        self.state = {
            "cars": self.np_random.integers(0, self.maxCars // 2, size=4, dtype=np.int32),
            "peds": self.np_random.integers(0, 2, size=2, dtype=np.int32)
        }
        self.steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        self.steps += 1
         


    
