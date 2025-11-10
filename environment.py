import math
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
        
        self.pPedArrivesNS = 0.33
        self.pPedArrivesEW = 0.25
        

        
        self.observation_space = spaces.Dict({
            "cars": spaces.Box(low=0, high=self.maxCars, shape=(4,), dtype=np.int32),
            "peds": [False, False]
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
            "peds_waiting": self.observation_space.peds[0] or self.observation_space.peds[1]
        }
    
    def reset(self):

        # set random num of cars and set peds
        self.state = {
            "cars": self.np_random.integers(0, self.maxCars // 2, size=4, dtype=np.int32),
            "peds": self.np_random.boolean(2)
        }
        self.steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        self.steps += 1
        waitingPeds = False
        waitingCars = 0

        lightGreen, time = action
        duration = (time + 1) * 4

        #North-South light is green
        if lightGreen == 0 : 
            carsThrough = math.floor(duration / 2.5)
            self.observation_space.cars[0] = self.observation_space.cars[0] - carsThrough
            self.observation_space.cars[1] = self.observation_space.cars[1] - carsThrough
            
            if duration > 5 and self.observation_space.peds[0] == True: 
                helpedPeds = True
                self.observation_space.peds[0] = False


        # East- West light is green
        else : 
            carsThrough = math.floor(duration / 2.5)
            self.observation_space.cars[2] = self.observation_space.cars[2] - carsThrough
            self.observation_space.cars[3] = self.observation_space.cars[3] - carsThrough

            if duration > 5 and self.observation_space.peds[1] == True : 
                helpedPeds = True
                self.observation_space.peds[1] = False

        waitingCars = np.sum(self.state["cars"])
        if (self.observation_space.peds[0] or self.observation_space.peds[1]) : 
            waitingPeds = True

        #add cars 
        self.observation_space.cars[0] += self.np_random.poisson(self.nCarArrivalRate * duration)
        self.observation_space.cars[1] += self.np_random.poisson(self.sCarArrivalRate * duration)
        self.observation_space.cars[2] += self.np_random.poisson(self.eCarArrivalRate * duration)
        self.observation_space.cars[3] += self.np_random.poisson(self.wCarArrivalRate * duration)

        #make sure not over the max
        self.observation_space.cars[0] = max(self.observation_space.cars[0], self.maxCars)
        self.observation_space.cars[1] = max(self.observation_space.cars[1], self.maxCars)
        self.observation_space.cars[2] = max(self.observation_space.cars[2], self.maxCars)
        self.observation_space.cars[3] = max(self.observation_space.cars[3], self.maxCars)

        #add people
        if np.random(0,1) <= self.pPedArrivesNS : 
            self.observation_space.peds[0] = True

        if np.random(0,1) <= self.pPedArrivesEW : 
            self.observation_space.peds[1] = True

        reward = carsThrough + (5 * helpedPeds) - (0.1 * duration) - (.5 * waitingCars) - (5 * waitingPeds)

        terminated = False  
        truncated = self.steps >= 100  
        
        
        observation = self._get_obs()
        info = self._get_info()
        info.update({
            "cars_served": carsThrough,
            "peds_served": helpedPeds,
            "light_direction": "N-S" if lightGreen == 0 else "E-W",
            "duration": duration
        })
        
        return observation, reward, terminated, truncated, info


            
            


    
