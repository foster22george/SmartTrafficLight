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
        
        self.fairness_weight = 0.25
        
        #Turning Behavior Parameters
        # left_turn[i], right_turn[i] keep track of turning queues in each direction
        # Direction order: [North, South, East, West]
        self.left_turn = np.zeros(4, dtype=np.int32)
        self.right_turn = np.zeros(4, dtype=np.int32)

        # Cars that will be turning left/right
        self.left_prob = 0.15   # ~15% of cars turn left
        self.right_prob = 0.10  # ~10% of cars turn right

        self.observation_space = spaces.Dict({
            "cars": spaces.Box(low=0, high=self.maxCars, shape=(4,), dtype=np.int32),
            "peds": spaces.MultiBinary(2)
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
            "peds_waiting": self.state["peds"][0] or self.state["peds"][1]
        }
    
    def reset(self, seed=None):
        super().reset(seed=seed)

        # initialize the environment RNG properly
        self.np_random = np.random.default_rng(seed)

        # set random num of cars and set peds
        cars_init = self.np_random.integers(0, self.maxCars // 2, size=4, dtype=np.int32)

        self.state = {
            "cars": self.np_random.integers(0, self.maxCars // 2, size=4, dtype=np.int32),
            "peds": self.np_random.integers(0, 2, size=2, dtype=np.int8)
        }

        # initialize turning queues (start with no particular turning cars)
        self.left_turn = np.zeros(4, dtype=np.int32)
        self.right_turn = np.zeros(4, dtype=np.int32)

        self.steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        self.steps += 1
        waitingPeds = False
        waitingCars = 0
        helpedPeds = 0

        lightGreen, time = action
        duration = (time + 1) * 4

        #North-South light is green
        if lightGreen == 0 : 
            # Cars accelerate for ~2 seconds, then reach a realistic flow rate (~0.25 cars/sec, 1 car every 4s)
            carsThrough = math.floor(0.25 * duration * (1 - math.exp(-duration / 2)))
            self.state["cars"][0] = self.state["cars"][0] - carsThrough
            self.state["cars"][1] = self.state["cars"][1] - carsThrough
            
            if duration > 5 and self.state["peds"][0] == True: 
                helpedPeds = 1
                self.state["peds"][0] = False


        # East- West light is green
        else : 
            # Realistic car acceleration for East-West cars
            carsThrough = math.floor(0.25 * duration * (1 - math.exp(-duration / 2)))
            self.state["cars"][2] = self.state["cars"][2] - carsThrough
            self.state["cars"][3] = self.state["cars"][3] - carsThrough

            if duration > 5 and self.state["peds"][1] == True : 
                helpedPeds = 1
                self.state["peds"][1] = False

        waitingCars = np.sum(self.state["cars"])
        if (self.state["peds"][0] or self.state["peds"][1]) : 
            waitingPeds = True

        # Fairness metrics: compare total NS vs total EW queues
        cars_NS = self.state["cars"][0] + self.state["cars"][1]
        cars_EW = self.state["cars"][2] + self.state["cars"][3]
        fairness_gap = abs(cars_NS - cars_EW)

        #add cars 
        self.state["cars"][0] += self.np_random.poisson(self.nCarArrivalRate * duration)
        self.state["cars"][1] += self.np_random.poisson(self.sCarArrivalRate * duration)
        self.state["cars"][2] += self.np_random.poisson(self.eCarArrivalRate * duration)
        self.state["cars"][3] += self.np_random.poisson(self.wCarArrivalRate * duration)

        #make sure not over the max
        self.state["cars"][0] = min(self.state["cars"][0], self.maxCars)
        self.state["cars"][1] = min(self.state["cars"][1], self.maxCars)
        self.state["cars"][2] = min(self.state["cars"][2], self.maxCars)
        self.state["cars"][3] = min(self.state["cars"][3], self.maxCars)

        #make sure not negative
        self.state["cars"][0] = max(self.state["cars"][0], 0)
        self.state["cars"][1] = max(self.state["cars"][1], 0)
        self.state["cars"][2] = max(self.state["cars"][2], 0)
        self.state["cars"][3] = max(self.state["cars"][3], 0)



        #add people
        if self.np_random.random() <= self.pPedArrivesNS : 
            self.state["peds"][0] = True

        if self.np_random.random() <= self.pPedArrivesEW : 
            self.state["peds"][1] = True

        reward = carsThrough + (5 * helpedPeds) - (0.1 * duration) - (.5 * waitingCars) - (5 * waitingPeds) - self.fairness_weight * fairness_gap

        terminated = False  
        truncated = self.steps >= 100  
        
        
        observation = self._get_obs()
        info = self._get_info()
        info.update({
            "cars_served": carsThrough,
            "peds_served": helpedPeds,
            "light_direction": "N-S" if lightGreen == 0 else "E-W",
            "duration": duration,
            "waiting_cars": waitingCars,
            "waiting_peds": waitingPeds,
            "cars_NS": cars_NS,
            "cars_EW": cars_EW,
            "fairness_gap": fairness_gap,
        })
        
        return observation, reward, terminated, truncated, info


            
            


    
