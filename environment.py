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

        self.right_on_red_rate = 0.3   # cars per second, slow trickle

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

        # Calculate base capacity
        # Cars accelerate for first ~2 seconds before coming to constant speed (around 4 seconds per car)
        base_capacity = math.floor(0.25 * duration * (1 - math.exp(-duration / 2)))

        # Helper function to consume turning cars from the queues
        def consume_turning(cars_to_move, idx):
            """ 
            When cars pass through, consume turning cars first (left + right),
            then straight cars. This keeps turning queues realistic.
            """
            if cars_to_move <= 0:
                return

            # Remove left-turners first
            left_used = min(self.left_turn[idx], cars_to_move)
            self.left_turn[idx] -= left_used
            cars_to_move -= left_used

            # Remove right-turners next
            right_used = min(self.right_turn[idx], cars_to_move)
            self.right_turn[idx] -= right_used
            cars_to_move -= right_used

            # Any remaining cars are straight cars (no extra queue for these)

        # Right turn on red logic
        def right_turn_on_red(red_dirs, ped_idx):
        
            # If pedestrians are crossing this axis → no right turns allowed
            if self.state["peds"][ped_idx]:
                return 0

            total_moved = 0

            for d in red_dirs:
                queue = self.right_turn[d]
                if queue <= 0:
                    continue

                # right-turn trickle capacity based on duration
                rt_capacity = math.floor(self.right_on_red_rate * duration)
                moved = min(queue, rt_capacity)

                # update turning queue and main car queue
                self.right_turn[d] -= moved
                self.state["cars"][d] -= moved

                total_moved += moved

            return total_moved

        #North-South light is green
        if lightGreen == 0 : 
            # right-turners can turn on red (from EW)
            right_turns = right_turn_on_red([2, 3], ped_idx=1)

            # Left turns slow down the lane
            slow_N = 0.6 if self.left_turn[0] > 0 else 1.0
            slow_S = 0.6 if self.left_turn[1] > 0 else 1.0

            # Calculate throughput
            carsThrough_N = min(self.state["cars"][0], math.floor(base_capacity * slow_N))
            carsThrough_S = min(self.state["cars"][1], math.floor(base_capacity * slow_S))
            
            # Update queues
            consume_turning(carsThrough_N, 0)
            self.state["cars"][0] -= carsThrough_N

            consume_turning(carsThrough_S, 1)
            self.state["cars"][1] -= carsThrough_S

            carsThrough = carsThrough_N + carsThrough_S
            carsThrough += right_turns

            if duration > 5 and self.state["peds"][0] == True: 
                helpedPeds = 1
                self.state["peds"][0] = False


        # East- West light is green
        else : 
            #right turners can turn (from NS)
            right_turns = right_turn_on_red([0, 1], ped_idx=0)

            # Left turns slow down the lane
            slow_E = 0.6 if self.left_turn[2] > 0 else 1.0
            slow_W = 0.6 if self.left_turn[3] > 0 else 1.0

            # Calculate throughput
            carsThrough_E = min(self.state["cars"][2], math.floor(base_capacity * slow_E))
            carsThrough_W = min(self.state["cars"][3], math.floor(base_capacity * slow_W))

            # Update queues
            consume_turning(carsThrough_E, 2)
            self.state["cars"][2] -= carsThrough_E

            consume_turning(carsThrough_W, 3)
            self.state["cars"][3] -= carsThrough_W

            carsThrough = carsThrough_E + carsThrough_W
            carsThrough += right_turns

            # Pedestrians crossing E-W
            if duration > 7 and self.state["peds"][1]:
                helpedPeds = 1
                self.state["peds"][1] = False

        # Fairness metrics: compare total NS vs total EW queues
        cars_NS = self.state["cars"][0] + self.state["cars"][1]
        cars_EW = self.state["cars"][2] + self.state["cars"][3]
        fairness_gap = abs(cars_NS - cars_EW)

        # Add arriving cars and split into left/straight/right turn queues
        def add_arrivals(rate, idx):
            incoming = self.np_random.poisson(rate * duration)
            if incoming <= 0:
                return
            left = math.floor(incoming * self.left_prob)
            right = math.floor(incoming * self.right_prob)
            straight = incoming - left - right
            # update turning queues
            self.left_turn[idx] += left
            self.right_turn[idx] += right
            # total cars in this direction = straight + left + right
            self.state["cars"][idx] += incoming
        
        #add cars
        add_arrivals(self.nCarArrivalRate, 0)  # North
        add_arrivals(self.sCarArrivalRate, 1)  # South
        add_arrivals(self.eCarArrivalRate, 2)  # East
        add_arrivals(self.wCarArrivalRate, 3)  # West

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

        waitingCars = np.sum(self.state["cars"])
        if (self.state["peds"][0] or self.state["peds"][1]) : 
            waitingPeds = True

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


            
            


    
