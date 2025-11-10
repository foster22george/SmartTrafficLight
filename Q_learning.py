"""
Q-Learning agent for the Smart Traffic Light Enviorment
"""

import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from environment import TrafficLightEnv

# Formatting for clean console output
BOLD = '\033[1m'
RESET = '\033[0m'

# Command-line flags
#Usage:
#   python Q_learning.py train  --> trains the model
#   python Q_learning.py        --> runs evaluation mode
train_flag = 'train' in sys.argv  
gui_flag = 'gui' in sys.argv      # placeholder for future visualization

# Environment setup
env = TrafficLightEnv()

#Hyperparameters
EPISODES = 300
GAMMA = 0.9
EPSILON = 1.0
DECAY_RATE = 0.995
MAX_STEPS = 100

#----------------
#HELPER FUNCTIONS
#----------------
def discretize_state(obs):
    """
    Convert observation dictionary into a discrete, hashable state representation.
    
    Args: 
        cars (list[int]): 4-length list, representing number of cars
                          waiting in each direction [North, South, East, West].
        peds (list[int]): 2-length binary list indicating pedestrian requests 
                          [N-S crossing, E-W crossing].
    
    Returns:
        tuple: Tuple containing discretized car bins and pedestrian indicators 
               (cars_bin_0, cars_bin_1, cars_bin_2, cars_bin_3, ped_0, ped_1).
    """
    
    cars = np.clip(obs["cars"], 0, env.maxCars)
    peds = obs["peds"]
    car_bins = tuple(np.round(cars / (env.maxCars / 5)).astype(int))  # bin into 5 levels
    return car_bins + tuple(peds)

def flatten_action(action_tuple):
    """
    Flatten a tuple action into a single integer index.

    Args:
        action (tuple): Tuple containing two discrete values (direction, duration), where:
                        - direction (int): 0 for N-S green light, 1 for E-W green light.
                        - duration (int): Integer in range [0, 19] representing duration multiplier.

    Returns:
        int: Single integer index in the range [0, 39].
    """
    light, duration = action_tuple
    return light * 20 + duration

def unflatten_action(a_idx):
    """
    Unflatten a single integer index back into a tuple action.

    Args:
        index (int): Integer in the range [0, 39] representing the flattened action.

    Returns:
        tuple: Tuple containing two discrete values (direction, duration), where:
                - direction (int): 0 for N-S green light, 1 for E-W green light.
                - duration (int): Integer in range [0, 19] representing duration multiplier.
    """
    light = a_idx // 20
    duration = a_idx % 20
    return (light, duration)

#-------------------
#Q-Learning Function
#-------------------

def Q_learning(num_episodes=EPISODES, gamma=GAMMA, epsilon=EPSILON, decay_rate=DECAY_RATE):
    """
	Run Q-learning algorithm for a specified number of episodes.

	Args:
		num_episodes (int): number of training episodes
		gamma (float): discount factor
		epsilon (float): exploration rate
		decay_rate (float): decay rate for epsilon after each episode

	Returns:
		dict: Q-table containing learned state-action values
	"""
    Q_table = {}
    N_table = {}
    episode_rewards = []

    print(f"{BOLD}Starting training for {num_episodes} episodes...{RESET}")

    for ep in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        state = discretize_state(obs)
        total_reward = 0
        terminated = truncated = False
        steps = 0

        while not (terminated or truncated) and steps < MAX_STEPS:
            # Initialize Q and N entries
            if state not in Q_table:
                Q_table[state] = np.zeros(40)  # 2 directions × 20 durations
                N_table[state] = np.zeros(40)

            # Epsilon-greedy policy
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(0, 40)
            else:
                action_idx = int(np.argmax(Q_table[state]))

            action = unflatten_action(action_idx)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_obs)

            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(40)
                N_table[next_state] = np.zeros(40)

            # Increment visit count
            N_table[state][action_idx] += 1
            eta = 1.0 / (1.0 + N_table[state][action_idx])

            # Q-learning update
            best_next = np.max(Q_table[next_state])
            target = reward + (0 if (terminated or truncated) else gamma * best_next)
            Q_table[state][action_idx] = (1 - eta) * Q_table[state][action_idx] + eta * target

            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        epsilon *= decay_rate

    print(f"{BOLD}Training complete!{RESET}")

    # Plot training rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label='Episode Reward', alpha=0.5)
    if len(episode_rewards) > 10:
        window = min(50, len(episode_rewards)//5)
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, color='red', label='Moving Average')
    plt.title(f"Traffic Light Q-Learning Rewards (episodes={num_episodes})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"traffic_rewards_{num_episodes}.png", dpi=300)
    plt.close()

    return Q_table

#----------------
#Training Control
#----------------
"""
Specify number of episodes and decay rate for training and evaluation.
"""

num_episodes = EPISODES
decay_rate = DECAY_RATE

"""
Run training if train_flag is set; otherwise, run evaluation using saved Q-table.
"""
if train_flag:
    Q_table = Q_learning(num_episodes=num_episodes, gamma=GAMMA, epsilon=EPSILON, decay_rate=decay_rate)

    # Save the trained Q-table
    with open(f"Q_table_{num_episodes}_{decay_rate}.pickle", "wb") as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ---------------
# EVALUATION MODE
# ---------------
"""
Evaluation mode: run episodes using a saved Q-table to assess performance.
"""
def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)

if not train_flag:
    filename = f"Q_table_{EPISODES}_{DECAY_RATE}.pickle"
    input(f"\n{BOLD}Loading saved Q-table from {filename}{RESET}. Press Enter to continue.\n")

    with open(filename, "rb") as f:
        Q_table = pickle.load(f)

    total_eval_episodes = 500
    rewards = []
    start_time = time.time()

    for ep in tqdm(range(total_eval_episodes), desc="Evaluating Agent"):
        obs, info = env.reset()
        state = discretize_state(obs)
        total_reward = 0

        while True:
            # Choose action based on softmax policy if known, else random
            try:
                probs = softmax(Q_table[state])
                action_idx = np.random.choice(len(probs), p=probs)
            except KeyError:
                action_idx = np.random.randint(0, 40)

            action = unflatten_action(action_idx)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = discretize_state(next_obs)

            if terminated or truncated:
                break

        rewards.append(total_reward)

    end_time = time.time()

    # Summary stats
    print("\n" + "=" * 60)
    print(f"{BOLD}Evaluation Summary{RESET}")
    print(f"Total episodes: {total_eval_episodes}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Total time: {end_time - start_time:.2f}s")
    print("=" * 60 + "\n")


