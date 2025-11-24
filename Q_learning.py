"""
Q-Learning agent for the Smart Traffic Light Enviorment
"""

import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob, os

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

#Improved Hyperparameters
EPISODES = 3000
GAMMA = 0.95
EPSILON = 1.0
DECAY_RATE = 0.9992
MAX_STEPS = 120


def get_run_number():
    # Read current run number
    with open("run_counter.txt", "r") as f:
        run_num = int(f.read().strip())
    
    return run_num

def increment_run_number():
    run_num = get_run_number() + 1
    with open("run_counter.txt", "w") as f:
        f.write(str(run_num))

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
    #avg fairness_gap per episode
    episode_fairness = []
    #avg waiting cars per episode
    episode_waiting_cars = []
    #avg ns and ew waiting cars per episode
    episode_ns_waiting_cars = []
    episode_ew_waiting_cars = []

    print(f"{BOLD}Starting training for {num_episodes} episodes...{RESET}")

    for ep in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        state = discretize_state(obs)
        total_reward = 0
        terminated = truncated = False
        steps = 0
        fairness_sum = 0.0
        waiting_cars_sum = 0.0
        ew_waiting_cars_sum = 0.0
        ns_waiting_cars_sum = 0.0

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
            fairness_sum += info["fairness_gap"]
            waiting_cars_sum += info["waiting_cars"]
            ns_waiting_cars_sum += info["cars_NS"]
            ew_waiting_cars_sum += info["cars_EW"]
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

        if steps > 0:
            episode_fairness.append(fairness_sum / steps)
            episode_waiting_cars.append(waiting_cars_sum / steps)
            episode_ew_waiting_cars.append(ew_waiting_cars_sum / steps)
            episode_ns_waiting_cars.append(ns_waiting_cars_sum / steps)
        else:
            episode_fairness.append(0.0)
            episode_waiting_cars.append(0.0)
            episode_ew_waiting_cars.append(info["cars_EW"])
            episode_ns_waiting_cars.append(info["cars_NS"])

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
    run_num = get_run_number()
    plot_filename = f"results/plots/rewards/traffic_rewards_run_{run_num}.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    # Plot fairness gap per episode
    plt.figure(figsize=(10, 6))
    plt.plot(episode_fairness, label='Average Fairness Gap', alpha=0.5)
    if len(episode_fairness) > 10:
        window = min(50, len(episode_fairness)//5)
        moving_avg_fair = np.convolve(episode_fairness, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_fairness)), moving_avg_fair, color='red', label='Moving Average')
    plt.title(f"Average Fairness Gap per Episode (episodes={num_episodes})")
    plt.xlabel("Episode")
    plt.ylabel("Fairness Gap (|cars_NS - cars_EW|)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fair_plot_filename = f"results/plots/fairness/fairness_run_{run_num}.png"
    plt.savefig(fair_plot_filename, dpi=300)
    plt.close()


    # Plot North-South and East-West waiting cars per episode
    plt.figure(figsize=(10, 6))
    plt.plot(episode_ew_waiting_cars, label='East-West Waiting Cars', alpha=0.5)
    plt.plot(episode_ns_waiting_cars, label='North-South Waiting Cars', alpha=0.5)
    if len(episode_fairness) > 10:
        #East-West
        window = min(50, len(episode_ew_waiting_cars)//5)
        moving_avg_ew = np.convolve(episode_ew_waiting_cars, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_ew_waiting_cars)), moving_avg_ew, color='red', label='Moving Average for East-West')
        #North-South
        window = min(50, len(episode_ns_waiting_cars)//5)
        moving_avg_ns = np.convolve(episode_ns_waiting_cars, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_ns_waiting_cars)), moving_avg_ns, color='red', label='Moving Average for North-South')
    plt.title(f"North-South and East-West Waiting Cars per Episode (episodes={num_episodes})")
    plt.xlabel("Episode")
    plt.ylabel("Waiting Cars")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    directional_waiting_cars_filename = f"results/plots/waiting_cars/waiting_cars{run_num}.png"
    plt.savefig(directional_waiting_cars_filename, dpi=300)
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
    run_num = get_run_number()
    q_filename = f"results/q_tables/Q_table_run_{run_num}.pickle"
    with open(q_filename, "wb") as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # increment run counter AFTER saving
    increment_run_number()  

def fixed_timing_baseline(episodes=200):
    """Baseline: simple alternating 30-second lights."""
    rewards = []
    for _ in tqdm(range(episodes), desc="Fixed-Timing Baseline"):
        obs, _ = env.reset()
        total_reward = 0
        light = 0  # Start with N-S
        for step in range(100):
            action = (light, 7)  # 7 * 4 seconds = 28 sec
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            light = 1 - light  # flip direction
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards)

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
    files = glob.glob("results/q_tables/*.pickle")
    if len(files) == 0:
        raise FileNotFoundError("❌ No Q-table found. Run training first:  python Q_learning.py train")

    # choose the most recent
    filename = max(files, key=os.path.getctime)

    print(f"\n{BOLD}Loading saved Q-table: {filename}{RESET}\n")

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
    baseline = fixed_timing_baseline(episodes=200)
    print(f"Fixed Timing Baseline Reward: {baseline:.2f}")
    print(f"Total time: {end_time - start_time:.2f}s")
    print("=" * 60 + "\n")


