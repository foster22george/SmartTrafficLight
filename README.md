# Smart Traffic Light Control with Q-Learning

## Overview

Urban traffic congestion is a persistent problem, causing delays, wasted fuel, and increased emissions. Traditional traffic lights operate on fixed timers and cannot adapt to varying traffic conditions throughout the day.
This project implements a **Q-learning agent** that learns how to dynamically adjust traffic light timings to reduce waiting times, improve fairness between approaches, and safely integrate pedestrian crossings.

The system is built from scratch using a custom reinforcement learning environment that simulates realistic vehicle behavior, including left and right turns, acceleration, pedestrian crossing, and 'right turn on red' rules.

This work was completed as part of Northeastern University’s CS4100: Artificial Intelligence final project.

---

## Features

### Reinforcement Learning

* Q-learning with discretized states (car queues + pedestrian requests)
* 40 possible actions
  (2 directions × 20 green-light durations)
* epsilon-greedy exploration with decay

### Realistic Traffic Simulation

* Independent queues for **North, South, East, West**
* Left-turn and right-turn traffic with configurable probabilities
* Acceleration model for vehicle flow (faster throughput as green time increases)
* Right-turn-on-red with pedestrian safety constraints
* Turning logic that blocks unsafe right turns when opposing left-turners are present

### Fairness & Pedestrian Handling

* Pedestrians modeled on N-S and E-W crossings
* Waiting pedestrians penalize the reward
* Cars served and pedestrians served tracked per step
* Fairness penalty based on difference between N-S and E-W congestion

### Baseline Comparisons

To evaluate the benefit of learning, the project compares Q-learning against:

1. **Fixed-Timing Baseline**
   Alternates NS/EW every ~30 seconds.

2. **Random Policy Baseline**
   Uniformly random selection of direction and duration.

---

## SmartTrafficLight Repository Structure

```
├── environment.py             # Custom Gymnasium-compatible environment
├── Q_learning.py              # Q-learning agent, training loop, evaluation, baselines
├── results/
│   ├── q_tables/              # Saved Q-tables from training runs
│   └── plots/
│       ├── rewards/           # Training reward curves
│       ├── fairness/          # Fairness gap plots
│       └── waiting_cars/      # Directional waiting plots
├── run_counter.txt            # Simple counter for naming runs
├── requirements.txt           # Install import requirements
├── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd SmartTrafficLight
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

This project uses:

* `numpy`
* `matplotlib`
* `gymnasium`
* `tqdm`

---

## How to Run

### **Train the Q-learning Agent**

```bash
python3 Q_learning.py train
```

This will:

* Train for 3000 episodes
* Save the Q-table in `results/q_tables/`
* Generate performance plots:

  * rewards vs episodes
  * fairness gap
  * directional waiting cars

### **Evaluate the Learned Policy**

```bash
python3 Q_learning.py
```

This loads the most recent Q-table and runs 500 evaluation episodes.
The fixed-timing baseline is also computed for comparison.

---

## Results Summary

During evaluation, the script prints:

* Average reward of the learned policy
* Average reward of fixed-timing baseline
* Total cars served
* Fairness metrics
* Execution time

Plots are saved under `results/plots/`.

---

## Methodology

### **State Representation**

Each state includes:

* Discretized car counts in {0,1,2,3,4,5} bins for N/S/E/W
* Pedestrian request flags for N-S and E-W

### **Action Space**

* Direction: {0 = N-S, 1 = E-W}
* Duration multiplier: {0–19}, mapped to 4–80 sec greens

Total actions: **40**

### **Reward Function**

Rewards encourage:

* Serving many cars
* Serving pedestrians
* Reasonable green light durations
* Balanced queues across directions

Penalties discourage:

* Long queues
* Pedestrian delays
* Extreme fairness differences
* Overly long or extremely short phases

---

## Experiments

### Experiments Conducted

* Q-learning training with decayed exploration
* Baseline comparison with fixed timing
* Analysis of fairness trends
* Evolution of waiting cars for N-S vs E-W
* Effect of turning traffic on throughput

### Outputs Collected

* Per-episode rewards
* Per-episode fairness gap
* Directional waiting time plots
* Saved Q-tables for reproducibility

---

## Running Your Own Experiments

You can modify:

* Arrival rates
* Turning probabilities
* Reward weights
* Max cars
* Episode length
* ε-greedy settings
* Duration multipliers

All parameters are documented and easy to adjust inside `environment.py` and `Q_learning.py`.

---

## Limitations & Future Work

* Only a single intersection is modeled
* No yellow-light or all-red clearance phases
* No multi-lane or lane-changing behavior
* State space is discretized; a deep RL approach could scale better

Ideas for extension:

* Multi-intersection coordination
* Incorporating real traffic datasets
* SARSA or Deep Q-Network (DQN)
* Adding emergency vehicles or buses with priority rules

---

## Contributors

Olivia Pivovar, George Foster, Samik Mukherjee , Michael Pimble
