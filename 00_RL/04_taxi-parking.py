"""
🚕 Taxi-v3 (OpenAI Gymnasium) -> Theoretical Overview

A compact grid-world problem used to test reinforcement learning algorithms.

* **System Description**:
   * A taxi moves on a fixed 5×5 grid.
   * There are 4 special landmark locations: R (red), G (green), Y (yellow), B (blue).
   * One passenger starts at one of these landmarks (or already in the taxi).
   * The taxi must pick up the passenger and drop them off at the specified destination.

* **Control Objective**:
   * Navigate the taxi to the passenger’s location, execute a **pickup**, navigate to the destination, and execute a **dropoff**—with minimal steps and penalties.

* **Action Space (Discrete(6))**:
   * 0 = move south
   * 1 = move north
   * 2 = move east
   * 3 = move west
   * 4 = pickup
   * 5 = dropoff

* **Observation Space (Discrete(500))**:
   * Encodes a tuple: (taxi_row, taxi_col, passenger_location, destination)
   * Combinatorics: 25 taxi positions × 5 passenger states (R/G/Y/B or in taxi) × 4 destinations = **500** states.
   * `env.unwrapped.decode(state)` can be used to recover the tuple.

* **Episode Termination Conditions**:
   * Successful dropoff at the correct destination.
   * (Often) a time limit wrapper ends the episode after a fixed number of steps (commonly ~200).

* **Reward System (typical defaults)**:
   * **-1** per timestep (to encourage shorter solutions)
   * **-10** for illegal pickup/dropoff
   * **+20** for successful dropoff

This benchmark stresses planning and sparse rewards while keeping the state/action spaces small enough to learn quickly with tabular methods (e.g., Q-learning). 
If you’d like, I can swap in a basic Q-learning agent instead of a random policy.
"""

import gymnasium as gym   # Gymnasium RL environments
import numpy as np        # Handy for numeric ops (optional here)
import time               # To slow down visualization if desired

# Create the Taxi environment with on-screen rendering
# "Taxi-v3" is the standard version
# render_mode="human" opens a window (or inline viewer depending on environment)
env = gym.make("Taxi-v3", render_mode="human")

# Simulation settings
episodes = 5      # Number of episodes to run
time_steps = 200  # Cap on steps per episode (env may end earlier)

for episode in range(episodes):
    state, info = env.reset()
    taxi_r, taxi_c, passenger_loc, destination = env.unwrapped.decode(state)
    print(f"\nEpisode {episode+1}")
    print(f"Initial decoded state -> taxi:({taxi_r},{taxi_c}) | passenger_loc:{passenger_loc} | destination:{destination}")

    total_reward = 0

    for t in range(time_steps):
        # Random action: replace with a learned policy for RL experiments
        action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Decode observation for human-friendly logging
        taxi_r, taxi_c, passenger_loc, destination = env.unwrapped.decode(obs)

        print(
            f"Step {t:03d} | Action:{action} | "
            f"State-> taxi:({taxi_r},{taxi_c}) pass_loc:{passenger_loc} dest:{destination} | "
            f"Reward:{reward:+}"
        )

        # Slow the loop so the rendering is watchable (tweak/remove as needed)
        time.sleep(0.05)

        if terminated or truncated:
            reason = "success" if terminated else "time/truncation"
            print(f"Episode finished ({reason}). Total reward: {total_reward}")
            break

# Clean up - close the rendering window
env.close()
