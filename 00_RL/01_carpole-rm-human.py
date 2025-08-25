"""
Cart Pole System -> Theoretical Overview

Here's a clear, theory-only summary of how the Cart Pole system works, based on the OpenAI Gym environment:

* System Description:
   * A cart moves along a horizontal track.
   * A pole (inverted pendulum) is attached to the cart via a pivot.
   * The goal is to keep the pole balanced upright by moving the cart left or right.

* Control Objective:
   * Apply horizontal forces to the cart to maintain the pole in a vertical position.
   * This mimics real-world control problems like rocket stabilization or robotic balance.

* Action Space:
   * Two discrete actions:
      * Push cart left (action = 0)
      * Push cart right (action = 1)

* Observation Space (State Variables):
   * Cart position (range: -4.8 to 4.8 units)
   * Cart velocity
   * Pole angle (range: -0.418 to 0.418 radians, or ±24°)
   * Pole angular velocity
   * Initial values are randomized within a small range (±0.05)

* Episode Termination Conditions:
   * Pole angle exceeds ±12° (≈±0.2095 radians)
   * Cart position exceeds ±2.4 units
   * Episode length exceeds 500 steps (v1) or 200 steps (v0)

* Reward System:
   * +1 reward for every time step the pole remains upright
   * Longer episodes with stable pole position yield higher cumulative rewards

This setup is a classic benchmark for testing reinforcement learning algorithms, as it requires 
balancing dynamic control with limited actions and noisy observations.
"""

# ==============================
# Imports
# ==============================
import gymnasium as gym   # OpenAI Gym library for reinforcement learning environments
import numpy as np        # For numerical operations (not used here but commonly imported)
import time               # For adding delays to slow down visualization


# ==============================
# Environment Setup
# ==============================
# "CartPole-v1" is the standard version with 500 max steps per episode
# render_mode="human" opens a window to watch the simulation
env = gym.make("CartPole-v1", render_mode="human")


# ==============================
# Simulation Settings
# ==============================
episodes = 25      # Number of complete games to play
time_steps = 100   # Maximum steps per episode (the environment may end sooner)


# ==============================
# Main Simulation Loop
# ==============================
for episode in range(episodes):
   # Reset the environment to start a new episode
   # This returns the initial state and additional info
   state, info = env.reset()   
   print(f"\nEpisode {episode+1}")
   # Run one episode - take actions until the pole falls or time runs out
   for t in range(time_steps):
      # Choose a random action (0 = push left, 1 = push right)
      # In a real RL algorithm, this would be replaced with a trained policy
      action = env.action_space.sample()  
      
      # Take the action in the environment
      # Returns: new observation, reward, whether episode ended, additional info
      obs, reward, terminated, truncated, info = env.step(action)
      
      # Print current step information
      # obs contains: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
      print(f"Step {t} | Obs: {obs} | Reward: {reward}")
      
      # Slow down the simulation so we can watch what's happening
      time.sleep(0.05)  
      
      # Check if the episode has ended
      # terminated: pole fell too far or cart went out of bounds
      # truncated: maximum episode length reached
      if terminated or truncated:
            print("Episode finished.")
            break


# ==============================
# Cleanup
# ==============================
env.close()
