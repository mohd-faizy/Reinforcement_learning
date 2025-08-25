"""
Cliff Walking System -> Theoretical Overview

This environment is based on the OpenAI Gymnasium implementation:

* System Description:
   * A 4x12 grid world.
   * Start state: bottom-left corner (row 3, column 0).
   * Goal state: bottom-right corner (row 3, column 11).
   * Cliff states: all cells between start and goal along the bottom row (row 3, columns 1–10).
   * If the agent steps into the cliff, the episode ends immediately with a heavy penalty.

* Control Objective:
   * Navigate from the start state to the goal state safely.

* Action Space (Discrete: 4):
   * 0 → Move Up
   * 1 → Move Right
   * 2 → Move Down
   * 3 → Move Left

* Observation Space (States):
   * Discrete state space of size 4 × 12 = 48 states (numbered 0–47).

* Reward System:
   * Normal step: -1 reward.
   * Falling into the cliff: -100 reward (episode terminates).
   * Reaching the goal: 0 reward (episode terminates).

* Episode Termination:
   * The agent reaches the goal or falls into the cliff.
"""

# ==============================
# Imports
# ==============================
import gymnasium as gym   # OpenAI Gymnasium library for reinforcement learning environments
import time               # For slowing down visualization


# ==============================
# Environment Setup
# ==============================
# "human" mode opens a graphical window to render the gridworld
env = gym.make("CliffWalking-v1", render_mode="human")


# ==============================
# Simulation Settings
# ==============================
episodes = 5       # Number of complete games to play
time_steps = 30    # Maximum steps per episode (the environment may end sooner)


# ==============================
# Main Simulation Loop
# ==============================
for episode in range(episodes):
    # Reset the environment to start a new episode
    state, info = env.reset()
    print(f"\nEpisode {episode+1} | Starting State: {state}")
    
    # Render the environment (displays a graphical window)
    env.render()
    
    # Run one episode - take actions until the goal is reached or cliff is hit
    for t in range(time_steps):
        # Choose a random action (0=Up, 1=Right, 2=Down, 3=Left)
        action = env.action_space.sample()
        
        # Take the action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Print current step info (console logs)
        print(f"Step {t} | Action: {action} | Next State: {next_state} | Reward: {reward}")
        
        # Render the updated environment (shows agent moving)
        env.render()
        
        # Slow down so we can watch the agent move
        time.sleep(0.3)
        
        # Check if the episode has ended
        if terminated or truncated:
            print("Episode finished (Reached Goal or Fell into Cliff).")
            break


# ==============================
# Cleanup
# ==============================
env.close()
