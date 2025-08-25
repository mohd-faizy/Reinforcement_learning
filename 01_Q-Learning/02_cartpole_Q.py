"""
🎯 CartPole Balance Game - Q - Learning
========================================================

WHAT IS THIS?
This program teaches an AI to balance a pole on a moving cart using Q-Learning.

WHAT IS Q-LEARNING?
- It's a way for AI to learn by trying actions and remembering which ones work best
- The AI builds a "cheat sheet" (Q-table) that tells it the best action for each situation
- Over time, it gets better at making good decisions
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: SET UP THE GAME ENVIRONMENT
# =============================================================================
# The game gives us 4 numbers that describe the current situation:
# 1. Cart position (where the cart is)
# 2. Cart velocity (how fast the cart is moving)
# 3. Pole angle (how tilted the pole is)
# 4. Pole angular velocity (how fast the pole is rotating)

env = gym.make("CartPole-v1", render_mode="human")


# =============================================================================
# STEP 2: SIMPLIFY THE PROBLEM (DISCRETIZATION)
# =============================================================================
# PROBLEM: The game gives us precise decimal numbers (like 1.23456789)
# SOLUTION: We'll group similar situations into "bins" to make learning easier
# Think of it like organizing clothes: instead of having a drawer for each 
# specific shirt, we have one drawer for "red shirts", one for "blue shirts", etc.

# How many "bins" (categories) for each measurement:
n_bins = [6, 6, 12, 12]  # More bins = more precise, but harder to learn

# Set the ranges we care about for each measurement:
state_bounds = [
    (-2.4, 2.4),    # Cart position: from -2.4 to +2.4 units
    (-3, 3),        # Cart velocity: from -3 to +3 units per second
    (-0.2, 0.2),    # Pole angle: from -0.2 to +0.2 radians (about ±11 degrees)
    (-5, 5)         # Pole angular velocity: from -5 to +5 radians per second
]

def convert_to_bins(measurements):
    """
    🎯 PURPOSE: Convert precise measurements into simple categories (bins)
    
    EXAMPLE: If pole angle is 0.05 radians and we have 12 bins from -0.2 to +0.2,
             this gets converted to bin 7 (roughly in the middle, slightly tilted right)
    """
    bin_indices = []
    
    for i in range(len(measurements)):
        measurement = measurements[i]
        min_val, max_val = state_bounds[i]
        
        # Handle extreme values
        if measurement <= min_val:
            bin_index = 0
        elif measurement >= max_val:
            bin_index = n_bins[i] - 1
        else:
            # Calculate which bin this measurement belongs to
            ratio = (measurement - min_val) / (max_val - min_val)
            bin_index = int(round((n_bins[i] - 1) * ratio))
        
        bin_indices.append(bin_index)
    
    return tuple(bin_indices)

# =============================================================================
# STEP 3: CREATE THE AI'S MEMORY (Q-TABLE)
# =============================================================================

# This is the AI's "cheat sheet" - it remembers the value of each action in each situation
# Think of it as a massive spreadsheet where:
# - Rows = different situations (combinations of cart position, velocity, etc.)
# - Columns = actions (move left or move right)
# - Values = how good that action is in that situation

print("🧠 Creating the AI's memory (Q-table)...")
total_situations = n_bins[0] * n_bins[1] * n_bins[2] * n_bins[3]
num_actions = env.action_space.n  # 2 actions: left (0) or right (1)

q_table = np.zeros(n_bins + [num_actions])
print(f"📊 Q-table size: {total_situations} situations × {num_actions} actions = {total_situations * num_actions} entries")

# =============================================================================
# STEP 4: SET LEARNING PARAMETERS
# =============================================================================

print("\n⚙️ Setting up learning parameters...")

# Learning rate (alpha): How much the AI updates its memory each time
# - Too high = AI forgets old lessons too quickly
# - Too low = AI learns too slowly
learning_rate = 0.1

# Discount factor (gamma): How much the AI cares about future rewards
# - Close to 1.0 = AI plans for the long term
# - Close to 0.0 = AI only cares about immediate rewards
discount_factor = 0.99

# Exploration vs Exploitation balance:
exploration_rate = 1.0        # Start by exploring everything (100% random)
min_exploration = 0.05        # End by exploring only 5% of the time
exploration_decay = 0.995     # How quickly to reduce exploration

# Training settings:
num_episodes = 100           # How many games to play for training
episode_rewards = []         # Keep track of performance

print(f"🎓 Will train for {num_episodes} episodes")
print(f"📈 Learning rate: {learning_rate}")
print(f"🔮 Discount factor: {discount_factor}")

# =============================================================================
# STEP 5: TRAINING THE AI
# =============================================================================

print("\n🏋️ Starting AI training...")
print("=" * 60)

for episode in range(num_episodes):
    # Start a new game
    current_state, _ = env.reset()
    current_bins = convert_to_bins(current_state)
    total_reward = 0
    game_over = False
    
    while not game_over:
        # DECISION TIME: Should the AI explore (try random actions) or exploit (use best known action)?
        if random.random() < exploration_rate:
            # EXPLORE: Try a random action (this helps discover new strategies)
            action = env.action_space.sample()
        else:
            # EXPLOIT: Use the best action we know for this situation
            action = np.argmax(q_table[current_bins])
        
        # Take the action and see what happens
        next_state, reward, terminated, truncated, _ = env.step(action)
        game_over = terminated or truncated
        next_bins = convert_to_bins(next_state)
        
        # UPDATE THE AI'S MEMORY | Q-Learning 
        if game_over:
            target_value = reward
        else:
            best_future_reward = np.max(q_table[next_bins])
            target_value = reward + discount_factor * best_future_reward
        
        current_estimate = q_table[current_bins + (action,)]
        learning_error = target_value - current_estimate
        q_table[current_bins + (action,)] += learning_rate * learning_error
        
        # Move to the next situation
        current_bins = next_bins
        total_reward += reward
    
    # Reduce exploration over time (become more confident in learned strategies)
    exploration_rate = max(min_exploration, exploration_rate * exploration_decay)
    
    # Record this episode's performance
    episode_rewards.append(total_reward)
    
    # Print progress every 50 episodes
    if (episode + 1) % 50 == 0:
        recent_average = np.mean(episode_rewards[-50:])
        print(f"📊 Episode {episode + 1:3d} | Recent average reward: {recent_average:.1f} | Exploration: {exploration_rate:.2f}")

print("\n✅ Training completed!")

# =============================================================================
# STEP 6: ANALYZE TRAINING RESULTS
# =============================================================================

print("\n📈 Analyzing training progress...")

# Plot the learning curve
plt.figure(figsize=(12, 5))

# Left plot: All episode rewards
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, alpha=0.7, color='blue')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Progress: All Episodes")
plt.grid(True, alpha=0.3)

# Right plot: Moving average for clearer trend
plt.subplot(1, 2, 2)
window_size = 50
moving_average = []
for i in range(len(episode_rewards)):
    start = max(0, i - window_size + 1)
    avg = np.mean(episode_rewards[start:i+1])
    moving_average.append(avg)

plt.plot(moving_average, color='red', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Average Reward (50-episode window)")
plt.title("Learning Progress: Smoothed Trend")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Performance summary
final_performance = np.mean(episode_rewards[-100:])
print(f"🎯 Final performance (last 100 episodes): {final_performance:.1f} points")
print(f"🏆 Best single episode: {max(episode_rewards):.0f} points")

# =============================================================================
# STEP 7: TEST THE TRAINED AI
# =============================================================================

print("\n🎮 Testing the trained AI...")

# Run a test game with no exploration (always use best known action)
test_state, _ = env.reset(seed=42)  # Use seed for reproducible results
test_bins = convert_to_bins(test_state)
test_reward = 0
game_over = False
steps = 0

while not game_over and steps < 500:  # Limit to 500 steps to prevent infinite loops
    # Always choose the best action (no exploration)
    action = np.argmax(q_table[test_bins])
    
    # Take the action
    next_state, reward, terminated, truncated, _ = env.step(action)
    game_over = terminated or truncated
    test_bins = convert_to_bins(next_state)
    test_reward += reward
    steps += 1

print(f"🏆 Test result: {test_reward} points in {steps} steps")

env.close()

# =============================================================================
# FINAL EXPLANATION
# =============================================================================

# The AI learned to balance the pole through these steps:
# 1. 🎯 EXPLORATION: Initially tried random actions to discover what works
# 2. 🧠 MEMORY: Stored the results of each action in each situation
# 3. 🔄 IMPROVEMENT: Gradually updated its strategy based on experience
# 4. 🎖️ MASTERY: Eventually learned to consistently keep the pole balanced

# KEY CONCEPTS:
# - Q-Table: The AI's memory of which actions work best in each situation
# - Exploration vs Exploitation: Balance between trying new things and using known strategies
# - Reward Signal: The AI learned that longer balance times = better performance
# - Discretization: Simplified the complex problem into manageable categories

# The final result shows how reinforcement learning can solve control problems
# through trial and error, just like how humans learn complex skills!
