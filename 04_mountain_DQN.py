"""
Deep Q-Network (DQN) for MountainCar Environment - IMPROVED VERSION
===================================================================
This code trains an AI agent to drive a car up a hill using Deep Q-Learning.
The car is underpowered, so the agent must learn to build momentum by going
back and forth before reaching the flag at the top.

"""

# Import necessary libraries
import gymnasium as gym           # OpenAI Gymnasium for MountainCar environment
import torch                      # PyTorch for building neural networks
import torch.nn as nn             # Neural network layers
import torch.optim as optim       # Optimizers for training
import random                     # For random action selection
import numpy as np                # For numerical operations
import matplotlib.pyplot as plt   # For plotting and visualization
from collections import deque     # For replay buffer (memory)

# -------------------------
# 1. Environment Setup
# -------------------------
env = gym.make("MountainCar-v0", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]  # 2 values: car position, velocity
n_actions = env.action_space.n              # 3 actions: push left (0), no push (1), push right (2)

print("State Dimension:", state_dim)
print("Number of Actions:", n_actions)

# -------------------------
# 2. Improved Q-Network (Larger)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        # Larger neural network with more capacity
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),    # Increased from 64
            nn.ReLU(),
            nn.Linear(128, 128),          # Added more neurons
            nn.ReLU(),
            nn.Linear(128, 64),           # Extra layer
            nn.ReLU(),
            nn.Linear(64, n_actions)      # Output: Q-value for each action
        )
    def forward(self, x):
        return self.net(x)

# Main network (trainable) and target network (for stable targets)
q_net = QNetwork(state_dim, n_actions)
target_net = QNetwork(state_dim, n_actions)
target_net.load_state_dict(q_net.state_dict())  # Copy initial weights

# Optimizer with higher learning rate
optimizer = optim.Adam(q_net.parameters(), lr=3e-3)  # Increased from 1e-3
loss_fn = nn.MSELoss()

# -------------------------
# 3. Improved Hyperparameters
# -------------------------
gamma = 0.99                  # Discount factor (importance of future rewards)
epsilon = 1.0                 # Exploration rate (start fully random)
epsilon_min = 0.01            # Lower minimum exploration (was 0.05)
epsilon_decay = 0.998         # Slower epsilon decay (was 0.995)
batch_size = 64               # Training batch size
memory = deque(maxlen=20000)  # Larger replay buffer (was 10000)
episodes = 1000               # More training episodes (was 500)
sync_rate = 10                # Update target network more frequently (was 20)
all_rewards = []              # Track rewards
success_count = 0             # Count successful episodes

# -------------------------
# 4. Reward Shaping Function
# -------------------------
def get_reward(state, action, next_state, original_reward, done):
    """
    Better reward shaping to encourage momentum building and progress
    """
    position, velocity = next_state[0], next_state[1]
    
    # Original reward
    reward = original_reward
    
    # Bonus for reaching the flag (success!)
    if done and position >= 0.5:
        reward += 100  # Big bonus for success!
        
    # Encourage building momentum (higher absolute velocity is good)
    momentum_bonus = abs(velocity) * 10
    reward += momentum_bonus
    
    # Encourage moving right when on the left side
    if position < 0 and velocity > 0:
        reward += 5
        
    # Encourage moving left when on the right side (to build momentum)
    if position > -0.3 and velocity < 0:
        reward += 5
        
    return reward

# -------------------------
# 5. Training Loop with Improvements
# -------------------------
print("Starting training...")
for ep in range(episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            with torch.no_grad():
                action = q_net(state).argmax().item()  # Exploit best action

        # Take action
        next_state, original_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        
        # Apply reward shaping
        shaped_reward = get_reward(state.numpy(), action, next_state, original_reward, done)
        
        # Store experience in memory
        memory.append((state, action, shaped_reward, next_state_tensor, done))

        # Update state and reward
        state = next_state_tensor
        total_reward += original_reward  # Track original reward for comparison
        step_count += 1

        # Learning step (train more frequently)
        if len(memory) >= batch_size and step_count % 4 == 0:  # Train every 4 steps
            batch = random.sample(memory, batch_size)
            s, a, r, s2, d = zip(*batch)

            s = torch.stack(s)
            a = torch.tensor(a)
            r = torch.tensor(r, dtype=torch.float32)
            s2 = torch.stack(s2)
            d = torch.tensor(d, dtype=torch.float32)

            # Current Q-values
            q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

            # Target Q-values (using target network)
            with torch.no_grad():
                max_next_q = target_net(s2).max(1)[0]
                target = r + gamma * max_next_q * (1 - d)

            # Loss and backpropagation
            loss = loss_fn(q_vals, target)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10)
            optimizer.step()

    # Count successful episodes
    if total_reward > -200:
        success_count += 1

    # Update epsilon (slower decay)
    if ep > 100:  # Start decay after some exploration
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Sync target network more frequently
    if ep % sync_rate == 0:
        target_net.load_state_dict(q_net.state_dict())

    # Track progress
    all_rewards.append(total_reward)
    
    # Print progress every 50 episodes
    if (ep + 1) % 50 == 0:
        avg_reward = np.mean(all_rewards[-50:])
        print(f"Episode {ep+1}/{episodes} | Avg Reward (last 50): {avg_reward:.1f} | "
              f"Current Reward: {total_reward} | Epsilon: {epsilon:.3f} | "
              f"Successes: {success_count}")

print(f"\nTraining completed! Total successes: {success_count}/{episodes}")

# -------------------------
# 6. Plot Training Progress
# -------------------------
plt.figure(figsize=(12, 4))

# Plot 1: All rewards
plt.subplot(1, 2, 1)
plt.plot(all_rewards, alpha=0.6, label='Episode Reward')
# Add moving average
window = 50
if len(all_rewards) >= window:
    moving_avg = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(all_rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window})')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("MountainCar Training Progress")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Success rate over time
plt.subplot(1, 2, 2)
success_rate = []
window = 100
for i in range(len(all_rewards)):
    if i >= window:
        recent_rewards = all_rewards[i-window:i]
        success_rate.append(sum(1 for r in recent_rewards if r > -200) / window * 100)
    else:
        success_rate.append(0)

plt.plot(success_rate)
plt.xlabel("Episode")
plt.ylabel("Success Rate (%)")
plt.title(f"Success Rate (last {window} episodes)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------
# 7. Test Trained Agent with Visualization
# -------------------------
print("\nTesting trained agent...")
state, _ = env.reset(seed=42)
state = torch.tensor(state, dtype=torch.float32)
done = False
total_reward = 0
positions = []
velocities = []

# Test without visualization first
test_episodes = 5
test_rewards = []

for test_ep in range(test_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    test_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            action = q_net(state).argmax().item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = torch.tensor(next_state, dtype=torch.float32)
        test_reward += reward
    
    test_rewards.append(test_reward)
    print(f"Test Episode {test_ep+1}: Reward = {test_reward}")

avg_test_reward = np.mean(test_rewards)
print(f"\n✅ Average Test Reward: {avg_test_reward:.1f}")

if avg_test_reward > -200:
    print("🎉 SUCCESS! Agent learned to reach the flag!")
else:
    print("❌ Agent still struggling. Consider training longer or adjusting hyperparameters.")

env.close()