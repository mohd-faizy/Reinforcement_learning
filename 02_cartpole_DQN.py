"""
Deep Q-Network (DQN) for CartPole Environment
===========================================
This code trains an AI agent to balance a pole on a cart using Deep Q-Learning.
The agent learns by trial and error, gradually improving its performance.
"""

# Import necessary libraries
import gymnasium as gym           # For the CartPole game environment
import torch                      # PyTorch for neural networks
import torch.nn as nn             # Neural network layers
import torch.optim as optim       # Optimizers for training
import random                     # For random number generation
import matplotlib.pyplot as plt   
from collections import deque     # For efficient memory storage

# -----------------
# 1. Environment Setup
# -----------------
# Create the CartPole game environment
# CartPole-v1: A pole is attached to a cart, goal is to keep pole upright
# render_mode="rgb_array": Allows us to get visual frames for display
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Get environment dimensions
state_dim = env.observation_space.shape[0]  # 4 values: cart position, cart velocity, pole angle, pole angular velocity
n_actions = env.action_space.n              # 2 actions: move left (0) or right (1)

# -----------------
# 2. Q-Network (Neural Network)
# -----------------
# The Q-Network estimates the "quality" (Q-value) of each action in each state
# Higher Q-value = better action to take
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        # Create a simple neural network:
        # Input layer: 4 neurons (state dimensions)
        # Hidden layer: 64 neurons with ReLU activation
        # Output layer: 2 neurons (one for each action)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),  # Input to hidden layer
            nn.ReLU(),                 # Activation function (adds non-linearity)
            nn.Linear(64, n_actions)   # Hidden to output layer
        )
    
    def forward(self, x):
        # Forward pass: input state -> Q-values for each action
        return self.net(x)

# Create two identical networks:
q_net = QNetwork(state_dim, n_actions)      # Main network (gets trained)
target_net = QNetwork(state_dim, n_actions) # Target network (provides stable targets)

# Copy weights from main network to target network
target_net.load_state_dict(q_net.state_dict())

# Set up training components
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)  # Adam optimizer for updating network weights
loss_fn = nn.MSELoss()                               # Mean Squared Error loss function

# -----------------
# 3. Hyperparameters (Training Settings)
# -----------------
gamma = 0.99                    # Discount factor: how much we care about future rewards (0.99 = care a lot)
epsilon = 1.0                   # Exploration rate: probability of taking random action (starts high)
epsilon_min = 0.05              # Minimum exploration rate (always explore at least 5%)
epsilon_decay = 0.995           # How fast exploration decreases over time
batch_size = 64                 # Number of experiences to train on at once
memory = deque(maxlen=10000)    # Replay buffer: stores past experiences (max 10,000)
episodes = 200                  # Number of games to play during training
sync_rate = 10                  # How often to update target network (every 10 episodes)
all_rewards = []                # List to store rewards from each episode

# -----------------
# 4. Training Loop (The AI learns here!)
# -----------------
for ep in range(episodes):
    # Start a new episode (game)
    state, _ = env.reset()  # Reset environment, get initial state
    state = torch.tensor(state, dtype=torch.float32)  # Convert to PyTorch tensor
    total_reward = 0        # Track total reward for this episode
    done = False            # Flag to know when episode ends
    
    # Play the game until it ends
    while not done:
        # DECISION MAKING: Should we explore (random) or exploit (use what we learned)?
        # Epsilon-greedy strategy: 
        if random.random() < epsilon:
            # EXPLORE: Take a random action to discover new strategies
            action = env.action_space.sample()
        else:
            # EXPLOIT: Use our neural network to pick the best action
            with torch.no_grad():  # Don't compute gradients (saves memory/computation)
                action = q_net(state).argmax().item()  # Get action with highest Q-value
        
        # TAKE ACTION: Execute the chosen action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Episode ends if pole falls or time limit reached
        next_state = torch.tensor(next_state, dtype=torch.float32)
        
        # STORE EXPERIENCE: Save this experience for learning later
        # Format: (current_state, action, reward, next_state, done)
        memory.append((state, action, reward, next_state, done))
        
        # Move to next state and update total reward
        state = next_state
        total_reward += reward
        
        # LEARNING: Train the network if we have enough experiences
        if len(memory) >= batch_size:
            # Sample a random batch of past experiences
            batch = random.sample(memory, batch_size)
            # Unpack the batch into separate arrays
            s, a, r, s2, d = zip(*batch)  # states, actions, rewards, next_states, done_flags
            
            # Convert to PyTorch tensors for neural network processing
            s = torch.stack(s)                              # Current states
            a = torch.tensor(a)                            # Actions taken
            r = torch.tensor(r, dtype=torch.float32)       # Rewards received
            s2 = torch.stack(s2)                           # Next states
            d = torch.tensor(d, dtype=torch.float32)       # Done flags (1 if episode ended, 0 otherwise)
            
            # COMPUTE CURRENT Q-VALUES: What did our network predict?
            q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze()  # Q-values for actions we actually took
            
            # COMPUTE TARGET Q-VALUES: What should our network have predicted?
            with torch.no_grad():  # Don't compute gradients for target computation
                # Get maximum Q-value for next states using target network
                max_next_q = target_net(s2).max(1)[0]
                # Bellman equation: Q(s,a) = reward + gamma * max_Q(s',a') * (1 - done)
                # If episode ended (done=1), there's no future reward, so multiply by (1-1)=0
                target = r + gamma * max_next_q * (1 - d)
            
            # COMPUTE LOSS: How wrong were our predictions?
            loss = loss_fn(q_vals, target)  # Mean squared error between predicted and target Q-values
            
            # BACKPROPAGATION: Update network weights to reduce loss
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights
    
    # DECREASE EXPLORATION: As agent learns, explore less and exploit more
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # UPDATE TARGET NETWORK: Copy main network weights to target network periodically
    # This provides more stable training targets
    if ep % sync_rate == 0:
        target_net.load_state_dict(q_net.state_dict())
    
    # TRACK PROGRESS: Store episode reward and print progress
    all_rewards.append(total_reward)
    print(f"Episode {ep+1} | Reward: {total_reward} | Epsilon: {epsilon:.2f}")

# -----------------
# 5. Plot Training Results
# -----------------
# Visualize how the agent's performance improved over time
plt.plot(all_rewards)
plt.xlabel("Episode")           # X-axis: episode number
plt.ylabel("Reward")            # Y-axis: total reward earned
plt.title("Training Progress")  # Title
plt.show()

# -----------------
# 6. Test the Trained Agent + Visual Display
# -----------------
# Now let's see how well our trained agent performs!
state, _ = env.reset(seed=42)  # Reset environment with fixed seed for reproducibility
state = torch.tensor(state, dtype=torch.float32)
done = False
total_reward = 0

# Set up real-time visualization
plt.ion()  # Turn on interactive mode for real-time plotting
fig, ax = plt.subplots()

# Run the trained agent and display the game
while not done:
    # Use trained network to select action (no more random exploration!)
    with torch.no_grad():
        action = q_net(state).argmax().item()  # Always pick best action
    
    # Take action and get results
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = torch.tensor(next_state, dtype=torch.float32)
    total_reward += reward
    
    # VISUAL DISPLAY: Show the game in real-time
    frame = env.render()  # Get current game frame as image
    ax.imshow(frame)      # Display the frame
    plt.axis("off")       # Hide axes for cleaner display
    plt.pause(0.01)       # Brief pause to create animation effect
    ax.clear()            # Clear previous frame

plt.ioff()  # Turn off interactive mode
print(f"✅ Test Reward: {total_reward}")  # Print final test performance
env.close()  # Clean up environment

"""
HOW IT WORKS - SIMPLE EXPLANATION:
1. The AI starts knowing nothing about the game
2. It tries random actions and sees what happens (exploration)
3. It remembers good and bad experiences in its "memory"
4. It uses a neural network to learn from these experiences
5. Over time, it explores less and uses what it learned more (exploitation)
6. The result: An AI that can balance the pole much better than random!

KEY CONCEPTS:
- Q-Learning: Learning the "quality" of actions in different situations
- Exploration vs Exploitation: Trying new things vs using what you know
- Neural Network: The "brain" that learns to evaluate actions
- Experience Replay: Learning from past experiences, not just current one
- Target Network: Provides stable learning targets
"""