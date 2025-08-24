
# ========================================================================================
# REINFORCEMENT LEARNING: Q-LEARNING ON FROZEN LAKE
# ========================================================================================
# This code teaches an AI agent to navigate a frozen lake using Q-Learning!
#
# THE FROZEN LAKE PROBLEM:
# - Agent starts at top-left corner (S = Start)
# - Goal is to reach bottom-right corner (G = Goal) 
# - There are holes (H) that end the game if stepped on
# - Safe frozen tiles (F) are okay to walk on
#
# Grid looks like this:
# ------ S F F F ------
# ------ F H F H ------  
# ------ F F F H ------
# ------ H F F G ------

# ========================================================================================
# KEY CONCEPTS:
# ========================================================================================
# 
# 🎯 STATE: Where the agent is (position on the grid)
# 🎮 ACTION: What the agent can do (move left, down, right, up)
# 🏆 REWARD: Feedback from environment (+1 for goal, 0 elsewhere, game ends if hole)
# 📚 Q-TABLE: Agent's memory of which actions work best in each state
# 🎲 EXPLORATION: Trying random actions to discover new strategies
# 🧠 EXPLOITATION: Using learned knowledge to make best decisions
# 📈 LEARNING: Updating Q-values based on experience using the Q-learning formula
# 
# The agent starts knowing nothing and gradually learns through trial and error!
# ========================================================================================


# -------------------------
# 1. Import Libraries
# -------------------------
import gymnasium as gym      
import matplotlib.pyplot as plt  # For displaying the game visually
import numpy as np          # For mathematical operations and arrays

# -------------------------
# 2. Environment Setup
# -------------------------
# Create the Frozen Lake game environment
# - "FrozenLake-v1": The specific game we want to play
# - is_slippery=False: Makes movement predictable (no random sliding)
# - render_mode="rgb_array": Allows us to visualize the game
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Get important information about our game world
n_states = env.observation_space.n    # How many different positions exist (16 squares)
n_actions = env.action_space.n        # How many actions we can take (4: up, down, left, right)

print(f"🎮 Game Environment Created!")
print(f"📍 Number of states (positions): {n_states}")
print(f"🎯 Number of actions (moves): {n_actions}")
print(f"🕹️  Actions: 0=Left, 1=Down, 2=Right, 3=Up")

# -------------------------
# 3. Visualization Function
# -------------------------
def render():
    """
    This function shows us what the game looks like visually
    - Gets the current game state as an image
    - Displays it using matplotlib
    - Pauses briefly so we can see each move
    """
    frame = env.render()        # Get current game state as image
    plt.imshow(frame)          # Display the image
    plt.axis("off")            # Hide axis numbers
    plt.show(block=False)      # Show without stopping the program
    plt.pause(0.4)             # Wait 0.4 seconds so we can see the move
    plt.clf()                  # Clear the plot for next frame

# -------------------------
# 4. Q-Learning Setup
# -------------------------
# Q-TABLE: The "brain" of our agent
# - Think of it as a cheat sheet with 16 rows (states) and 4 columns (actions)
# - Each cell contains a "Q-value" = how good that action is from that state
# - Higher Q-value = better action
# - Starts with all zeros (agent knows nothing initially)
Q = np.zeros((n_states, n_actions))

print(f"\n🧠 Q-Table initialized with shape: {Q.shape}")
print("This table will store the agent's learned knowledge!")

# HYPERPARAMETERS: These control how the agent learns
alpha = 0.8       # LEARNING RATE: How much new info overrides old info (0-1)
                  # Higher = learns faster but might be unstable
                  
gamma = 0.95      # DISCOUNT FACTOR: How much future rewards matter (0-1)
                  # Higher = cares more about long-term rewards
                  
epsilon = 0.3     # EXPLORATION RATE: How often to try random actions (0-1)
                  # Higher = more exploration, lower = more exploitation
                  
max_steps = 100   # Maximum steps per episode (prevents infinite loops)

print(f"\n⚙️ Learning Parameters:")
print(f"📚 Learning Rate (alpha): {alpha} - How fast we learn")
print(f"🔮 Discount Factor (gamma): {gamma} - How much we value future rewards") 
print(f"🎲 Exploration Rate (epsilon): {epsilon} - How often we explore randomly")

# Training tracking variables
episode = 0           # Count of training episodes
goal_reached = False  # Flag to stop training when agent succeeds

# -------------------------
# 5. Training Loop (Learning Phase)
# -------------------------
print(f"\n🚀 Starting Training! The agent will learn through trial and error...")

while not goal_reached:
    episode += 1
    
    # RESET ENVIRONMENT: Start a new episode
    state, _ = env.reset()           # Agent starts at position 0 (top-left)
    terminated, truncated = False, False  # Episode hasn't ended yet

    print(f"\n🎬 Episode {episode} starting at state {state}...")

    # RUN ONE EPISODE (sequence of moves until game ends)
    for step in range(max_steps):
        
        # EPSILON-GREEDY ACTION SELECTION
        # This balances exploration (trying new things) vs exploitation (using what we know)
        if np.random.rand() < epsilon:
            # EXPLORE: Choose random action (helps discover new strategies)
            action = env.action_space.sample()
            action_type = "🎲 Random"
        else:
            # EXPLOIT: Choose action with highest Q-value (use learned knowledge)
            action = np.argmax(Q[state, :])
            action_type = "🧠 Learned"

        # TAKE THE ACTION and see what happens
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Q-LEARNING UPDATE (This is where the magic happens!)
        # The agent updates its knowledge based on what just happened
        # 
        # Q-Learning Formula Breakdown:
        # old_q_value = Q[state, action]                    # What we thought before
        # max_future_q = np.max(Q[next_state, :])          # Best possible future value
        # target = reward + gamma * max_future_q            # What we should have thought
        # new_q_value = old_q_value + alpha * (target - old_q_value)  # Updated knowledge
        
        old_q_value = Q[state, action]
        max_future_q = np.max(Q[next_state, :])
        target_q_value = reward + gamma * max_future_q
        
        # Update the Q-table with new knowledge
        Q[state, action] = old_q_value + alpha * (target_q_value - old_q_value)

        # VISUALIZATION AND LOGGING
        render()  # Show the current game state
        
        # Convert action number to direction name for clarity
        action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
        print(f"Step {step}: {action_type} action {action} ({action_names[action]})")
        print(f"From state {state} → state {next_state}, Reward: {reward}")
        print(f"Q-value updated: {old_q_value:.3f} → {Q[state, action]:.3f}")

        # MOVE TO NEXT STATE
        state = next_state

        # CHECK IF EPISODE IS OVER
        if terminated or truncated:
            if reward == 1:
                # SUCCESS! Agent reached the goal
                goal_reached = True
                print(f"🎉 SUCCESS! Goal reached in episode {episode}, step {step}!")
                print(f"🏆 Agent has learned how to solve the problem!")
            else:
                # FAILURE: Agent fell in a hole or ran out of time
                print("💀 FAILURE: Fell into a hole or episode ended without reaching goal.")
            break

    # DECAY EXPLORATION: Gradually reduce random actions as agent learns
    # This makes the agent more likely to use its learned knowledge over time
    old_epsilon = epsilon
    epsilon = max(0.01, epsilon * 0.995)  # Multiply by 0.995, but never go below 0.01
    
    if old_epsilon != epsilon:
        print(f"🎲 Exploration rate reduced: {old_epsilon:.3f} → {epsilon:.3f}")

print(f"\n🚀 Training Complete! Agent learned in {episode} episodes!")

# -------------------------
# 6. Testing the Learned Policy
# -------------------------
print(f"\n" + "="*60)
print(f"🎯 TESTING PHASE: Let's see how well our agent learned!")
print(f"   The agent will now use only its learned knowledge (no random moves)")
print(f"="*60)

# Reset environment for testing
state, _ = env.reset(seed=42)  # Use seed for reproducible results
terminated, truncated = False, False
test_step = 0

print(f"\n🎯 Running final learned policy from state {state}...")

while not (terminated or truncated):
    test_step += 1
    
    # GREEDY POLICY: Always choose the action with highest Q-value
    action = np.argmax(Q[state, :])
    q_values_for_state = Q[state, :]
    
    # Take the action
    state, reward, terminated, truncated, _ = env.step(action)
    
    # Show the move
    render()
    
    action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    print(f"🎯 Test Step {test_step}: Chose action {action} ({action_names[action]})")
    print(f"   Q-values for previous state: {q_values_for_state}")
    print(f"   Moved to state {state}, Reward: {reward}")

    # Check results
    if terminated and reward == 1:
        print(f"\n🎉 SUCCESS! Agent successfully reached the goal using learned policy!")
        print(f"🏆 The agent solved the problem in {test_step} steps!")
    elif terminated or truncated:
        print(f"\n⛔ FAILURE: Episode ended without reaching goal.")
        print(f"🤔 The agent might need more training...")

# -------------------------
# 7. Show Final Q-Table (The Agent's "Knowledge")
# -------------------------
print(f"\n" + "="*60)
print(f"🧠 FINAL Q-TABLE (Agent's Learned Knowledge):")
print(f"="*60)
print("Each row = state (position), Each column = action (Left, Down, Right, Up)")
print("Higher values = better actions\n")

# Print Q-table in a readable format
for state in range(n_states):
    row = state // 4  # Which row in the 4x4 grid
    col = state % 4   # Which column in the 4x4 grid
    print(f"State {state:2d} (row {row}, col {col}): {Q[state, :]}")

# -------------------------
# 8. Cleanup
# -------------------------
env.close()      # Close the game environment
plt.close()      # Close any open plots
print(f"\n✅ Program finished successfully!")