'''
In the classic CartPole environment used in reinforcement learning (like CartPole-v1 from OpenAI Gym or Gymnasium), 
there are two discrete actions available to the agent:
    - Action = 0 → Push the cart to the left
    - Action = 1 → Push the cart to the right
'''
import gymnasium as gym
import matplotlib.pyplot as plt

# -------------------------
# 1. Environment Setup
# -------------------------
env = gym.make('CartPole-v1', render_mode='rgb_array')
state, info = env.reset(seed=42)
print("Initial State:", state)

# -------------------------
# 2. Render Function
# -------------------------
def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.axis("off")  # hide axis for cleaner visualization
    plt.show(block=False)  
    plt.pause(0.5)   # small pause for animation effect
    plt.clf()        # clear frame for next render

# -------------------------
# 3. Run Episode
# -------------------------
terminated = False
truncated = False

while not (terminated or truncated):
    action = 1    # Always move right
    # action = 0  # Always move left
    state, reward, terminated, truncated, info = env.step(action)
    render()

    # Debug info
    print(f"State: {state}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

# -------------------------
# 4. Cleanup
# -------------------------
env.close()
plt.close()
print("Episode finished.")
