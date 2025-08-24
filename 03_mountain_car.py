import gymnasium as gym
import matplotlib.pyplot as plt

# -------------------------
# 1. Environment Setup
# -------------------------
env = gym.make('MountainCar-v0', render_mode='rgb_array')
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
    plt.pause(0.05)   # smaller pause for smoother animation
    plt.clf()         # clear frame for next render

# -------------------------
# 3. Run Episode
# -------------------------
terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()  # random action (0=left, 1=neutral, 2=right)
    state, reward, terminated, truncated, info = env.step(action)
    render()

    # Optional debug info
    print(f"State: {state}, Reward: {reward}, Done: {terminated}, Truncated: {truncated}")

# -------------------------
# 4. Cleanup
# -------------------------
env.close()
plt.close()
print("Episode finished.")
