# -------------------------
# 1. Import Libraries
# -------------------------
import gymnasium as gym
import matplotlib.pyplot as plt

# -------------------------
# 2. Environment Setup
# -------------------------
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
state, info = env.reset(seed=42)

# -------------------------
# 3. Render Function
# -------------------------
def render():
    frame = env.render()
    plt.imshow(frame)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.8)  # slower so you see steps clearly
    plt.clf()

# -------------------------
# 4. Define Action Sequence
# -------------------------
# 0 = Left
# 1 = Down
# 2 = Right
# 3 = Up

actions = [1, 1, 2, 2, 1, 2]  

# -------------------------
# 5. Run Actions
# -------------------------
terminated, truncated = False, False

for action in actions:
    if terminated or truncated:
        break

    state, reward, terminated, truncated, info = env.step(action)
    render()

    print(f"Action: {action}, State: {state}, Reward: {reward}")

    if terminated:
        print("🎉 You reached the goal!")
    elif truncated:
        print("⛔ Episode truncated (fell in hole or max steps).")

# -------------------------
# 6. Cleanup
# -------------------------
env.close()
plt.close()
print("Episode finished.")
