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
state, info = env.reset(seed=42) # Reset the environment
print("Initial State:", state)

# -------------------------
# 2. Render Function (Matplotlib)
# -------------------------
def render(env):
    frame = env.render()             # returns RGB array
    plt.imshow(frame)                # show frame
    plt.axis("off")                  # hide axis
    plt.show(block=False)            # non-blocking display
    plt.pause(0.05)                  # small pause for animation effect
    plt.clf()                        # clear figure for next frame

# -------------------------
# 3. Run One Episode
# -------------------------
terminated, truncated = False, False

while not (terminated or truncated):
    action = 1    # Always move right (try 0 for left)
    state, reward, terminated, truncated, info = env.step(action)

    render(env)   # show frame with matplotlib

    # Debug info
    print(f"State: {state}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

# -------------------------
# 4. Cleanup
# -------------------------
env.close()
plt.close()
print("Episode finished.")



'''
------------------------------------------------------------------------------------------
⭐Render Mode => rgb_array
- The environment returns a NumPy array of shape (H, W, 3) (RGB pixels).
- Nothing is shown on screen automatically.
- You need to manually display it using something like:
    - matplotlib.pyplot.imshow()
    - cv2.imshow() (OpenCV)
- Useful for:
    - Training agents (recording frames to make GIFs or videos).
    - Running environments on headless servers (no display window).
    - Collecting data for computer vision models.

------------------------------------------------------------------------------------------
⭐Render Mode => human
- The environment opens a window and shows the simulation in real-time.
- You don’t get any image array back — it’s purely for visualization.
- Easier when you just want to watch the agent play.
- Not suitable for training pipelines because you can’t save/process the frames directly.
------------------------------------------------------------------------------------------
"""
⚖️ Comparison of render modes

+------------------------+----------------------+------------------------+
| Feature                | rgb_array            | human                  |
+------------------------+----------------------+------------------------+
| Returns image (NumPy)  | ✅ Yes    ----------| ❌ No                  |
| Opens a live window    | ❌ No     ----------| ✅ Yes                 |
| Can record / save frames| ✅ Yes   ----------| ❌ No                  |
| Good for training logs | ✅ Yes    ----------| ❌ No                  |
| Good for just watching | ⚠️ Needs extra code | ✅ Yes (automatic)     |
+------------------------+---------------------+-------------------------+
"""

'''