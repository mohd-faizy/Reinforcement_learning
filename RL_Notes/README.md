# Reinforcement Learning Notes✨

A comprehensive reference summarizing the most important concepts and algorithms in reinforcement learning (RL), designed to minimize the need to constantly refer back to original notebooks.

## Table of Contents

- [Core Concepts](#core-concepts)
  - [Agent-Environment Loop](#agent-environment-loop)
  - [Markov Decision Process (MDP)](#markov-decision-process-mdp)
  - [Value Functions & Bellman Equations](#value-functions--bellman-equations)
  - [Exploration vs. Exploitation](#exploration-vs-exploitation)
- [Basic / Tabular Methods](#basic--tabular-methods)
  - [Simple Exploration Bot](#simple-exploration-bot)
  - [Q-Learning](#q-learning)
  - [SARSA](#sarsa)
  - [Expected SARSA](#expected-sarsa)
  - [Dyna-Q](#dyna-q)
- [Policy Gradient Methods](#policy-gradient-methods)
  - [REINFORCE (Monte Carlo Policy Gradient)](#reinforce-monte-carlo-policy-gradient)
  - [Trust Region Policy Optimization (TRPO)](#trust-region-policy-optimization-trpo)
- [Actor-Critic Methods](#actor-critic-methods)
  - [Advantage Actor-Critic (A2C)](#advantage-actor-critic-a2c)
  - [Asynchronous Advantage Actor-Critic (A3C)](#asynchronous-advantage-actor-critic-a3c)
  - [Deep Deterministic Policy Gradient (DDPG)](#deep-deterministic-policy-gradient-ddpg)
  - [Soft Actor-Critic (SAC)](#soft-actor-critic-sac)
  - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
- [Value-Based Deep Methods](#value-based-deep-methods)
  - [Deep Q-Networks (DQN)](#deep-q-networks-dqn)
- [Multi-Agent RL (MARL)](#multi-agent-rl-marl)
  - [Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](#multi-agent-deep-deterministic-policy-gradient-maddpg)
  - [QMIX (Monotonic Value Function Factorization)](#qmix-monotonic-value-function-factorization)
- [Hierarchical RL (HRL)](#hierarchical-rl-hrl)
  - [Hierarchical Actor-Critic (HAC)](#hierarchical-actor-critic-hac)
- [Planning & Model-Based Methods](#planning--model-based-methods)
  - [Monte Carlo Tree Search (MCTS)](#monte-carlo-tree-search-mcts)
  - [PlaNet (Deep Planning Network)](#planet-deep-planning-network)
- [Quick Reference Tables](#quick-reference-tables)

---

## Core Concepts

### Agent-Environment Loop

The fundamental interaction cycle in RL follows a systematic pattern:

1. **Observation**: Agent observes state $s_t$
2. **Action Selection**: Agent selects action $a_t$ based on policy $\pi(a_t|s_t)$
3. **Environment Response**: Environment transitions to next state $s_{t+1}$
4. **Reward Signal**: Environment provides reward $r_t$
5. **Learning Update**: Agent updates policy/values based on $(s_t, a_t, r_t, s_{t+1})$

### Markov Decision Process (MDP)

Formal framework for RL problems, defined by the tuple $(S, A, P, R, \gamma)$:

- **$S$**: Set of states
- **$A$**: Set of actions  
- **$P(s'|s, a)$**: Transition probability function
- **$R(s, a, s')$**: Reward function
- **$\gamma$**: Discount factor ($0 \le \gamma \le 1$)

### Value Functions & Bellman Equations

#### State-Value Function
Expected return starting from state $s$ and following policy $\pi$:

```math
V^\pi(s) = \mathbb{E}_\pi \left[ G_t | S_t=s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} | S_t=s \right]
```

#### Action-Value Function
Expected return starting from state $s$, taking action $a$, and following policy $\pi$:

```math
Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t | S_t=s, A_t=a \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} | S_t=s, A_t=a \right]
```

#### Bellman Equations

**Bellman Expectation Equation for $V^\pi$:**
```math
V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s', r | s, a) \left[r + \gamma V^\pi(s')\right] 
```

**Bellman Optimality Equation for $Q^*$ (used by Q-Learning):**
```math
Q^*(s, a) = \sum_{s', r} P(s', r | s, a) \left[r + \gamma \max_{a'} Q^*(s', a')\right]
```

### Exploration vs. Exploitation

- **Exploration**: Trying new actions to discover better rewards
- **Exploitation**: Choosing the action currently known to yield the best expected reward
- **$\epsilon$-Greedy Strategy**: 
  - With probability $\epsilon$: explore (random action)
  - With probability $1-\epsilon$: exploit (greedy action)
  - $\epsilon$ typically decays over time

---

## Basic / Tabular Methods

#### Core Concept
Demonstrates basic agent-environment loop. Remembers immediate rewards for state-action pairs using epsilon-greedy policy based on average immediate rewards. **Note: Does not perform true RL value learning.**

#### Mathematical Formulation
No Bellman updates. Policy based on:
```math
\text{AvgR}(s, a) = \frac{\sum \text{rewards observed after } (s, a)}{\text{count of } (s, a)}
```

#### Algorithm Steps
1. Initialize memory `mem[s][a] -> [rewards]`
2. For each episode:
   - Reset environment to get initial state `s`
   - For each step:
     - Choose action `a` using $\epsilon$-greedy on `AvgR(s, a)`
     - Execute action `a`, observe reward `r` and next state `s'`
     - Store reward `r` in `mem[s][a]`
     - Update state: `s = s'`

#### Code Implementation
```python
# Action selection based on average immediate reward
avg_rewards = []
for a in range(n_actions):
    rewards = memory[state][a]
    avg_rewards.append(np.mean(rewards) if rewards else 0)
best_action = np.random.choice(np.where(avg_rewards == np.max(avg_rewards))[0])

# Memory update
memory[state][action].append(reward)
```

#### Key Parameters
- `epsilon`: Exploration rate
- `epsilon_decay`: Rate of exploration decay

#### Advantages & Limitations
- **Pros**: Simple illustration of interaction loop and memory
- **Cons**: 
  - Does not learn long-term values (only immediate rewards)
  - Not true RL algorithm
  - Inefficient memory usage
- **Use Cases**: Educational demonstration of basic agent structure

### Q-Learning

#### Core Concept
Learns optimal action-value function ($Q^*$) using **off-policy** Temporal Difference (TD) updates.

#### Mathematical Formulation
Bellman Optimality update rule:
```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]
```

Where:
- $\alpha$: Learning rate
- $\gamma$: Discount factor  
- $\max_{a'} Q(s_{t+1}, a')$: Maximum Q-value in next state (greedy estimate)

#### Algorithm Steps
1. Initialize Q-table `Q(s, a)` to zeros
2. For each episode:
   - Initialize state `s`
   - For each step:
     - Choose action `a` from `s` using policy derived from Q (e.g., $\epsilon$-greedy)
     - Execute action `a`, observe reward `r` and next state `s'`
     - Update `Q(s, a)` using Q-learning rule
     - Update state: `s = s'`

#### Code Implementation
```python
# Q-Learning update
current_q = q_table[state][action]
max_next_q = max(q_table[next_state].values()) if next_state in q_table else 0.0
td_target = reward + gamma * max_next_q
td_error = td_target - current_q
q_table[state][action] += alpha * td_error
```

#### Key Parameters
- `alpha`: Learning rate
- `gamma`: Discount factor
- `epsilon`: Exploration rate
- `epsilon_decay`: Exploration decay rate

#### Advantages & Limitations
- **Pros**: 
  - Off-policy learning (can learn optimal policy while exploring)
  - Simple concept with theoretical guarantees
  - Guaranteed convergence under certain conditions
- **Cons**: 
  - Tabular form doesn't scale to large state spaces
  - Can suffer from maximization bias (addressed by Double Q-learning)
- **Common Pitfalls**: Tuning learning rate and exploration parameters
- **Use Cases**: Small, discrete state/action spaces, foundational understanding

### SARSA

#### Core Concept
Learns action-value function ($Q^\pi$) for the policy currently being followed using **on-policy** TD updates.

#### Mathematical Formulation
Update uses the next action chosen by the current policy:
```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
```

Where $a_{t+1}$ is the action chosen in state $s_{t+1}$ by the current policy.

#### Algorithm Steps
1. Initialize Q-table `Q(s, a)`
2. For each episode:
   - Initialize state `s`
   - Choose action `a` from `s` using policy derived from Q
   - For each step:
     - Execute action `a`, observe reward `r` and next state `s'`
     - Choose next action `a'` from `s'` using policy derived from Q
     - Update `Q(s, a)` using `r`, `s'`, and `a'`
     - Update: `s = s'`, `a = a'`

#### Code Implementation
```python
# SARSA update
current_q = q_table[state][action]
next_q = q_table[next_state][next_action]  # Q-value of next action taken
td_target = reward + gamma * next_q
td_error = td_target - current_q
q_table[state][action] += alpha * td_error
```

#### Key Parameters
- `alpha`: Learning rate
- `gamma`: Discount factor
- `epsilon`: Exploration rate
- `epsilon_decay`: Exploration decay rate

#### Advantages & Limitations
- **Pros**: 
  - On-policy learning (learns value of exploration policy)
  - Often more stable/conservative in risky environments than Q-learning
- **Cons**: 
  - Tabular representation limits scalability
  - Can be slower to converge to optimal if exploration persists
  - Sensitive to policy changes
- **Common Pitfalls**: Ensuring next action is chosen correctly before update
- **Use Cases**: When evaluating current policy is important, safer exploration needed

### Expected SARSA

#### Core Concept
Similar to SARSA but updates using expected value over next actions, weighted by policy probabilities. Reduces variance while maintaining **on-policy** nature.

#### Mathematical Formulation
```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \mathbb{E}_{\pi}[Q(s_{t+1}, A')] - Q(s_t, a_t)]
```

Expected value calculation:
```math
\mathbb{E}_{\pi}[Q(s', A')] = \sum_{a'} \pi(a'|s') Q(s', a')
```

For $\epsilon$-greedy policy:
```math
\mathbb{E}_{\pi}[Q(s', A')] = (1 - \epsilon) \max_{a''} Q(s', a'') + \epsilon \frac{\sum_{a'} Q(s', a')}{|\mathcal{A}|}
```

#### Algorithm Steps
1. Initialize Q-table `Q(s, a)`
2. For each episode:
   - Initialize state `s`
   - For each step:
     - Choose action `a` from `s` using policy derived from Q
     - Execute action `a`, observe reward `r` and next state `s'`
     - Calculate expected Q-value $\mathbb{E}[Q(s', A')]$ based on policy in state `s'`
     - Update `Q(s, a)` using `r` and expected Q-value
     - Update state: `s = s'`

#### Code Implementation
```python
# Expected SARSA update (epsilon-greedy)
current_q = q_table[state][action]
if next_state in q_table and q_table[next_state]:
    q_values_next = q_table[next_state]
    max_q_next = max(q_values_next.values())
    num_actions = len(action_space)
    expected_q_next = (1.0 - epsilon) * max_q_next + \
                      (epsilon / num_actions) * sum(q_values_next.values())
else:
    expected_q_next = 0.0
    
td_target = reward + gamma * expected_q_next
td_error = td_target - current_q
q_table[state][action] += alpha * td_error
```

#### Key Parameters
- `alpha`: Learning rate
- `gamma`: Discount factor
- `epsilon`: Exploration rate
- `epsilon_decay`: Exploration decay rate

#### Advantages & Limitations
- **Pros**: 
  - On-policy learning with lower variance than SARSA
  - More stable updates
  - Same computational cost as Q-learning per update
- **Cons**: 
  - Tabular representation limits scalability
  - Slightly more complex update calculation than SARSA
- **Use Cases**: Applications where SARSA is suitable but stability/variance is a concern

### Dyna-Q

#### Core Concept
Integrates **model-free learning** (Q-learning) with **model-based planning**. Learns environment model from real experience and uses it for additional "planning" updates using simulated experience.

#### Mathematical Formulation
- **Direct RL**: Standard Q-learning update on real transition $(s, a, r, s')$
- **Model Learning**: $\text{Model}(s, a) \leftarrow (r, s')$ (deterministic environment)
- **Planning**: For $k$ steps:
  - Sample $(s_p, a_p)$ from previously experienced pairs
  - Get $(r_p, s'_p) = \text{Model}(s_p, a_p)$
  - Apply Q-learning update to $Q(s_p, a_p)$ using $(s_p, a_p, r_p, s'_p)$

#### Algorithm Steps
1. Initialize `Q(s, a)` and `Model(s, a)`
2. For each episode:
   - Initialize state `s`
   - For each step:
     - Choose action `a` using policy based on Q
     - Execute action `a`, observe reward `r` and next state `s'`
     - **Direct RL**: Update `Q(s, a)` with $(s, a, r, s')$
     - **Model Update**: Update `Model(s, a)` with $(r, s')$
     - **Planning**: Repeat `k` times:
       - Sample previously seen $(s_p, a_p)$
       - Get $(r_p, s'_p)$ from `Model(s_p, a_p)`
       - Update `Q(s_p, a_p)` with $(s_p, a_p, r_p, s'_p)$
     - Update state: `s = s'`

#### Code Implementation
```python
# Direct RL Update (Q-Learning)
q_learning_update(q_table, state, action, reward, next_state, ...)

# Model Update
model[(state, action)] = (reward, next_state)
if (state, action) not in observed_pairs:
    observed_pairs.append((state, action))
    
# Planning Step
for _ in range(planning_steps_k):
    if not observed_pairs: 
        break
    s_p, a_p = random.choice(observed_pairs)
    r_p, s_prime_p = model[(s_p, a_p)]
    q_learning_update(q_table, s_p, a_p, r_p, s_prime_p, ...)
```

#### Key Parameters
- `alpha`: Learning rate
- `gamma`: Discount factor
- `epsilon`: Exploration rate
- `k`: Number of planning steps per real step

#### Advantages & Limitations
- **Pros**: 
  - Improved sample efficiency compared to pure Q-learning
  - Simple integration of learning and planning
  - Reuses experience effectively
- **Cons**: 
  - Tabular representation limits scalability
  - Effectiveness depends on model accuracy
  - Assumes deterministic model in simple form
- **Common Pitfalls**: Poor model accuracy leading to suboptimal policy, tuning `k`
- **Use Cases**: Environments where interaction is costly but computation is cheap

---

## Policy Gradient Methods

### REINFORCE (Monte Carlo Policy Gradient)

#### Core Concept
Directly learns parameterized policy $\pi(a|s; \theta)$ by increasing probability of actions that led to high cumulative episode returns. **On-policy**, **Monte Carlo** approach.

#### Mathematical Formulation
Updates policy parameters $\theta$ via gradient ascent on expected return $J(\theta) = \mathbb{E}[G_t]$:

**Policy Gradient:**
```math
\nabla_\theta J(\theta) \approx \sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi(a_t | s_t; \theta)
```

**Loss Function (for minimization):**
```math
L(\theta) = -\sum_{t=0}^{T-1} G_t \log \pi(a_t | s_t; \theta)
```

Where $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$ is the discounted return from step $t$.

#### Algorithm Steps
1. Initialize policy network $\pi(a|s; \theta)$
2. For each episode:
   - Generate trajectory $(s_0, a_0, r_1, s_1, a_1, r_2, ...)$ by sampling actions $a_t \sim \pi(a|s_t; \theta)$
   - Store log probabilities $\log \pi(a_t|s_t; \theta)$ and rewards $r_{t+1}$
   - Calculate discounted returns $G_t$ for all steps $t$
   - Compute loss $L(\theta)$
   - Update $\theta$ using gradient descent on $L(\theta)$

#### Code Implementation
```python
# Calculate returns (backward loop)
returns = []
G = 0.0
for r in reversed(episode_rewards):
    G = r + gamma * G
    returns.insert(0, G)
returns = torch.tensor(returns)

# Standardize returns (optional but recommended)
returns = (returns - returns.mean()) / (returns.std() + 1e-8)

# Calculate loss
log_probs_tensor = torch.stack(episode_log_probs)
loss = -torch.sum(returns * log_probs_tensor)

# Update policy
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

#### Key Parameters
- `learning_rate`: Policy network learning rate
- `gamma`: Discount factor
- Network architecture parameters

#### Advantages & Limitations
- **Pros**: 
  - Simple policy gradient concept
  - Works with discrete and continuous actions
  - Learns stochastic policies naturally
- **Cons**: 
  - High variance due to Monte Carlo returns
  - Episodic updates (must wait for episode completion)
  - On-policy sample inefficiency
- **Common Pitfalls**: High variance leading to unstable training, careful learning rate tuning required
- **Use Cases**: Simple benchmarks, conceptual understanding, foundation for actor-critic methods

### Trust Region Policy Optimization (TRPO)

#### Core Concept
Improves policy gradient updates by constraining policy changes (measured by KL divergence) at each step, ensuring stable and monotonic improvement. **On-policy** approach.

#### Mathematical Formulation
Solves constrained optimization problem:
```math
\max_{\theta} \quad \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right]
```
```math
\text{subject to} \quad \mathbb{E}_t [D_{KL}(\pi_{\theta_{old}}(\cdot|s_t) || \pi_{\theta}(\cdot|s_t))] \le \delta
```

Solved using Conjugate Gradient (for direction $\approx F^{-1}g$) and Line Search (for constraint satisfaction), where $F$ is the Fisher Information Matrix.

#### Algorithm Steps
1. Initialize actor $\pi_\theta$ and critic $V_\phi$
2. For each iteration:
   - Collect trajectories using $\pi_{\theta_{old}}$
   - Store states, actions, rewards, log probabilities
   - Compute advantages $\hat{A}_t$ using GAE with $V_\phi$
   - Compute policy gradient $g$
   - Use Conjugate Gradient + Fisher-Vector Products to find step direction $s \approx F^{-1}g$
   - Perform line search to find step size $\beta \alpha$ satisfying:
     - KL constraint $\delta$
     - Improvement in surrogate objective
   - Update actor: $\theta_{new} \leftarrow \theta_{old} + \beta \alpha s$
   - Update critic $V_\phi$ using collected data

#### Code Implementation
```python
# Conceptual TRPO update process
policy_gradient = calculate_policy_gradient(...)
step_direction = conjugate_gradient(
    fisher_vector_product_func, 
    policy_gradient, 
    ...
)
initial_step_size = calculate_initial_step_size(
    step_direction, 
    policy_gradient, 
    max_kl, 
    ...
)
final_update, success = backtracking_line_search(
    actor, 
    ..., 
    step_direction, 
    initial_step_size, 
    max_kl, 
    ...
)
if success:
    apply_update(actor, final_update)
update_critic(...)
```

#### Key Parameters
- `delta`: KL divergence constraint
- `gamma`: Discount factor
- `lambda`: GAE parameter
- CG iterations and damping parameters
- Line search parameters

#### Advantages & Limitations
- **Pros**: 
  - Theoretical monotonic improvement guarantee
  - Very stable updates
  - Strong performance on continuous control
- **Cons**: 
  - Complex implementation (FVP, CG, line search)
  - Computationally expensive per update
  - On-policy sample inefficiency
- **Common Pitfalls**: Implementing FVP and CG correctly, tuning trust region size
- **Use Cases**: Continuous control, high-stability requirements, benchmark for simpler algorithms

---

## Actor-Critic Methods

### Advantage Actor-Critic (A2C)

#### Core Concept
**Synchronous** version of A3C using actor (policy) and critic (value function) trained on experience batches. Reduces variance compared to REINFORCE through advantage estimates. **On-policy** learning.

#### Mathematical Formulation
**Actor Loss (minimize):**
```math
L_{actor} = - \mathbb{E}_t [ \log \pi(a_t | s_t; \theta) \hat{A}_t^{\text{detached}} + c_e H(\pi) ]
```

**Critic Loss (minimize):**
```math
L_{critic} = \mathbb{E}_t [ (R_t - V(s_t; \phi))^2 ]
```

**Advantage Estimate:**
```math
\hat{A}_t = R_t - V(s_t; \phi)
```

Where $R_t$ often uses n-step returns or GAE for better estimates.

#### Algorithm Steps
1. Initialize shared actor $\pi_\theta$ and critic $V_\phi$
2. Loop for iterations:
   - Collect batch of N steps: $(s_t, a_t, r_{t+1}, s_{t+1}, d_t)$ using $\pi_\theta$
   - Compute n-step returns $R_t$ and advantages $\hat{A}_t$ using $V_\phi$
   - Compute actor loss (policy gradient + entropy) and critic loss (MSE)
   - Calculate gradients for both networks based on batch
   - Apply synchronous gradient updates to $\theta$ and $\phi$

#### Code Implementation
```python
# Calculate advantages and returns (e.g., using GAE)
advantages, returns_to_go = compute_gae_and_returns(...)

# Evaluate current policy and value
policy_dist = actor(states)
log_probs = policy_dist.log_prob(actions)
entropy = policy_dist.entropy().mean()
values_pred = critic(states).squeeze()

# Calculate losses
policy_loss = -(log_probs * advantages.detach()).mean() - \
              entropy_coeff * entropy
value_loss = F.mse_loss(values_pred, returns_to_go.detach())

# Optimize Actor
actor_optimizer.zero_grad()
policy_loss.backward()
actor_optimizer.step()

# Optimize Critic
critic_optimizer.zero_grad()
(value_loss_coeff * value_loss).backward()
critic_optimizer.step()
```

#### Key Parameters
- `learning_rates`: Separate for actor and critic
- `gamma`: Discount factor
- `lambda`: GAE parameter
- `n_steps`: Rollout length
- `value_loss_coeff`: Value loss coefficient
- `entropy_coeff`: Entropy bonus coefficient

#### Advantages & Limitations
- **Pros**: 
  - More stable than REINFORCE
  - Simpler than A3C/TRPO/PPO
  - Good baseline performance
  - Efficient GPU utilization
- **Cons**: 
  - On-policy sample inefficiency
  - Updates can still have variance
  - Performance sometimes lower than PPO
- **Common Pitfalls**: Balancing actor/critic learning rates, choosing rollout length
- **Use Cases**: Discrete/continuous control benchmarks, simpler alternative to A3C/PPO

### Asynchronous Advantage Actor-Critic (A3C)

#### Core Concept
Uses multiple parallel workers with local actor-critic networks and environment instances. Workers compute gradients locally and asynchronously update shared global network. **On-policy** learning.

#### Mathematical Formulation
Same loss functions as A2C per worker, but updates applied asynchronously to global parameters $\theta_{global}, \phi_{global}$ using gradients from local parameters $\theta', \phi'$.

#### Algorithm Steps (Per Worker)
1. Initialize local network, synchronize with global
2. Loop:
   - Reset local gradients and sync with global network
   - Collect n-steps of experience using local policy
   - Calculate n-step returns $R_t$ and advantages $\hat{A}_t$
   - Compute gradients for actor and critic losses
   - Apply gradients asynchronously to global network
   - Reset environment if episode completed

#### Code Implementation
```python
# Inside worker loop
local_model.load_state_dict(global_model.state_dict())

# ... collect n-steps data ...
returns, advantages = compute_n_step_returns_advantages(...)

# Calculate losses
policy_loss = -(log_probs * advantages.detach()).mean() - \
              entropy_coeff * entropy
value_loss = F.mse_loss(values_pred, returns.detach())
total_loss = policy_loss + value_loss_coeff * value_loss

# Update global model
global_optimizer.zero_grad()
total_loss.backward()  # Calculates gradients on local model

# Transfer gradients to global model
for local_param, global_param in zip(local_model.parameters(), 
                                   global_model.parameters()):
    if global_param.grad is not None: 
        global_param.grad.data.zero_()
    if local_param.grad is not None:
         global_param.grad = local_param.grad.clone()
         
global_optimizer.step()  # Updates global model
```

#### Key Parameters
- `num_workers`: Number of parallel workers
- `n_steps`: Steps per worker update
- Learning rates for actor/critic
- `gamma`: Discount factor
- Coefficients for value loss and entropy
- Optimizer details (shared Adam)

#### Advantages & Limitations
- **Pros**: 
  - No replay buffer needed
  - Decorrelates data via parallelism
  - Efficient on multi-core CPUs
  - Stable learning through asynchronous updates
- **Cons**: 
  - Complex implementation (multiprocessing, shared memory)
  - Potential for stale gradients
  - Often less GPU-efficient than A2C
- **Common Pitfalls**: Race conditions, worker synchronization issues
- **Use Cases**: Historically significant, CPU-based parallel training

### Deep Deterministic Policy Gradient (DDPG)

#### Core Concept
**Off-policy** actor-critic algorithm for **continuous action spaces**. Learns deterministic policy (actor) with Q-function (critic). Uses DQN ideas (replay buffer, target networks) for stability.

#### Mathematical Formulation
**Critic Update (minimize loss):**
```math
L(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} [ (y - Q(s, a; \phi))^2 ]
```
where $y = r + \gamma Q'(s', \mu'(s'; \theta'); \phi')$

**Actor Update (maximize objective, minimize negative):**
```math
L(\theta) = - \mathbb{E}_{s \sim \mathcal{D}} [ Q(s, \mu(s; \theta); \phi) ]
```

Where $\mu, Q$ are main networks and $\mu', Q'$ are target networks.

#### Algorithm Steps
1. Initialize actor $\mu_\theta$, critic $Q_\phi$, target networks $\mu'_{\theta'}, Q'_{\phi'}$, replay buffer $\mathcal{D}$
2. For each step:
   - Select action $a = \mu(s; \theta) + \text{Noise}$ (exploration)
   - Execute action $a$, observe reward $r$ and next state $s'$
   - Store transition $(s, a, r, s')$ in replay buffer $\mathcal{D}$
   - Sample mini-batch from $\mathcal{D}$
   - Update critic $Q_\phi$ using TD error from target networks
   - Update actor $\mu_\theta$ using gradient from critic's output $Q(s, \mu(s))$
   - Soft-update target networks:
     - $\theta' \leftarrow \tau \theta + (1-\tau)\theta'$
     - $\phi' \leftarrow \tau \phi + (1-\tau)\phi'$

#### Code Implementation
```python
# Critic Update
with torch.no_grad():
    next_actions = target_actor(next_state_batch)
    target_q = target_critic(next_state_batch, next_actions)
    y = reward_batch + gamma * (1 - done_batch) * target_q
    
current_q = critic(state_batch, action_batch)
critic_loss = F.mse_loss(current_q, y)

critic_optimizer.zero_grad()
critic_loss.backward()
critic_optimizer.step()

# Actor Update
actor_actions = actor(state_batch)
q_for_actor = critic(state_batch, actor_actions)  # Gradients flow through critic
actor_loss = -q_for_actor.mean()

actor_optimizer.zero_grad()
actor_loss.backward()
actor_optimizer.step()

# Soft Updates
soft_update(target_critic, critic, tau)
soft_update(target_actor, actor, tau)
```

#### Key Parameters
- `buffer_size`: Replay buffer capacity
- `batch_size`: Mini-batch size for training
- `gamma`: Discount factor
- `tau`: Soft update rate for target networks
- Actor/critic learning rates
- Exploration noise parameters

#### Advantages & Limitations
- **Pros**: 
  - Off-policy sample efficiency
  - Handles continuous actions directly
  - Stable learning with target networks
- **Cons**: 
  - Sensitive to hyperparameters
  - Can suffer from Q-value overestimation
  - Exploration can be challenging in continuous spaces
- **Common Pitfalls**: Learning rate tuning, noise scale/decay, target update rate
- **Use Cases**: Continuous control tasks (robotics, physics simulation)

### Soft Actor-Critic (SAC)

#### Core Concept
**Off-policy** actor-critic algorithm for **continuous actions** based on **maximum entropy** framework. Learns stochastic policy maximizing both expected return and policy entropy for improved exploration and robustness.

#### Mathematical Formulation
**Objective includes entropy term:**
```math
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[\sum \gamma^t (R_t + \alpha H(\pi(\cdot|s_t)))\right]
```

Uses twin Q-critics, target critics, and often auto-tunes entropy coefficient $\alpha$.

**Critic Updates (minimize loss for $Q_1, Q_2$):**
```math
L(\phi_i) = \mathbb{E} [ (Q_i(s,a) - y)^2 ]
```
where $y = r + \gamma (1-d) [\min_{j=1,2} Q'_j(s', a') - \alpha \log \pi(a'|s')]$ and $a' \sim \pi(\cdot|s')$

**Actor Update (minimize loss):**
```math
L(\theta) = \mathbb{E}_{s, a \sim \pi} \left[ \alpha \log \pi(a|s) - \min_{j=1,2} Q_j(s, a) \right]
```

**Alpha Update (minimize loss for auto-tuning):**
```math
L(\log \alpha) = \mathbb{E}_{a \sim \pi} [ -\log \alpha (\log \pi(a|s) + \bar{H}) ]
```
where $\bar{H}$ is target entropy.

#### Algorithm Steps
1. Initialize actor $\pi_\theta$, twin critics $Q_{\phi_1}, Q_{\phi_2}$, target critics $Q'_{\phi'_1}, Q'_{\phi'_2}$, replay buffer $\mathcal{D}$, $\log \alpha$
2. For each step:
   - Select action $a \sim \pi(\cdot|s; \theta)$ (sampling from stochastic policy)
   - Execute action $a$, observe reward $r$ and next state $s'$
   - Store transition $(s, a, r, s')$ in replay buffer $\mathcal{D}$
   - Sample mini-batch from $\mathcal{D}$
   - Update critics $Q_{\phi_1}, Q_{\phi_2}$ using soft TD target (min of target Q's minus scaled log prob)
   - Update actor $\pi_\theta$ using gradient based on min Q and log probability
   - Update $\alpha$ (if auto-tuning) based on policy entropy
   - Soft-update target critics

#### Code Implementation
```python
# Critic Target Calculation
with torch.no_grad():
    next_action, next_log_prob = actor(next_state_batch)
    q1_target_next, q2_target_next = target_critic(next_state_batch, next_action)
    q_target_next = torch.min(q1_target_next, q2_target_next)
    alpha = torch.exp(log_alpha).detach()
    soft_target = q_target_next - alpha * next_log_prob
    y = reward_batch + gamma * (1.0 - done_batch) * soft_target

# Critic Updates (MSE loss for both critics)
# ... standard critic update using y as target ...

# Actor Update
pi_action, pi_log_prob = actor(state_batch)
q1_pi, q2_pi = critic(state_batch, pi_action)  # Gradients enabled
min_q_pi = torch.min(q1_pi, q2_pi)
actor_loss = (alpha * pi_log_prob - min_q_pi).mean()

# Alpha Update (entropy regularization auto-tuning)
alpha_loss = -(log_alpha * (pi_log_prob.detach() + target_entropy)).mean()

# Soft Updates for target networks
# ...
```

#### Key Parameters
- `buffer_size`: Replay buffer capacity
- `batch_size`: Mini-batch size
- `gamma`: Discount factor
- `tau`: Soft update rate
- Learning rates (actor, critic, alpha)
- Initial `alpha` value
- `target_entropy`: Target entropy for auto-tuning

#### Advantages & Limitations
- **Pros**: 
  - State-of-the-art sample efficiency on continuous control
  - Robust performance across diverse tasks
  - Excellent exploration through entropy maximization
  - Automatic entropy coefficient tuning
- **Cons**: 
  - More complex than DDPG/PPO
  - Requires careful implementation (tanh squashing correction)
  - Multiple networks to tune
- **Common Pitfalls**: Log probability calculation errors, alpha tuning stability, target entropy selection
- **Use Cases**: Continuous control (robotics, benchmarks), tasks requiring robust exploration

### Proximal Policy Optimization (PPO)

#### Core Concept
**On-policy** actor-critic method simplifying TRPO's constrained update using **clipped surrogate objective**. Allows multiple epochs of updates on collected data for better sample efficiency.

#### Mathematical Formulation
**Policy Ratio:**
```math
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
```

**Clipped Objective (minimize negative):**
```math
L^{CLIP}(\theta) = -\mathbb{E}_t [ \min( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t ) ]
```

**Combined Loss:**
```math
L = L^{CLIP} + c_1 L^{VF} - c_2 S
```
where $L^{VF}$ is value function loss and $S$ is entropy bonus.

#### Algorithm Steps
1. Initialize actor $\pi_\theta$ and critic $V_\phi$
2. For each iteration:
   - Collect batch of trajectories using $\pi_{\theta_{old}}$
   - Store states, actions, rewards, dones, old log probabilities
   - Compute advantages $\hat{A}_t$ (using GAE) and returns $R_t$
   - For K epochs:
     - For each mini-batch in collected data:
       - Calculate policy ratio $r_t(\theta)$
       - Compute clipped surrogate loss $L^{CLIP}$
       - Compute value loss $L^{VF}$ and entropy bonus $S$
       - Compute combined loss $L$
       - Update $\theta$ and $\phi$ using gradient descent on $L$

#### Code Implementation
```python
# Inside PPO update loop (for one epoch/minibatch)
policy_dist = actor(states)
log_probs_new = policy_dist.log_prob(actions)
entropy = policy_dist.entropy().mean()
values_pred = critic(states).squeeze()

# Calculate ratio
ratio = torch.exp(log_probs_new - log_probs_old)

# Calculate policy loss
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages
policy_loss = -torch.min(surr1, surr2).mean() - entropy_coeff * entropy

# Calculate value loss
value_loss = F.mse_loss(values_pred, returns_to_go)

# Combined optimization (or separate updates)
total_loss = policy_loss + value_loss_coeff * value_loss
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

#### Key Parameters
- `clip_epsilon`: Clipping parameter for policy ratio
- `gamma`: Discount factor
- `lambda`: GAE parameter
- Learning rates for actor/critic
- `num_epochs`: Number of epochs per iteration
- `mini_batch_size`: Size of mini-batches
- `value_loss_coeff`: Value loss coefficient
- `entropy_coeff`: Entropy bonus coefficient

#### Advantages & Limitations
- **Pros**: 
  - Simpler implementation than TRPO
  - Stable and reliable updates
  - Often state-of-the-art or near-SOTA performance
  - Good sample efficiency for on-policy method
- **Cons**: 
  - Still on-policy (less sample efficient than off-policy)
  - Performance sensitive to implementation details
  - Hyperparameter sensitive
- **Common Pitfalls**: Advantage/observation normalization, learning rate scheduling, clipping parameter selection
- **Use Cases**: Default choice for many discrete/continuous control tasks, RLHF for large language models

---

## Value-Based Deep Methods

### Deep Q-Networks (DQN)

#### Core Concept
Combines Q-learning with deep neural networks to approximate $Q(s, a; \theta)$. Uses **Experience Replay** and **Target Networks** for training stability. **Off-policy** learning.

#### Mathematical Formulation
Minimizes TD error using target network $Q'$:
```math
L(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} [ (y - Q(s, a; \theta))^2 ]
```
where:
```math
y = r + \gamma (1-d) \max_{a'} Q'(s', a'; \theta^{-})
```

$\mathcal{D}$ is the replay buffer and $\theta^{-}$ are target network parameters.

#### Algorithm Steps
1. Initialize Q-network $Q_\theta$, target network $Q'_{\theta^-}$, replay buffer $\mathcal{D}$
2. For each episode:
   - For each step:
     - Choose action using $\epsilon$-greedy policy on $Q_\theta$
     - Execute action, observe reward and next state
     - Store transition $(s, a, r, s', \text{done})$ in $\mathcal{D}$
     - Sample mini-batch from $\mathcal{D}$
     - Compute target using $Q'_{\theta^-}$
     - Update $Q_\theta$ by minimizing loss $L(\theta)$
     - Periodically update target network: $\theta^- \leftarrow \theta$

#### Code Implementation
```python
# DQN Optimization Step
non_final_mask = torch.tensor(...)  # Mask for non-terminal next states
non_final_next_states = torch.cat(...)
state_batch, action_batch, reward_batch, done_batch = ...  # From replay buffer

# Current Q-values
state_action_values = policy_net(state_batch).gather(1, action_batch)

# Next state values
next_state_values = torch.zeros(batch_size, device=device)
with torch.no_grad():
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
# Expected Q-values
expected_state_action_values = (next_state_values * gamma) + reward_batch

# Compute loss
loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

# Optimize
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

#### Key Parameters
- `buffer_size`: Replay buffer capacity
- `batch_size`: Mini-batch size for training
- `gamma`: Discount factor
- `tau` or `target_update_freq`: Target network update frequency
- `learning_rate`: Network learning rate
- `epsilon` schedule: Exploration schedule

#### Advantages & Limitations
- **Pros**: 
  - Handles high-dimensional states (e.g., pixel observations)
  - Off-policy sample efficiency
  - Stable learning through replay buffer and target networks
  - Foundation for many advanced value-based methods
- **Cons**: 
  - Primarily designed for discrete actions
  - Can overestimate Q-values
  - Sensitive to hyperparameters
  - Requires careful tuning
- **Common Pitfalls**: Target network update frequency, replay buffer management, hyperparameter selection
- **Use Cases**: Atari games from pixels, discrete action tasks with large state spaces

---

## Multi-Agent RL (MARL)

### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

#### Core Concept
Extends DDPG to multi-agent settings using "centralized training, decentralized execution" paradigm. Each agent has an actor and **centralized critic** that observes joint states and actions. **Off-policy** learning.

#### Mathematical Formulation
**Centralized Critic for agent $i$:**
```math
Q_i(x, a_1, ..., a_N; \phi_i)
```
where $x = (o_1, ..., o_N)$ is joint observation.

**Critic Update:**
```math
L(\phi_i) = \mathbb{E} [ (y_i - Q_i(x, \mathbf{a}))^2 ]
```
where $y_i = r_i + \gamma Q'_i(x', \mu'_1(o'_1), ..., \mu'_N(o'_N))$

**Actor Update:**
```math
L(\theta_i) = - \mathbb{E} [ Q_i(x, \mu_1(o_1), ..., \mu_N(o_N)) ]
```

#### Algorithm Steps
1. Initialize actors $\mu_i$, critics $Q_i$, target networks $\mu'_i, Q'_i$, replay buffer $\mathcal{D}$
2. For each step:
   - Each agent $i$ chooses action $a_i = \mu_i(o_i) + \text{Noise}$
   - Execute joint action $\mathbf{a} = (a_1, ..., a_N)$
   - Observe rewards $r = (r_1, ..., r_N)$ and next observations $o' = (o'_1, ..., o'_N)$
   - Store transition $(o, \mathbf{a}, r, o')$ in replay buffer $\mathcal{D}$
   - Sample mini-batch from $\mathcal{D}$
   - For each agent $i$:
     - Update critic $Q_i$ using centralized information
     - Update actor $\mu_i$ using critic gradient
   - Soft-update all target networks

#### Code Implementation
```python
# Critic Update (Agent i)
with torch.no_grad():
    target_actions_next = [target_actors[j](next_obs_batch[:, j]) 
                          for j in range(num_agents)]
    target_q_next = target_critics[i](joint_next_obs_batch, 
                                     torch.cat(target_actions_next, dim=1))
    y_i = rewards_batch[:, i] + gamma * (1 - dones_batch[:, i]) * target_q_next

current_q_i = critics[i](joint_obs_batch, joint_actions_batch)
critic_loss_i = F.mse_loss(current_q_i, y_i)

# Optimize critic i
critic_optimizers[i].zero_grad()
critic_loss_i.backward()
critic_optimizers[i].step()

# Actor Update (Agent i)
current_actions_policy = [actors[j](obs_batch[:, j]) for j in range(num_agents)]
q_actor_loss = critics[i](joint_obs_batch, torch.cat(current_actions_policy, dim=1))
actor_loss_i = -q_actor_loss.mean()

# Optimize actor i
actor_optimizers[i].zero_grad()
actor_loss_i.backward()
actor_optimizers[i].step()
```

#### Key Parameters
- Similar to DDPG but potentially per-agent
- Buffer size, batch size, discount factor, soft update rate
- Learning rates for actors and critics
- Exploration noise parameters

#### Advantages & Limitations
- **Pros**: 
  - Addresses non-stationarity in multi-agent environments
  - Decentralized execution (scalable deployment)
  - Handles mixed cooperative/competitive settings
  - Leverages centralized training for stability
- **Cons**: 
  - Centralized critic scales poorly with many agents
  - Credit assignment can be challenging in cooperative settings
  - Implementation complexity increases with agent count
- **Use Cases**: Multi-robot coordination, predator-prey environments, cooperative navigation

### QMIX (Monotonic Value Function Factorization)

#### Core Concept
Value-based MARL algorithm for **cooperative** tasks. Learns individual Q-functions $Q_i$ and mixes them **monotonically** using mixing network conditioned on global state. **Off-policy**, **centralized training, decentralized execution**.

#### Mathematical Formulation
**Value Function Factorization:**
```math
Q_{tot}(x, \mathbf{a}) = f_{mix}(Q_1(o_1, a_1), ..., Q_N(o_N, a_N); x)
```

**Monotonicity Constraint:**
```math
\frac{\partial Q_{tot}}{\partial Q_i} \ge 0
```
Enforced by non-negative mixer weights (often via hypernetworks).

**Loss Function:**
```math
L = \mathbb{E} [ (y - Q_{tot}(x, \mathbf{a}))^2 ]
```
where $y = r + \gamma Q'_{tot}(x', \mathbf{a}')$ and $a'_i = \arg\max_a Q'_i(o'_i, a)$

#### Algorithm Steps
1. Initialize agent networks $Q_i$, target networks $Q'_i$, mixer $f_{mix}$, target mixer $f'_{mix}$, replay buffer $\mathcal{D}$
2. For each step:
   - Each agent $i$ chooses action $a_i$ using $\epsilon$-greedy on $Q_i(o_i)$
   - Execute joint action $\mathbf{a}$, observe team reward $r$, next observations $o'$, next global state $x'$
   - Store transition $(o, \mathbf{a}, r, o', x, x', \text{done})$ in replay buffer $\mathcal{D}$
   - Sample mini-batch from replay buffer
   - Calculate target $y$ using target networks $Q'_i$ and target mixer $f'_{mix}$
   - Calculate current $Q_{tot}$ using main networks $Q_i$ and mixer $f_{mix}$
   - Compute loss $L$ and update all parameters via gradient descent
   - Soft-update target networks

#### Code Implementation
```python
# Calculate Target Q_tot
with torch.no_grad():
    # Get max Q'_i for each agent i in next state
    target_agent_qs = []
    for i in range(num_agents):
        target_q_i = target_agent_nets[i](next_obs_batch[:, i])
        target_agent_qs.append(target_q_i.max(dim=1, keepdim=True)[0])
    
    target_agent_qs = torch.cat(target_agent_qs, dim=1)  # Shape (batch, num_agents)
    q_tot_target = target_mixer(target_agent_qs, next_global_state_batch)
    y = reward_batch + gamma * (1 - done_batch) * q_tot_target

# Calculate Current Q_tot
current_agent_qs = []
for i in range(num_agents):
    q_i = agent_nets[i](obs_batch[:, i])
    # Get Q_i for the action taken by agent i
    q_i_taken = q_i.gather(1, actions_batch[:, i].unsqueeze(1))
    current_agent_qs.append(q_i_taken)

current_agent_qs = torch.cat(current_agent_qs, dim=1)  # Shape (batch, num_agents)
q_tot_current = mixer(current_agent_qs, global_state_batch)

# Loss and Optimization
loss = F.mse_loss(q_tot_current, y)
optimizer.zero_grad()
loss.backward()  # Gradients flow to all agent nets and mixer
optimizer.step()

# Soft update targets
soft_update_targets()
```

#### Key Parameters
- Similar to DQN parameters
- Buffer size, batch size, discount factor
- Learning rate, exploration schedule
- Mixing network architecture
- Hypernetwork details (if used)

#### Advantages & Limitations
- **Pros**: 
  - Excellent performance on cooperative tasks
  - Enforces Individual-Global-Max (IGM) principle
  - Scales better in action space than joint Q-learning
  - Theoretical guarantees under monotonicity constraint
- **Cons**: 
  - Limited representational power due to monotonicity requirement
  - Requires global state information for mixer
  - Only applicable to cooperative settings
- **Use Cases**: Cooperative multi-agent tasks (SMAC benchmark), resource allocation, team coordination

---

## Hierarchical RL (HRL)

### Hierarchical Actor-Critic (HAC)

#### Core Concept
Learns policies at multiple abstraction levels. Higher levels set subgoals as "actions" for lower levels, which execute primitive actions to achieve them within time limits. Uses **intrinsic rewards** and **hindsight** for subgoal learning. **Off-policy** approach.

**Note: The notebook implementation has known issues and should be used for conceptual understanding rather than direct implementation.**

#### Mathematical Formulation
**Multi-Level Structure:**
- Level $i$: Policy $\pi_i(a_i | s, g_i)$, Q-function $Q_i(s, a_i, g_i)$
- $a_i$ represents subgoals for level $i-1$
- Low Level (0): Uses intrinsic reward $r_{int}$ (success/failure to reach subgoal)
- High Level (1): Uses environment reward $R = \sum r_{env}$

**Hindsight Learning:**
Relabel transitions with achieved states as goals, creating artificial success experiences.

#### Algorithm Steps (2-Level Example)
1. Initialize networks $Q_0, Q'_0, Q_1, Q'_1$ and replay buffers $D_0, D_1$
2. For each episode:
   - Reset environment: `s = env.reset()`
   - While overall goal $G$ not reached:
     - **High-Level Decision**: Choose subgoal $g_0 = \text{select_action}(\text{level}=1, s, G)$
     - Initialize: `transitions = []`, `total_env_reward = 0`, `s_start = s`
     - **Low-Level Execution**: For $h$ from 1 to $H$ (time limit):
       - Choose primitive action: $a = \text{select_action}(\text{level}=0, s, g_0)$
       - Execute action: $(r_{env}, s_{next}, env_{done}) = \text{env.step}(a)$
       - Calculate intrinsic reward: $r_{int} = \text{get_intrinsic_reward}(s_{next}, g_0)$
       - Store low-level transition: $(s, a, r_{int}, s_{next}, g_0, \text{done}, s_{next})$
       - Update state: $s = s_{next}$
       - If subgoal reached or environment done: break
     - **High-Level Storage**: Store high-level transition with cumulative reward
     - **Hindsight Processing**: Add transitions to buffers with hindsight relabeling
     - **Network Updates**: Update $Q_0, Q_1$ from respective buffers
     - If environment done: break outer loop

#### Code Implementation
```python
# Inside low-level execution loop
# Execute action and observe transition
next_state_norm, env_done = env.step(action)

# Calculate intrinsic reward
intrinsic_reward = -1.0  # Default failure
subgoal_achieved = self._test_goal(next_state_norm, goal_norm)
if subgoal_achieved:
    intrinsic_reward = 0.0  # Success reward

# Store original transition
buffer.push(
    state_norm, 
    action, 
    reward=intrinsic_reward, 
    next_state=next_state_norm,
    goal=goal_norm, 
    done=env_done or subgoal_achieved, 
    achieved_goal=next_state_norm,
    level=0
)

# Hindsight relabeling handled in buffer.sample()
# Replaces goal with achieved_goal and adjusts reward/done accordingly
```

#### Key Parameters
- Number of hierarchical levels
- Time limits $H$ for each level
- Learning rates for each level
- Discount factors, exploration schedules
- Buffer sizes for each level
- Hindsight probability $p$
- Goal tolerance thresholds

#### Advantages & Limitations
- **Pros**: 
  - Can solve long-horizon tasks with sparse rewards
  - Structured exploration through hierarchical decomposition
  - Potential for skill reuse across different tasks
  - Addresses curse of dimensionality in temporal dimension
- **Cons**: 
  - **Very complex implementation with many components**
  - **Highly sensitive to goal definitions and time limits**
  - **Known issues in current implementation**
  - **Potential for suboptimal subgoal selection**
  - **Difficult to tune multiple interacting components**
- **Common Pitfalls**: 
  - Incorrect hindsight logic implementation
  - Poor intrinsic reward design
  - Infeasible subgoal generation
  - Improper time limit tuning
- **Use Cases**: 
  - Complex robotics manipulation
  - Long-horizon navigation tasks
  - Multi-step planning problems
  - Research into hierarchical decomposition

---

## Planning & Model-Based Methods

### Monte Carlo Tree Search (MCTS)

#### Core Concept
**Online planning** algorithm building search trees using simulated trajectories from current state. Uses statistics (visit counts, values) and Upper Confidence bounds applied to Trees (UCT) for exploration/exploitation balance. **Requires environment simulator/model**.

#### Mathematical Formulation
**Tree Structure:**
- Nodes store: state $s$, visit count $N(s)$, total value $W(s)$
- Edges store: action $a$, count $N(s, a)$, value $W(s, a)$

**UCT Selection Policy:**
Choose action $a$ maximizing:
```math
Q(s, a) + C \sqrt{\frac{\ln N(s)}{N(s, a)}}
```
where $C$ is the exploration constant.

#### Algorithm Steps (Single MCTS Iteration)
1. Initialize tree with root node = current state $s_{root}$
2. Repeat for $N$ simulations:
   - **Selection**: Starting from root, traverse tree using UCT until reaching expandable node
   - **Expansion**: If node not fully expanded and not terminal, add one child node
   - **Simulation**: Run rollout from expanded node using default policy, obtain return $R$
   - **Backpropagation**: Update visit counts and values for all nodes/edges from expanded node back to root
3. **Action Selection**: Choose best action from root based on visit counts or values

#### Code Implementation
```python
# UCT Selection (inside select_best_child_uct method)
def select_best_child_uct(self, node):
    best_score = -float('inf')
    best_child = None
    
    for action, child in node.children.items():
        if child.visit_count == 0:
            uct_score = float('inf')  # Unvisited nodes have highest priority
        else:
            # UCT formula
            exploit = child.total_value / child.visit_count
            explore = self.exploration_constant * math.sqrt(
                math.log(node.visit_count) / child.visit_count
            )
            uct_score = exploit + explore
            
        if uct_score > best_score:
            best_score = uct_score
            best_child = child
            
    return best_child

# Main MCTS search loop
def search(self, state, num_simulations):
    root = Node(state)
    
    for _ in range(num_simulations):
        # Selection & Expansion
        leaf = self.select_and_expand(root)
        
        # Simulation
        reward = self.rollout(leaf.state)
        
        # Backpropagation
        self.backpropagate(leaf, reward)
    
    return self.best_action(root)
```

#### Key Parameters
- `num_simulations`: Computational budget per action selection
- `exploration_constant` (C): Balance between exploration and exploitation
- `rollout_depth`: Maximum depth for simulation phase
- `gamma`: Discount factor for rollout rewards

#### Advantages & Limitations
- **Pros**: 
  - Anytime algorithm (can be stopped at any point)
  - Handles large state/action spaces effectively
  - No explicit value function required
  - Asymmetric tree growth focuses computation on promising areas
  - Strong theoretical foundation
- **Cons**: 
  - Requires accurate environment simulator/model
  - Computationally intensive per action selection
  - Performance heavily dependent on rollout policy quality
  - May struggle with very deep trees
- **Common Pitfalls**: 
  - Tuning exploration constant C
  - Implementing efficient tree operations
  - Designing effective rollout policies
- **Use Cases**: Game playing (Go, Chess), planning with simulators, discrete decision problems

### PlaNet (Deep Planning Network)

#### Core Concept
**Model-based** RL agent learning **latent dynamics model** from high-dimensional observations. Performs **planning in latent space** using Cross-Entropy Method (CEM) for action selection. **Off-policy** learning with world model.

#### Mathematical Formulation
**World Model Components:**
- Transition model: $p(s_{t+1}|s_t, a_t)$
- Reward model: $p(r_t|s_t)$
- Observation model: $p(o_t|s_t)$ (reconstruction)
- Encoder: $q(s_t|...)$ (state inference)

**Model Loss (simplified):**
Maximize data likelihood via ELBO, including reconstruction, reward prediction, and KL regularization terms.

**Planning Objective (CEM):**
Optimize expected return over action sequences:
```math
\mathbb{E} \left[ \sum_{k=t}^{t+H-1} \gamma^{k-t} \hat{r}_k \right]
```
over action sequences $a_t, ..., a_{t+H-1}$ using learned latent model.

#### Algorithm Steps
1. Initialize latent dynamics model components and replay buffer $\mathcal{D}$
2. Main loop:
   - **Environment Interaction**:
     - Observe current state $s_t$
     - Plan action $a_t$ using CEM in latent space with current model
     - Execute action $a_t$, observe reward $r_t$ and next state $s_{t+1}$
     - Store transition $(s_t, a_t, r_t, s_{t+1})$ in replay buffer $\mathcal{D}$
   - **Model Training**:
     - Sample sequences from replay buffer $\mathcal{D}$
     - Update world model parameters to minimize prediction/reconstruction losses
     - Train all model components (transition, reward, observation, encoder)

#### Code Implementation
```python
# Main training loop structure
def train_step(self):
    # Environment interaction
    state = self.get_current_state()
    
    # Planning step using CEM
    action = self.cem_planner(
        dynamics_model=self.world_model,
        state=state,
        horizon=self.PLANNING_HORIZON,
        num_candidates=self.CEM_CANDIDATES,
        num_elites=self.CEM_ELITES,
        num_iterations=self.CEM_ITERATIONS,
        gamma=self.CEM_GAMMA
    )
    
    # Execute in environment
    next_state, reward, done = self.env.step(action)
    
    # Store transition
    self.replay_buffer.add(state, action, reward, next_state, done)
    
    # Train world model
    if len(self.replay_buffer) > self.min_buffer_size:
        sequences = self.replay_buffer.sample_sequences(
            batch_size=self.batch_size,
            sequence_length=self.sequence_length
        )
        self.train_world_model(sequences)

# CEM Planning (conceptual structure)
def cem_planning(self, model, initial_state, horizon, candidates, elites, iterations):
    # Initialize action distribution
    action_dist = self.initialize_action_distribution()
    
    for iteration in range(iterations):
        # Sample action sequences
        action_sequences = action_dist.sample(candidates)
        
        # Evaluate sequences using world model
        returns = []
        for sequence in action_sequences:
            state = initial_state
            total_return = 0
            for t, action in enumerate(sequence):
                reward = model.predict_reward(state)
                total_return += (self.gamma ** t) * reward
                state = model.predict_next_state(state, action)
            returns.append(total_return)
        
        # Select elite sequences
        elite_indices = np.argsort(returns)[-elites:]
        elite_sequences = action_sequences[elite_indices]
        
        # Update action distribution
        action_dist = self.fit_distribution(elite_sequences)
    
    # Return first action of best sequence
    return action_dist.mean[0]
```

#### Key Parameters
- **Model Architecture**: Latent state size, hidden dimensions, network depths
- **Model Training**: Learning rate, sequence length, batch size, buffer size
- **Planning**: 
  - `horizon` (H): Planning horizon length
  - `num_candidates` (J): CEM candidate sequences
  - `num_elites` (M): Elite sequences for distribution update
  - `num_iterations`: CEM optimization iterations
  - `gamma`: Discount factor for planning

#### Advantages & Limitations
- **Pros**: 
  - Exceptional sample efficiency (especially from high-dimensional observations)
  - Learns compact, interpretable world representations
  - Effective planning capabilities
  - Works well with continuous control from pixels
  - Strong empirical performance on various tasks
- **Cons**: 
  - Complex model training requiring careful tuning
  - Planning computationally expensive (but parallelizable)
  - Model inaccuracies can compound over planning horizon
  - Sensitive to model architecture and training stability
- **Common Pitfalls**: 
  - Model convergence issues
  - Balancing planning horizon vs model accuracy
  - Computational cost of planning step
  - Hyperparameter sensitivity
- **Use Cases**: 
  - Control from pixel observations
  - Sample-constrained robotics
  - Model-based benchmarks
  - Tasks requiring long-horizon planning

---

## Quick Reference Tables

### Algorithm Comparison Matrix

| Algorithm | Type | Policy | Sample Efficiency | Stability | Action Space | Complexity |
|-----------|------|--------|------------------|-----------|--------------|------------|
| **Q-Learning** | Value-Based | Off-Policy | Low | Moderate | Discrete | Low |
| **SARSA** | Value-Based | On-Policy | Low | High | Discrete | Low |
| **DQN** | Value-Based | Off-Policy | Moderate | Moderate | Discrete | Moderate |
| **REINFORCE** | Policy Gradient | On-Policy | Low | Low | Both | Low |
| **A2C** | Actor-Critic | On-Policy | Moderate | Moderate | Both | Moderate |
| **PPO** | Actor-Critic | On-Policy | Moderate | High | Both | Moderate |
| **DDPG** | Actor-Critic | Off-Policy | High | Moderate | Continuous | Moderate |
| **SAC** | Actor-Critic | Off-Policy | Very High | High | Continuous | High |
| **TRPO** | Policy Gradient | On-Policy | Moderate | Very High | Both | Very High |

### Hyperparameter Guidelines

#### Learning Rates
- **Tabular Methods**: 0.1 - 0.5
- **Deep Networks**: 1e-4 - 1e-3
- **Actor Networks**: Often 1e-4 - 1e-3
- **Critic Networks**: Often 1e-3 - 1e-2

#### Discount Factors (γ)
- **Episodic Tasks**: 0.95 - 0.99
- **Continuing Tasks**: 0.9 - 0.999
- **Short Horizons**: 0.9 - 0.95
- **Long Horizons**: 0.99 - 0.999

#### Exploration Parameters
- **Initial ε**: 0.9 - 1.0
- **Final ε**: 0.01 - 0.1
- **Decay Rate**: 0.995 - 0.9999

#### Buffer Sizes
- **Simple Tasks**: 10K - 100K
- **Complex Tasks**: 100K - 1M
- **Memory Constraints**: Adjust based on state/action dimensionality

### Common Implementation Patterns

#### Advantage Estimation (GAE)
```python
def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0 if dones[i] else values[i]
        else:
            next_value = values[i + 1]
        
        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * lambda_ * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    
    return advantages
```

#### Soft Target Updates
```python
def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
```

#### Experience Replay Buffer
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

---

## Best Practices & Tips

### Training Stability
1. **Normalize Advantages**: Subtract mean and divide by standard deviation within each batch
2. **Clip Gradients**: Prevent exploding gradients with gradient clipping (typically 0.5-1.0)
3. **Learning Rate Scheduling**: Use learning rate decay or adaptive optimizers
4. **Target Networks**: Use target networks for value-based methods
5. **Experience Replay**: Decorrelate training data for off-policy methods

### Hyperparameter Tuning
1. **Start Simple**: Begin with proven hyperparameter combinations
2. **Grid Search**: Systematic exploration of key parameters
3. **Random Search**: Often more efficient than grid search
4. **Bayesian Optimization**: For expensive hyperparameter searches
5. **Population-Based Training**: Parallel hyperparameter optimization

### Debugging Strategies
1. **Monitor Learning Curves**: Track rewards, losses, and key metrics
2. **Sanity Checks**: Test on simple environments first
3. **Implementation Verification**: Compare against reference implementations
4. **Ablation Studies**: Remove components to identify issues
5. **Gradient Monitoring**: Watch for vanishing/exploding gradients

### Environment Considerations
1. **Reward Shaping**: Design informative reward functions
2. **State Preprocessing**: Normalize observations appropriately
3. **Action Scaling**: Ensure actions are in appropriate ranges
4. **Episode Termination**: Handle terminal states correctly
5. **Stochasticity**: Account for environment randomness

---

## Advanced Topics & Extensions

### Modern Developments
- **Distributional RL**: Learn full return distributions (C51, QR-DQN)
- **Multi-Task RL**: Share knowledge across related tasks
- **Meta-Learning**: Learn to adapt quickly to new tasks
- **Offline RL**: Learn from fixed datasets without environment interaction
- **Model-Based RL**: Advanced world models and planning techniques

### Theoretical Foundations
- **Policy Gradient Theorem**: Mathematical foundation for policy optimization
- **Bellman Equations**: Core recursive relationships in RL
- **Convergence Guarantees**: Conditions under which algorithms converge
- **Regret Bounds**: Theoretical performance guarantees
- **Exploration Theory**: Principled approaches to exploration

### Practical Applications
- **Robotics**: Manipulation, locomotion, navigation
- **Game Playing**: Strategic games, real-time strategy
- **Autonomous Systems**: Self-driving cars, drones
- **Finance**: Algorithmic trading, portfolio optimization
- **Healthcare**: Treatment optimization, drug discovery

---

## Key Insights & Takeaways

> **Pro Tip**: Standardizing advantages (in policy gradient/actor-critic methods) or returns (in REINFORCE) often significantly stabilizes training. This involves subtracting the mean and dividing by the standard deviation within each batch or episode.

> **Algorithm Combinations**: Many state-of-the-art algorithms combine multiple ideas. PPO and A2C use actor-critic with policy gradients. SAC combines actor-critic with maximum entropy RL. PlaNet integrates model-based learning with CEM planning.

> **Implementation Details Matter**: Small implementation details can dramatically affect performance. Pay attention to network initialization, normalization techniques, and update frequencies.

> **Environment Design**: The choice of state representation, action space, and reward function often matters more than the specific algorithm used.

> **Sample Efficiency vs Stability Trade-off**: Off-policy methods (SAC, DDPG) are typically more sample efficient, while on-policy methods (PPO, A2C) are often more stable and easier to tune.
