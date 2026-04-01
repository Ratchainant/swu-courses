# Week 5 — Value-Based Deep Reinforcement Learning (DQN)

---

## 1. The Scalability Crisis: When Q-Tables Break Down

> *Imagine trying to memorize every possible board position in chess — all $10^{43}$ of them — and writing down a score for each one. No computer on Earth has enough memory for that. Yet AI systems play chess at superhuman level. How?*

### The Fatal Flaw of Tabular Q-Learning

In Week 4, we mastered Q-Learning — an elegant algorithm that builds a **Q-table**: a giant lookup table storing the value $Q(s, a)$ for every state-action pair. When the agent visits $(s, a)$, it looks up the table and updates that entry.

This works beautifully for small environments. But the moment we step into the real world, the Q-table becomes impossible.

**Counting states in real problems:**

| Environment | State representation | Number of states | Q-table size (4 actions) |
|-------------|---------------------|-----------------|--------------------------|
| 3-room robot (Week 1) | Room ID | 3 | 12 entries |
| GridWorld 10×10 | (row, col) | 100 | 400 entries |
| Tic-Tac-Toe | Board configuration | ~5,478 | ~22,000 entries |
| **Atari Breakout** | **84×84 grayscale pixels** | **$\approx 256^{84 \times 84} \approx 10^{17,000}$** | **Physically impossible** |
| **Autonomous driving** | **Camera + LiDAR + GPS** | **Continuous, infinite** | **Does not exist** |

This is the **curse of dimensionality**: the number of states grows exponentially with the number of dimensions describing the state. A tabular approach collapses entirely.

**Two more problems beyond memory:**

1. **Generalization:** A Q-table treats every state as completely independent. If the agent learns that position $(5, 7)$ in a grid is dangerous, it learns nothing about the nearly identical position $(5, 8)$. Every state must be learned from scratch. Humans don't work this way — we generalize from similar experiences.

2. **Continuous state spaces:** Real sensor readings (camera pixels, joint angles, velocities) are continuous. You can't have a Q-table row for "velocity = 3.14159265…" — there are infinitely many possible values.

**The solution:** Replace the Q-table with a **function approximator** — something that can take any state as input and output Q-values for all actions, even for states never seen before. The most powerful function approximator we have is a **neural network**.

---

## 2. Neural Networks as Q-Function Approximators

> *Instead of a phonebook that requires looking up a specific name, imagine a brilliant assistant who has read thousands of phonebooks and can estimate anyone's number — even people not in any book — based on patterns they've noticed. That's what a neural network does for Q-values.*

### 2.1 The Core Idea: Parameterizing Q

We replace the Q-table $Q(s, a)$ with a neural network $Q(s, a;\, \boldsymbol{\theta})$, where $\boldsymbol{\theta}$ represents all the learnable weights and biases in the network.

| Aspect | Tabular Q-Learning | Deep Q-Learning |
|--------|-------------------|----------------|
| Q stored as | A lookup table | A neural network with weights $\boldsymbol{\theta}$ |
| Query | Look up row $(s,a)$ | Forward pass: network takes $s$ as input |
| Update | Change one table entry | Backpropagation — shift all weights slightly |
| Generalizes? | ✗ Never (each state independent) | ✓ Yes (similar states → similar outputs) |
| Handles continuous states? | ✗ No | ✓ Yes |
| Memory required | $|\mathcal{S}| \times |\mathcal{A}|$ entries | Fixed by network architecture |

**The generalization power** is transformative. Once the network has seen a car swerving left on a rainy road, it can estimate Q-values for a slightly different car swerving left on a slightly rainier road — because the network has learned the underlying *features* that matter (swerving, rain, speed), not just memorized specific configurations.

### 2.2 Network Architecture for Atari (The Original DQN)

The 2013/2015 DeepMind DQN paper used this architecture for Atari games:

```
Input: 4 stacked 84×84 grayscale frames
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  Conv Layer 1: 32 filters, 8×8 kernel, stride 4    │  → Detects edges, basic shapes
  │  ReLU activation                                    │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  Conv Layer 2: 64 filters, 4×4 kernel, stride 2    │  → Detects objects, motion
  │  ReLU activation                                    │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  Conv Layer 3: 64 filters, 3×3 kernel, stride 1    │  → Combines features
  │  ReLU activation                                    │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  Flatten → Fully Connected Layer: 512 units         │  → High-level reasoning
  │  ReLU activation                                    │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  Output Layer: N units (one per action)             │  → Q(s, a₁), Q(s, a₂), ..., Q(s, aₙ)
  │  Linear activation (no squashing)                   │
  └─────────────────────────────────────────────────────┘
```

**Why output ALL Q-values at once?** A single forward pass produces Q-values for every possible action simultaneously. This is much more efficient than running a separate forward pass for each action.

**Why 4 stacked frames?** A single frame can't tell you about velocity or direction. Stacking 4 consecutive frames gives the network temporal information: the ball moved from left to right between frame 1 and frame 4, so it's moving rightward. This encodes motion without explicitly designing motion features.

### 2.3 A Minimal Network: Worked Forward Pass

To make neural network computation concrete, let's use a tiny network with actual numbers.

**Problem:** A robot has 2 state features: $[\text{distance to goal}, \text{speed}]$. It has 2 actions: `Accelerate` and `Brake`.

**Network architecture:** 2 inputs → 3 hidden units → 2 outputs (one Q-value per action)

**Weights (Layer 1 — Input to Hidden):**

$$W^{(1)} = \begin{bmatrix} 0.5 & -0.3 & 0.8 \\ -0.2 & 0.6 & 0.1 \end{bmatrix}, \quad \boldsymbol{b}^{(1)} = \begin{bmatrix} 0.1 \\ -0.1 \\ 0.2 \end{bmatrix}$$

Dimensions: $W^{(1)}$ is $2 \times 3$ (2 inputs → 3 hidden units); each row is one input feature.

**Weights (Layer 2 — Hidden to Output):**

$$W^{(2)} = \begin{bmatrix} 0.4 & 0.7 \\ -0.5 & 0.3 \\ 0.6 & -0.2 \end{bmatrix}, \quad \boldsymbol{b}^{(2)} = \begin{bmatrix} 0.0 \\ 0.1 \end{bmatrix}$$

Dimensions: $W^{(2)}$ is $3 \times 2$ (3 hidden units → 2 output Q-values).

**Input state:** $\boldsymbol{s} = [3.0,\; 1.5]^T$ (distance = 3.0 m, speed = 1.5 m/s)

---

**Step 1 — Compute hidden layer pre-activations** $\boldsymbol{z}^{(1)} = W^{(1)T} \boldsymbol{s} + \boldsymbol{b}^{(1)}$:

$$z^{(1)}_1 = (0.5)(3.0) + (-0.2)(1.5) + 0.1 = 1.5 - 0.3 + 0.1 = \mathbf{1.3}$$
$$z^{(1)}_2 = (-0.3)(3.0) + (0.6)(1.5) + (-0.1) = -0.9 + 0.9 - 0.1 = \mathbf{-0.1}$$
$$z^{(1)}_3 = (0.8)(3.0) + (0.1)(1.5) + 0.2 = 2.4 + 0.15 + 0.2 = \mathbf{2.75}$$

**Step 2 — Apply ReLU activation** $h_i = \max(0, z^{(1)}_i)$:

$$\boldsymbol{h} = [\max(0, 1.3),\; \max(0, -0.1),\; \max(0, 2.75)] = [\mathbf{1.3},\; \mathbf{0.0},\; \mathbf{2.75}]$$

The second hidden unit was negative, so ReLU clips it to 0 — it "fires" only on certain input patterns.

**Step 3 — Compute output layer** $\boldsymbol{Q} = W^{(2)T} \boldsymbol{h} + \boldsymbol{b}^{(2)}$:

$$Q(\text{Accelerate}) = (0.4)(1.3) + (-0.5)(0.0) + (0.6)(2.75) + 0.0 = 0.52 + 0 + 1.65 = \mathbf{2.17}$$
$$Q(\text{Brake}) = (0.7)(1.3) + (0.3)(0.0) + (-0.2)(2.75) + 0.1 = 0.91 + 0 - 0.55 + 0.1 = \mathbf{0.46}$$

**Result:** The network outputs $Q(s, \text{Accelerate}) = 2.17$ and $Q(s, \text{Brake}) = 0.46$.

**Greedy action:** $\arg\max = \text{Accelerate}$ ✓ (the robot is still far from the goal and not moving fast — accelerating makes sense).

This single forward pass takes ~microseconds and generalizes to any $(distance,\; speed)$ combination without re-training.

---

## 3. The Deadly Triad: Why Naïve Deep Q-Learning Fails

> *Combining Q-Learning with neural networks sounds straightforward — just replace the table with a network. But practitioners discovered this "obvious" approach almost never works. Three forces conspire to destabilize training.*

If you take Q-Learning from Week 4 and directly replace the Q-table with a neural network, training typically **diverges** — the loss explodes, the Q-values go to infinity, or the agent's performance collapses catastrophically. This is known as the **deadly triad**.

### 3.1 Problem 1 — Correlated Experience (The "Echo Chamber")

**What goes wrong:** Q-Learning processes experience sequentially — step 1, step 2, step 3, … Each update depends on the previous one. Consecutive experiences are highly correlated (they all come from the same trajectory in the same part of the environment).

**Analogy:** Imagine learning to cook by watching one chef make only pasta for 1,000 hours, then only soup for 1,000 hours. Your neural network for "cooking" would first become excellent at pasta and forget everything else, then excellent at soup and forget pasta. It would constantly overwrite what it just learned.

In deep RL, if the agent is exploring one room of a maze, all experiences are from that room. The network overfits to that room, its weights shift dramatically, and it "forgets" the rest of the maze — a phenomenon called **catastrophic forgetting**.

**Why correlation hurts:** Gradient descent assumes training samples are drawn i.i.d. (independently and identically distributed). Sequential RL experience violates this assumption severely, causing unstable, oscillating updates.

### 3.2 Problem 2 — Non-Stationary Targets (The "Moving Goalpost")

**What goes wrong:** In standard supervised learning, you have fixed labels $y$. In Q-Learning, the learning target $r + \gamma \max_{a'} Q(s', a';\, \boldsymbol{\theta})$ depends on the **same network** being trained. Every time you update $\boldsymbol{\theta}$, the target changes too.

**Analogy:** Imagine learning to hit a dart bullseye, but the bullseye moves every time you throw a dart. You aim where it was, throw, and by the time the dart arrives, the target has shifted. You'd never converge — you'd just chase the moving target forever.

Mathematically: we update $\boldsymbol{\theta}$ to minimize $[y - Q(s, a;\, \boldsymbol{\theta})]^2$ where $y = r + \gamma \max_{a'} Q(s', a';\, \boldsymbol{\theta})$. But $y$ itself changes with $\boldsymbol{\theta}$ — we're chasing our own tail.

### 3.3 Problem 3 — Overestimation Bias (From Maximization)

As noted in Week 4, the $\max$ operator over noisy Q-value estimates causes systematic overestimation. In tabular Q-Learning, this bias is modest. With neural networks that have millions of parameters and high-dimensional outputs, the overestimation can explode and destabilize training.

---

**DeepMind's 2013/2015 breakthrough with DQN solved all three problems.** The solutions are elegant, and each one is worth understanding deeply.

---

## 4. DQN: The Two Breakthrough Innovations

### 4.1 Innovation 1 — Experience Replay

> *A medical student doesn't only study cases encountered today. They review their entire case history — randomly revisiting past patients, shuffling the order, learning from diverse experiences simultaneously. Experience Replay does the same for the agent.*

**The mechanism:**

DQN maintains a **Replay Buffer** (also called Experience Replay Memory) — a large circular queue storing past transitions:

$$\mathcal{D} = \{(s_1, a_1, r_1, s'_1, \text{done}_1),\; (s_2, a_2, r_2, s'_2, \text{done}_2),\; \ldots\}$$

The buffer has a fixed capacity (e.g., 1,000,000 transitions). When full, the oldest transitions are overwritten.

**During training:** Instead of learning from the most recent transition only, DQN randomly samples a **mini-batch** of $N$ transitions from $\mathcal{D}$ (e.g., $N = 32$) and performs one gradient descent step on that batch.

```
Every step:
  1. Agent acts → observe (s, a, r, s', done)
  2. Store in replay buffer D
  3. If |D| > batch_size:
       Sample random mini-batch of N transitions from D
       Compute targets and loss
       Gradient descent step on network weights θ
```

**How it solves Problem 1 (Correlated Experience):**

Random sampling from the buffer breaks the temporal correlation. A single mini-batch might contain experience from 10 different games, 5 different rooms of the maze, and 3 different time periods — the network sees diverse, uncorrelated examples in every update, just like i.i.d. supervised learning.

**Bonus — Sample Efficiency:**

Each transition can be reused multiple times for training (replayed many times). This is crucial when experience is expensive to collect (e.g., physical robots, slow simulations).

**Buffer size matters:**

| Buffer too small | Buffer too large |
|-----------------|-----------------|
| Recent transitions dominate → correlated | Memory issues; too much stale data from early (bad) policy |
| Fast to sample | Slower to sample |
| **Typical sweet spot: 100K–1M transitions** | |

### 4.2 Innovation 2 — Target Network

> *A ship navigator uses a fixed star to plot a course — not a star that moves every time they glance up. The Target Network gives DQN a stable reference point, updated only occasionally.*

**The mechanism:**

DQN maintains **two identical networks** with the same architecture:

| Network | Symbol | Update frequency | Role |
|---------|--------|-----------------|------|
| **Online network** (main) | $Q(s, a;\, \boldsymbol{\theta})$ | Every training step | Selects actions; parameters actively updated via gradient descent |
| **Target network** | $\hat{Q}(s, a;\, \boldsymbol{\theta}^-)$ | Every $C$ steps (e.g., every 1,000 steps) | Computes the learning target; parameters frozen between updates |

The target for training is computed using the **target network**, not the online network:

$$y_t = r_{t+1} + \gamma \max_{a'} \hat{Q}(s_{t+1}, a';\, \boldsymbol{\theta}^-)$$

The loss is then:

$$\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \bigl( y_t - Q(s, a;\, \boldsymbol{\theta}) \bigr)^2 \right]$$

Every $C$ steps, the target network is synchronized: $\boldsymbol{\theta}^- \leftarrow \boldsymbol{\theta}$ (hard update) or gradually blended (soft update: $\boldsymbol{\theta}^- \leftarrow \tau \boldsymbol{\theta} + (1-\tau)\boldsymbol{\theta}^-$, where $\tau \ll 1$).

**How it solves Problem 2 (Non-Stationary Targets):**

Because $\boldsymbol{\theta}^-$ is frozen for $C$ steps, the target $y_t$ is **stable** during that window. The online network can make consistent progress toward a fixed goal, instead of chasing a moving one. After $C$ steps, we "refresh" the goalpost — but it only moves occasionally, not continuously.

**Analogy refined:** It's like a student taking a practice exam every week, using last week's answer key ($\boldsymbol{\theta}^-$) to evaluate their work. The answer key doesn't change mid-exam — it's fixed. Only after the week ends do we update it with new knowledge.

---

## 5. The Complete DQN Algorithm

With both innovations combined, we can now write the full DQN algorithm:

```
╔══════════════════════════════════════════════════════════════════╗
║                     DQN Algorithm                                ║
╠══════════════════════════════════════════════════════════════════╣
║ Initialize:                                                      ║
║   Online network Q(s,a; θ) with random weights θ                ║
║   Target network Q̂(s,a; θ⁻) with θ⁻ ← θ                       ║
║   Replay buffer D (empty, capacity N)                            ║
║   ε ← ε_start (e.g., 1.0)                                       ║
║                                                                  ║
║ For each episode:                                                ║
║   Reset environment → get initial state s                        ║
║                                                                  ║
║   For each time step t:                                          ║
║     ① SELECT ACTION (ε-greedy):                                  ║
║       With prob ε:  a ← random action                           ║
║       Otherwise:    a ← argmax_a Q(s, a; θ)                     ║
║                                                                  ║
║     ② EXECUTE & OBSERVE:                                         ║
║       Take action a → observe r, s', done                       ║
║                                                                  ║
║     ③ STORE in replay buffer:                                    ║
║       D ← D ∪ {(s, a, r, s', done)}                             ║
║                                                                  ║
║     ④ SAMPLE mini-batch from D:                                  ║
║       {(sⱼ, aⱼ, rⱼ, s'ⱼ, doneⱼ)} ~ Uniform(D), j = 1..B      ║
║                                                                  ║
║     ⑤ COMPUTE TARGETS:                                           ║
║       yⱼ = rⱼ                          if doneⱼ = True          ║
║       yⱼ = rⱼ + γ · max_a' Q̂(s'ⱼ,a';θ⁻)  if doneⱼ = False    ║
║                                                                  ║
║     ⑥ COMPUTE LOSS & UPDATE θ:                                   ║
║       L(θ) = (1/B) Σⱼ (yⱼ - Q(sⱼ, aⱼ; θ))²                   ║
║       θ ← θ - α · ∇_θ L(θ)   [gradient descent]               ║
║                                                                  ║
║     ⑦ UPDATE TARGET NETWORK every C steps:                       ║
║       θ⁻ ← θ                                                    ║
║                                                                  ║
║     ⑧ DECAY ε:                                                   ║
║       ε ← max(ε_min, ε · ε_decay)                               ║
║                                                                  ║
║     s ← s'                                                       ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 6. Worked Calculation: One DQN Training Step

Let's trace through a single complete training step with concrete numbers, using our 2-input → 3-hidden → 2-output toy network.

**Setup:**
- State features: $s = [\text{distance}, \text{speed}]$
- Actions: $\{0 = \text{Accelerate},\; 1 = \text{Brake}\}$
- $\gamma = 0.9$, learning rate $\alpha = 0.01$
- Online network weights: $\boldsymbol{\theta}$ (same as Section 2.3 forward pass)
- Target network weights: $\boldsymbol{\theta}^-$ (identical to $\boldsymbol{\theta}$ at this moment)

**Mini-batch of 3 transitions sampled from replay buffer:**

| $j$ | $s_j$ | $a_j$ | $r_j$ | $s'_j$ | done |
|-----|--------|--------|--------|---------|------|
| 1 | $[3.0, 1.5]$ | Accelerate (0) | $-1$ | $[2.5, 2.0]$ | False |
| 2 | $[0.5, 0.2]$ | Brake (1) | $+10$ | $[0.0, 0.0]$ | True |
| 3 | $[1.0, 3.0]$ | Accelerate (0) | $-1$ | $[0.8, 3.2]$ | False |

---

### Step A — Forward Pass on Online Network (to get predicted Q-values)

We already computed $Q([3.0, 1.5];\, \boldsymbol{\theta})$ in Section 2.3:
$$Q(s_1, \text{Accelerate}) = 2.17, \quad Q(s_1, \text{Brake}) = 0.46$$

For transition 1: the agent took `Accelerate`, so the **predicted Q-value** is:
$$Q(s_1, a_1;\, \boldsymbol{\theta}) = Q([3.0, 1.5], \text{Accelerate}) = \mathbf{2.17}$$

*(For a full training step we would compute forward passes for all 3 samples; we focus on sample 1 for clarity.)*

---

### Step B — Forward Pass on Target Network (to get next-state values)

Use the **target network** $\hat{Q}$ to compute $\max_{a'} \hat{Q}(s'_j, a';\, \boldsymbol{\theta}^-)$ for each transition.

**Transition 1:** $s'_1 = [2.5, 2.0]$, done = False

Compute hidden layer pre-activations using the same weights (since $\boldsymbol{\theta}^- = \boldsymbol{\theta}$ here):

$$z^{(1)}_1 = (0.5)(2.5) + (-0.2)(2.0) + 0.1 = 1.25 - 0.4 + 0.1 = 0.95$$
$$z^{(1)}_2 = (-0.3)(2.5) + (0.6)(2.0) + (-0.1) = -0.75 + 1.2 - 0.1 = 0.35$$
$$z^{(1)}_3 = (0.8)(2.5) + (0.1)(2.0) + 0.2 = 2.0 + 0.2 + 0.2 = 2.4$$

Apply ReLU: $\boldsymbol{h}' = [0.95,\; 0.35,\; 2.4]$ (all positive, none clipped)

Output layer:
$$\hat{Q}(s'_1, \text{Accelerate}) = (0.4)(0.95) + (-0.5)(0.35) + (0.6)(2.4) + 0.0 = 0.38 - 0.175 + 1.44 = \mathbf{1.645}$$
$$\hat{Q}(s'_1, \text{Brake}) = (0.7)(0.95) + (0.3)(0.35) + (-0.2)(2.4) + 0.1 = 0.665 + 0.105 - 0.48 + 0.1 = \mathbf{0.39}$$

$$\max_{a'} \hat{Q}(s'_1, a') = \max(1.645, 0.39) = \mathbf{1.645}$$

**Transition 2:** $s'_2 = [0.0, 0.0]$, done = **True** → next state value = 0 (terminal)

**Transition 3:** $s'_3 = [0.8, 3.2]$, done = False — (we would compute similarly; suppose $\max_{a'} \hat{Q}(s'_3, a') = 1.20$)

---

### Step C — Compute Targets $y_j$

$$y_1 = r_1 + \gamma \max_{a'} \hat{Q}(s'_1, a') = -1 + 0.9 \times 1.645 = -1 + 1.481 = \mathbf{0.481}$$
$$y_2 = r_2 = +10 \quad \text{(terminal — no future value)} = \mathbf{10.0}$$
$$y_3 = r_3 + \gamma \max_{a'} \hat{Q}(s'_3, a') = -1 + 0.9 \times 1.20 = -1 + 1.08 = \mathbf{0.08}$$

---

### Step D — Compute the Loss

We need the predicted Q-values $Q(s_j, a_j;\, \boldsymbol{\theta})$ for each transition (using the **online** network):

| $j$ | Predicted $Q(s_j, a_j)$ | Target $y_j$ | TD error $\delta_j = y_j - Q$ | $\delta_j^2$ |
|-----|------------------------|-------------|------------------------------|-------------|
| 1 | 2.17 | 0.481 | $0.481 - 2.17 = -1.689$ | 2.852 |
| 2 | (suppose) 0.85 | 10.0 | $10.0 - 0.85 = +9.15$ | 83.72 |
| 3 | (suppose) 1.95 | 0.08 | $0.08 - 1.95 = -1.87$ | 3.497 |

**Mean Squared Error (MSE) Loss:**

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{3} \sum_{j=1}^{3} \delta_j^2 = \frac{2.852 + 83.72 + 3.497}{3} = \frac{90.07}{3} = \mathbf{30.02}$$

**Observation:** The loss is dominated by transition 2 — the terminal state with a huge reward (+10) that the network didn't anticipate (predicted only 0.85). This large TD error produces a large gradient, causing weights to shift substantially toward better predicting high-reward terminal states.

---

### Step E — Gradient Descent Update

Backpropagation computes $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ — how much to adjust each weight to reduce the loss. Then:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$$

For the output layer weights connecting hidden unit $i$ to the `Accelerate` output, the gradient for transition 1 is:

$$\frac{\partial \mathcal{L}_1}{\partial W^{(2)}_{i,\,\text{Acc}}} = 2 \times (Q(s_1, \text{Acc}) - y_1) \times h_i = 2 \times (2.17 - 0.481) \times h_i = 2 \times 1.689 \times h_i$$

For hidden unit 1 ($h_1 = 1.3$):

$$\frac{\partial \mathcal{L}_1}{\partial W^{(2)}_{1,\,\text{Acc}}} = 2 \times 1.689 \times 1.3 = \mathbf{4.391}$$

Update: $W^{(2)}_{1,\,\text{Acc}} \leftarrow 0.4 - 0.01 \times 4.391 = 0.4 - 0.044 = \mathbf{0.356}$

The weight decreased (the network predicted too high a Q-value for Accelerate in state $s_1$; now it will predict slightly lower — correcting toward the target 0.481).

**This propagates through the entire network via the chain rule**, adjusting all weights to make the network's predictions more accurate on this mini-batch.

---

## 7. DQN Hyperparameters in Practice

The following table summarizes the key hyperparameters and their typical values in the original Atari DQN:

| Hyperparameter | Typical Value | Effect of increasing |
|----------------|--------------|---------------------|
| Replay buffer size | 1,000,000 | More diverse experience; more memory |
| Mini-batch size $B$ | 32 | More stable gradients; slower per step |
| Target update freq $C$ | 1,000–10,000 steps | More stable targets; slower adaptation |
| Learning rate $\alpha$ | 0.00025 | Slower but more stable learning |
| Discount $\gamma$ | 0.99 | More forward-looking; slower convergence |
| $\varepsilon$ start | 1.0 | Always start with full exploration |
| $\varepsilon$ end | 0.01–0.1 | Less exploration in converged policy |
| $\varepsilon$ decay duration | 1M steps | Slower decay → more thorough exploration |
| Replay start size | 50,000 | Buffer pre-filled before learning starts |

**Replay start size is critical:** DQN doesn't begin gradient updates until the replay buffer has accumulated at least 50,000 transitions. This ensures the network never trains on a nearly empty, highly correlated buffer during the first episodes.

---

## 8. DQN Variants: Making It Even Better

> *DQN was just the beginning. Within a few years, researchers identified specific weaknesses and developed elegant solutions, each adding a small but powerful fix.*

### 8.1 Double DQN (DDQN) — Fixing Overestimation

**Problem:** DQN uses the same network to both *select* the best action ($\arg\max$) and *evaluate* how good it is. This conflation causes systematic overestimation — especially in early training when Q-values are noisy.

**Solution (van Hasselt et al., 2015):** Decouple selection from evaluation:

$$y^{\text{DDQN}} = r + \gamma \hat{Q}\!\left(s',\; \underbrace{\arg\max_{a'} Q(s', a';\, \boldsymbol{\theta})}_{\text{online network selects action}};\; \boldsymbol{\theta}^-\right)$$

- The **online network** $\boldsymbol{\theta}$ selects which action is best at $s'$
- The **target network** $\boldsymbol{\theta}^-$ evaluates how good that action is

By using different networks for selection and evaluation, the systematic optimism cancels out. DDQN consistently outperforms DQN with the same compute budget.

**Analogy:** Don't ask the same person who suggested a restaurant to also rate how good it was. Ask a different reviewer to evaluate the suggestion. Separating these roles removes self-serving bias.

### 8.2 Dueling DQN — Decomposing Q into Value and Advantage

**Insight (Wang et al., 2015):** In many states, the choice of action matters very little — all actions lead to roughly the same outcome. For example, when a car is moving straight on an empty highway, whether you slightly accelerate or slightly brake barely changes the outcome. Only in critical states (intersection, sudden obstacle) does the action choice matter enormously.

**The Advantage Function:** Define the **advantage** of action $a$ in state $s$:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

This measures: *"How much better is action $a$ compared to the average action in state $s$?"*

The Q-value can be decomposed as: $Q(s, a) = V(s) + A(s, a)$

**Dueling architecture:** Instead of one output head producing Q-values directly, use **two separate heads** — one for $V(s)$ and one for $A(s, a)$ — then combine them:

```
                    ┌──────────────────┐
                    │  Shared Layers   │
                    │  (convolutional) │
                    └─────────┬────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────────┐
    │  Value stream    │           │   Advantage stream   │
    │  FC → V(s): 1    │           │   FC → A(s,a): |A|   │
    └─────────┬────────┘           └──────────┬───────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
                  Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]
```

**Why subtract the mean of A?** Without this, $V$ and $A$ aren't uniquely determined — the decomposition is ambiguous. Subtracting the mean forces $A$ to be zero-mean, making $V$ represent the true state value.

**Why it helps:** The value stream can quickly learn that some states are fundamentally good or bad regardless of action. The advantage stream only needs to learn the relative ordering of actions. Both streams learn faster because each focuses on what it's good at.

### 8.3 Prioritized Experience Replay (PER) — Smarter Sampling

**Problem:** Standard Experience Replay samples uniformly at random. But not all transitions are equally informative. A transition with a large TD error (the network was very wrong about it) is much more valuable for learning than a transition the network has already mastered.

**Solution (Schaul et al., 2015):** Sample transitions with probability proportional to their **TD error magnitude**:

$$P(j) = \frac{|\delta_j|^\alpha}{\sum_k |\delta_k|^\alpha}$$

Where:
- $|\delta_j| = |y_j - Q(s_j, a_j;\, \boldsymbol{\theta})|$ is the absolute TD error of transition $j$
- $\alpha \in [0, 1]$ controls how much prioritization is applied ($\alpha = 0$ = uniform, $\alpha = 1$ = fully prioritized)

**Importance Sampling correction:** Prioritized sampling introduces a bias (high-error transitions are seen more often than they "should" be). This is corrected by multiplying each gradient by an **importance sampling weight**:

$$w_j = \left(\frac{1}{N \cdot P(j)}\right)^\beta$$

where $\beta$ starts at 0.4 and anneals to 1.0 over training. This down-weights the frequently sampled transitions so their gradients don't dominate.

**Intuition:** A student who keeps failing certain types of problems should practice those specific types more — not just practice random problems with equal probability. PER gives more "practice time" to the examples where the network is still confused.

### 8.4 Rainbow — All in One

In 2017, DeepMind published **Rainbow** — a DQN agent combining 6 improvements simultaneously:

| Component | What it fixes |
|-----------|--------------|
| Double DQN | Overestimation bias |
| Prioritized replay | Suboptimal sample selection |
| Dueling networks | Conflation of V and A |
| Multi-step returns (n-step) | Slow credit assignment |
| Distributional RL (C51) | Only learning expected value, not full distribution |
| Noisy Networks | Heuristic ε-greedy exploration |

**Rainbow achieved state-of-the-art on 57 Atari games**, significantly outperforming any single component — demonstrating these improvements are largely orthogonal and complementary.

---

## 9. Training Dynamics and Stability

### 9.1 The Learning Curve

A typical DQN training curve passes through four recognizable phases:

```
Average score per episode
      │
  800 │                                          ▁▃▅▇█████████
  600 │                                      ▁▃▅▇
  400 │                               ▁▁▃▄▆▇
  200 │                     ▁▁▂▂▃▄▅▆
    0 │─────────────────────────────────────────────────── Episodes
      0      100K     500K      1M       2M       3M
      │        │        │        │
   Random   Replay    Q-values  Policy
   policy   filling   improving stabilizes
   (ε=1.0)  buffer    (ε decaying)
```

**Phase 1 (Random):** $\varepsilon = 1$, all actions random, replay buffer filling.

**Phase 2 (Warming up):** Replay buffer reaches minimum size, gradient updates begin — but Q-values are still noisy, performance barely improves.

**Phase 3 (Rapid improvement):** Q-values become meaningful, ε is decreasing, the policy becomes increasingly exploitative — performance rises sharply.

**Phase 4 (Convergence):** Q-values stabilize, $\varepsilon$ at minimum, performance plateaus near the optimal level.

### 9.2 Reward Clipping and Preprocessing

In the original Atari DQN, several preprocessing steps were critical for stability:

**Reward clipping:** All rewards clipped to $[-1, +1]$. This prevents rare large rewards (some Atari games give +100 for clearing a level) from producing catastrophically large gradients that destabilize training.

**Grayscaling:** Color information is rarely important in Atari games. Converting to grayscale reduces input dimensionality from $84 \times 84 \times 3$ to $84 \times 84 \times 1$ without losing critical information.

**Frame skipping:** The agent repeats the same action for 4 consecutive frames. Games are often 60 fps, but state changes meaningful enough to require a new decision happen much less frequently. This speeds up training by 4×.

**Huber loss (Smooth L1):** Instead of pure MSE, DQN often uses the Huber loss:

$$\mathcal{L}_{\text{Huber}}(\delta) = \begin{cases} \frac{1}{2}\delta^2 & \text{if } |\delta| \leq 1 \\ |\delta| - \frac{1}{2} & \text{if } |\delta| > 1 \end{cases}$$

For small errors, it behaves like MSE (smooth, good gradients). For large errors, it behaves like MAE (linear, not explosive). This prevents huge TD errors from causing catastrophic weight updates.

---

## 10. The Atari Result: Why It Matters

In 2013/2015, DeepMind published results that shocked the AI world: a single DQN agent, trained with the **same architecture and hyperparameters**, achieved **superhuman performance on 29 out of 49 Atari games** — learning directly from raw pixels, with no hand-crafted features, no domain knowledge, and no game-specific tuning.

**What made this remarkable:**

| Criterion | Traditional AI | DQN |
|-----------|---------------|-----|
| State representation | Hand-designed features by experts | Raw 84×84 pixels — no manual engineering |
| Algorithm per game | Custom, game-specific | Single universal algorithm |
| Domain knowledge required | Extensive | Zero — same code for all 49 games |
| Performance | Expert-tuned | Exceeded human level on 29/49 games |

**Games DQN mastered:** Breakout, Pong, Space Invaders, Enduro, Beam Rider (well above human).

**Games DQN struggled with:** Montezuma's Revenge (requires long-horizon planning and exploration), Pitfall (sparse reward, very long credit assignment). These became targets for future research (curiosity-driven exploration, hierarchical RL).

**Historical significance:** DQN demonstrated for the first time that a single algorithm could learn to play diverse games at superhuman level from raw sensory input — a landmark step toward artificial general intelligence.

---

## 11. Limitations of Value-Based Deep RL

Understanding what DQN *cannot* do motivates the remaining weeks of the course:

| Limitation | Description | Solution (future weeks) |
|------------|-------------|------------------------|
| **Discrete actions only** | $\arg\max_a Q(s,a)$ requires enumerating all actions — impossible with continuous actions (robot joint torques, steering angles) | Policy Gradient methods (Week 6) |
| **Deterministic policy** | DQN's greedy policy is deterministic — can't represent "50% Left, 50% Right" strategies | Stochastic policy gradient (Week 6) |
| **Sample inefficiency** | Needs millions of steps to converge — impractical for physical robots | Model-based RL, imitation learning |
| **Reward shaping sensitivity** | Performance highly sensitive to reward design | Inverse RL, RLHF (Week 10) |
| **Partial observability** | Standard DQN assumes full state information; real sensors are noisy | Recurrent architectures (DRQN) |

---

## 12. Real-World Applications

| Application | What's used | Key insight |
|-------------|------------|-------------|
| **Atari game mastery** (DeepMind 2015) | DQN | First proof that one algorithm can master diverse tasks from pixels |
| **AlphaGo (2016)** | DQN + Policy Gradient + MCTS | Q-values guide tree search in Go |
| **Chip design** (Google, 2021) | DQN variants | Floorplan arrangement for TPU chips; trained in simulation |
| **Data center cooling** (DeepMind) | DQN | Reduced Google's data center cooling energy by 40% |
| **Drug discovery** | DQN-style agents | Navigate molecular space — atoms are "actions," molecular properties are Q-values |
| **Traffic signal control** | DQN | Minimizes vehicle wait times by learning signal timing from traffic sensor data |
| **Dialogue management** | DQN | Decide what to say next in a conversation (state = dialogue history, actions = possible utterances) |
| **Robotic manipulation** (simulation) | DQN + Dueling + PER | Learns grasping policies in simulation before hardware transfer |

---

## 13. Summary

| Concept | Key Idea |
|---------|----------|
| **Curse of dimensionality** | Tabular Q-tables are impossible for large/continuous state spaces |
| **Function approximation** | Replace the Q-table with $Q(s, a;\, \boldsymbol{\theta})$ — a neural network |
| **Generalization** | Neural networks estimate Q-values for states never seen before, using learned features |
| **Correlated experience** | Sequential transitions are highly correlated — violates i.i.d. assumption of gradient descent |
| **Non-stationary targets** | Q-Learning targets depend on the same network being trained — the goalpost moves |
| **Experience Replay** | Store transitions in a buffer; train on random mini-batches — breaks correlations |
| **Target Network** | A frozen copy of the network; provides stable targets; updated every $C$ steps |
| **DQN loss** | $\mathcal{L} = \mathbb{E}[(r + \gamma \max_{a'}\hat{Q}(s',a';\theta^-) - Q(s,a;\theta))^2]$ |
| **Double DQN** | Decouple action selection ($\boldsymbol{\theta}$) from action evaluation ($\boldsymbol{\theta}^-$) — reduces overestimation |
| **Dueling DQN** | Decompose $Q = V + A$ — two streams learn state value and action advantage separately |
| **Prioritized Replay** | Sample transitions by TD error magnitude — focus learning where the network is most wrong |
| **Rainbow** | Combines 6 DQN improvements for state-of-the-art performance |
| **Reward clipping** | Clip rewards to $[-1, +1]$ for training stability |
| **Huber loss** | Smooth L1 loss — behaves like MSE for small errors, MAE for large errors |
| **DQN limitation** | Requires discrete actions; policy is deterministic → motivates Policy Gradient (Week 6) |

---

## 14. Exercises

### Conceptual Questions

1. **Explain the "deadly triad"** — what are the three forces that make naïve deep Q-Learning unstable? For each one, name the specific DQN innovation that addresses it.

2. **Experience Replay has two benefits.** One is breaking temporal correlation. What is the second benefit? Why is it especially valuable in settings where each environment step is costly (e.g., physical robots)?

3. **The target network is updated every $C$ steps.** What would happen if $C$ were set to 1 (target network updated every step)? What if $C$ were set to 1,000,000 (target network barely ever updated)? What does this suggest about the optimal choice of $C$?

4. **In the Dueling DQN architecture**, why can't we simply define $Q(s,a) = V(s) + A(s,a)$ without subtracting the mean of $A$? (Hint: consider two different decompositions — e.g., $V(s) = 5$, $A(s,a_1) = 3$ versus $V(s) = 7$, $A(s,a_1) = 1$ — that give the same Q-value. Why is this ambiguity a problem for learning?)

5. **DQN uses frame stacking** (4 consecutive frames as input). Why is a single frame insufficient? Give a concrete example of a situation where you cannot determine the correct action from a single frame alone.

---

### Calculation Problems

**Problem 1 — Neural Network Forward Pass**

A simple network with 2 inputs, 2 hidden units (ReLU), and 2 outputs has the following weights:

$$W^{(1)} = \begin{bmatrix} 1.0 & -1.0 \\ 0.5 & 2.0 \end{bmatrix}, \quad \boldsymbol{b}^{(1)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$W^{(2)} = \begin{bmatrix} 2.0 & -1.0 \\ 1.0 & 3.0 \end{bmatrix}, \quad \boldsymbol{b}^{(2)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Input state: $\boldsymbol{s} = [1.0,\; 2.0]^T$

(a) Compute the hidden layer pre-activations $z_1$ and $z_2$.

(b) Apply ReLU: compute $h_1$ and $h_2$.

(c) Compute the output Q-values $Q(s, a_0)$ and $Q(s, a_1)$.

(d) What is the greedy action?

---

**Problem 2 — DQN Target Computation**

After a forward pass on the **target network** for $s' = [0.5,\; 1.0]^T$, using the same network weights as Problem 1, you obtain: $\hat{Q}(s', a_0) = 0.75$ and $\hat{Q}(s', a_1) = 2.10$.

Parameters: $r = +1$, $\gamma = 0.9$, done = False.

(a) What is the Q-Learning target $y$?

(b) Using your answer from Problem 1(c), what is the TD error $\delta$ for action $a_0$?

(c) What is the squared TD error (single-sample loss) for this transition?

---

**Problem 3 — Experience Replay Mini-Batch**

Three transitions are sampled from the replay buffer:

| $j$ | $Q(s_j, a_j;\theta)$ (online) | $y_j$ (target) |
|-----|-------------------------------|----------------|
| 1 | 3.5 | 5.0 |
| 2 | 7.2 | 2.1 |
| 3 | 1.0 | 1.8 |

(a) Compute the TD error $\delta_j$ for each transition.

(b) Compute the MSE loss over the mini-batch.

(c) In Prioritized Experience Replay with $\alpha = 1$, compute the unnormalized sampling priority $|\delta_j|^\alpha$ for each transition and identify which transition would be most likely to be sampled next.

---

**Problem 4 — Double DQN vs. Standard DQN**

At state $s'$, the online network predicts: $Q(s', a_0;\, \boldsymbol{\theta}) = 3.2$ and $Q(s', a_1;\, \boldsymbol{\theta}) = 5.8$.

The target network predicts: $\hat{Q}(s', a_0;\, \boldsymbol{\theta}^-) = 2.1$ and $\hat{Q}(s', a_1;\, \boldsymbol{\theta}^-) = 4.5$.

Given $r = 0$, $\gamma = 0.9$, done = False:

(a) Compute the **standard DQN** target $y^{\text{DQN}}$.

(b) Compute the **Double DQN** target $y^{\text{DDQN}}$.

(c) If the true optimal $Q^*(s, a) = 4.0$, which target is more accurate? What does this illustrate?

---

### Answer Key

**Problem 1:**

(a) $z_1 = (1.0)(1.0) + (0.5)(2.0) = 1.0 + 1.0 = \mathbf{2.0}$
$\quad z_2 = (-1.0)(1.0) + (2.0)(2.0) = -1.0 + 4.0 = \mathbf{3.0}$

(b) $h_1 = \max(0, 2.0) = \mathbf{2.0}$, $\quad h_2 = \max(0, 3.0) = \mathbf{3.0}$

(c) $Q(s, a_0) = (2.0)(2.0) + (1.0)(3.0) = 4.0 + 3.0 = \mathbf{7.0}$
$\quad Q(s, a_1) = (-1.0)(2.0) + (3.0)(3.0) = -2.0 + 9.0 = \mathbf{7.0}$

(d) Tie — both $Q(s, a_0) = Q(s, a_1) = 7.0$ → **random tie-break**.

---

**Problem 2:**

(a) $y = r + \gamma \max_{a'} \hat{Q}(s', a') = 1 + 0.9 \times \max(0.75, 2.10) = 1 + 0.9 \times 2.10 = 1 + 1.89 = \mathbf{2.89}$

(b) $\delta = y - Q(s, a_0) = 2.89 - 7.0 = \mathbf{-4.11}$ (the network overestimated the value of $a_0$)

(c) Squared TD error $= \delta^2 = (-4.11)^2 = \mathbf{16.89}$

---

**Problem 3:**

(a) $\delta_1 = 5.0 - 3.5 = +1.5$, $\quad \delta_2 = 2.1 - 7.2 = -5.1$, $\quad \delta_3 = 1.8 - 1.0 = +0.8$

(b) $\mathcal{L} = \frac{1}{3}(1.5^2 + (-5.1)^2 + 0.8^2) = \frac{2.25 + 26.01 + 0.64}{3} = \frac{28.9}{3} = \mathbf{9.633}$

(c) Priorities: $|\delta_1|^1 = 1.5$, $|\delta_2|^1 = 5.1$, $|\delta_3|^1 = 0.8$. **Transition 2** has the highest priority (5.1) and would be sampled most frequently — the network was most wrong here (predicted 7.2, target was only 2.1).

---

**Problem 4:**

(a) Standard DQN: $y^{\text{DQN}} = 0 + 0.9 \times \max(\hat{Q}(s', a_0), \hat{Q}(s', a_1)) = 0.9 \times \max(2.1, 4.5) = 0.9 \times 4.5 = \mathbf{4.05}$

(b) Double DQN:
- Online selects: $\arg\max_{a'} Q(s', a';\, \boldsymbol{\theta}) = a_1$ (since $5.8 > 3.2$)
- Target evaluates $a_1$: $\hat{Q}(s', a_1;\, \boldsymbol{\theta}^-) = 4.5$
- $y^{\text{DDQN}} = 0 + 0.9 \times 4.5 = \mathbf{4.05}$

(c) Both give 4.05 here — they happen to agree because both networks agree $a_1$ is best. But when the online network is overconfident (picks a high-noise action), DDQN uses the target network to dampen the overestimation. In this example, both targets are close to the true 4.0 — illustrating that when networks roughly agree, DDQN and DQN behave similarly. The difference emerges when the online network's $\arg\max$ is an overestimated outlier that the target network correctly scores lower.

---

*Coming up in Week 6 — Policy Gradient Methods: we'll move beyond Q-values entirely and teach agents to directly optimize the policy itself — unlocking continuous action spaces, stochastic policies, and the foundations of modern robot control.*
