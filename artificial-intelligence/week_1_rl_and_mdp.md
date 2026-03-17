# Week 1 — Reinforcement Learning & Markov Decision Processes

---

## 1. Introduction to Reinforcement Learning

> *How does a child learn to walk? Not from a textbook — but from trying, falling, and trying again.*

### The Reinforcement Learning Paradigm

Reinforcement Learning (RL) is a learning paradigm where an **agent** learns by **interacting with an environment**. Unlike supervised learning (learning from labeled examples) or unsupervised learning (finding patterns in data), RL learns from **trial and error** guided by **rewards and punishments**.

**Intuitive Example — Training a Dog:**
- You ask the dog to sit. If it sits → give a treat (reward). If it doesn't → no treat (no reward).
- Over time, the dog learns that sitting = treat, and starts doing it reliably.
- The dog is the agent. You and the surroundings are the environment. The treat is the reward signal.

**Intuitive Example — Playing a Video Game:**
- A player (agent) takes actions (move left, jump, shoot) in a game world (environment).
- Defeating an enemy gives +10 points (reward). Falling into a pit gives -100 (penalty).
- The player gradually learns which actions lead to high scores.

### Agent and Environment

The two central entities in RL:

| Entity | Role |
|--------|------|
| **Agent** | The learner and decision-maker. Observes the environment and takes actions. |
| **Environment** | Everything outside the agent. Responds to actions and provides new states and rewards. |

The interaction loop:
1. Agent observes the current **state** $s_t$
2. Agent selects an **action** $a_t$
3. Environment transitions to a new **state** $s_{t+1}$
4. Environment emits a **reward** $r_{t+1}$
5. Repeat

```
        ┌─────────┐   action aₜ   ┌─────────────┐
        │  Agent  │ ────────────▶ │ Environment │
        │         │ ◀──────────── │             │
        └─────────┘  state sₜ₊₁  └─────────────┘
                      reward rₜ₊₁
```

### The Goal of Reinforcement Learning

The agent's goal is to learn a **policy** (a strategy for choosing actions) that **maximizes the total cumulative reward over time** — not just the immediate reward.

Formally, the agent wants to maximize the **expected return**:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

Where:
- $r_{t+k+1}$ is the **reward received at time step $t+k+1$** — a numerical signal from the environment telling the agent how good or bad the outcome of its action was at that moment
- $\gamma \in [0, 1]$ is the **discount factor** that controls how much the agent cares about future rewards versus immediate ones

---

## 2. Multi-Armed Bandit Problem

> *Imagine you walk into a casino with 10 slot machines. Each machine has a different (unknown) payout rate. How do you maximize your winnings?*

### Motivation

The **Multi-Armed Bandit** is the simplest RL problem — no states, no transitions, just repeated action selection and reward observation. It perfectly captures the core tension in RL: **should I use what I know, or try something new?**

**Intuitive Example — Choosing a Restaurant:**
- You have 5 restaurants near your office. You've tried a few and know roughly how good they are.
- Do you go to your current favorite (exploit), or try one you've never visited (explore)?
- If you always exploit, you might miss a hidden gem. If you always explore, you never enjoy your known favorites.

### Exploration and Exploitation Dilemma

| Strategy | Description | Risk |
|----------|-------------|------|
| **Exploitation** | Choose the action with the highest known reward | Miss better unknown options |
| **Exploration** | Try less-known actions to gather information | Waste time on bad options |

The dilemma: you must **balance** both to maximize long-term reward. Pure exploitation gets stuck in local optima; pure exploration never converges.

### Action-Value Methods

#### Greedy Algorithm

Estimate the value of each action $a$ as the average of observed rewards:

$$Q(a) = \frac{\text{sum of rewards when } a \text{ was taken}}{\text{number of times } a \text{ was taken}}$$

Always pick the action with the highest estimated value:

$$a^{*} = \arg\max_a Q(a)$$

**Problem:** Never explores. Gets permanently stuck on the first good option it finds.

#### Worked Calculation: Greedy Q-Value Updates

You have **3 slot machines** (Arm A, B, C) with unknown true reward means. Start with $Q(a) = 0$ for all arms.

| Round | Arm Tried | Reward | Update: $Q(a) = \frac{\sum \text{rewards}}{N(a)}$ | Q(A) | Q(B) | Q(C) | Next Greedy Pick |
|-------|-----------|--------|----------------------------------------------------|------|------|------|-----------------|
| 1 | A *(random tie)* | 2 | $Q(A) = 2/1 = 2.0$ | 2.0 | 0.0 | 0.0 | **A** |
| 2 | A | 1 | $Q(A) = (2+1)/2 = 1.5$ | 1.5 | 0.0 | 0.0 | **A** |
| 3 | A | 3 | $Q(A) = (2+1+3)/3 = 2.0$ | 2.0 | 0.0 | 0.0 | **A** |
| 4 | A | 0 | $Q(A) = (2+1+3+0)/4 = 1.5$ | 1.5 | 0.0 | 0.0 | **A** |
| 5 | A | 2 | $Q(A) = (2+1+3+0+2)/5 = 1.6$ | 1.6 | 0.0 | 0.0 | **A** |

**Result:** Greedy is permanently locked on Arm A (~1.6 average). It never discovers that Arm B's true mean is 5.0 *(known only to us as the observer — the agent never has access to this)* — a far better option that was never tried.

---

#### Epsilon-Greedy Algorithm ($\varepsilon$-greedy)

A simple fix: with probability $\varepsilon$, pick a **random** action (explore); otherwise, pick the greedy action (exploit).

$$a_t = \begin{cases} \text{random action} & \text{with probability } \varepsilon \\ \arg\max_a Q(a) & \text{with probability } 1 - \varepsilon \end{cases}$$

- Small $\varepsilon$ (e.g., 0.1) → mostly exploit, a little exploration
- Large $\varepsilon$ → more exploration, slower convergence

**Intuition:** Every 1 in 10 meals, force yourself to try a new restaurant. The rest of the time, enjoy your current favorite.

#### Worked Calculation: ε-Greedy in Action

Continuing from round 6 with $\varepsilon = 0.2$ and current estimates: $Q(A)=1.6,\ Q(B)=0.0,\ Q(C)=0.0$.

**Step 1 — Roll a random number** $u \sim \text{Uniform}(0, 1)$:

| Random Roll $u$ | Condition | Decision | Action Taken |
|-----------------|-----------|----------|--------------|
| $u = 0.05$ | $0.05 < \varepsilon\ (0.2)$ | **Explore** | Random pick → Arm B |
| $u = 0.72$ | $0.72 \geq \varepsilon\ (0.2)$ | **Exploit** | $\arg\max Q(a)$ → Arm A |

**Step 2 — Suppose $u = 0.05$ → Explore → choose Arm B → observe reward 6:**

$$Q(B)_{\text{new}} = \frac{\text{total reward}}{N(B)} = \frac{0 + 6}{1} = 6.0$$

**Step 3 — Updated estimates flip the greedy choice:**

| Arm | Q Value | Greedy? |
|-----|---------|---------|
| A | 1.6 | |
| B | **6.0** | ✓ New best! |
| C | 0.0 | |

One random exploration unlocked the superior arm — something pure greedy could never achieve.

**Real-World Connection — Website A/B Testing:**

A product team wants to find the highest-converting button for their signup page:

| Arm | Button Text | Trials | Avg. Conversion |
|-----|-------------|--------|----------------|
| A | "Sign Up Now" | 200 | 3.2% |
| B | "Get Started Free" | 50 | 4.8% ← current best |
| C | "Try It Today" | 50 | 2.9% |

With $\varepsilon = 0.1$: send **90% of traffic** to Arm B while sending **10% randomly** to keep learning. This earns revenue now while still searching for something better — a business-safe RL strategy you can deploy in production today.

---

#### Upper Confidence Bound (UCB)

Instead of exploring randomly, UCB selects actions that are either **high-value** *or* **under-explored** — giving a "bonus" to actions we're uncertain about.

$$a_t = \arg\max_a \left[ Q(a) + c \sqrt{\frac{\ln t}{N(a)}} \right]$$

Where:
- $Q(a)$ = estimated value of action $a$
- $N(a)$ = number of times action $a$ has been selected
- $t$ = total number of steps so far
- $c$ = confidence level parameter

**Intuition:** When $N(a)$ is small (rarely tried), the bonus term is large → UCB is curious about untested options. As $N(a)$ grows, the bonus shrinks → UCB trusts its estimates more.

**Advantage over $\varepsilon$-greedy:** UCB explores *intelligently* — it targets uncertainty rather than exploring blindly at random.

#### Worked Calculation: UCB Scores

At step $t = 12$ with $c = 2$, you are choosing between 3 food delivery apps. Total trials: $N(A)+N(B)+N(C) = 8+3+1 = 12$.

| Arm | $Q(a)$ (avg rating) | $N(a)$ (orders) | Bonus: $c\sqrt{\dfrac{\ln t}{N(a)}}$ | **UCB Score** |
|-----|---------------------|-----------------|--------------------------------------|---------------|
| A | 4.5 | 8 | $2\sqrt{\frac{\ln 12}{8}} = 2\sqrt{\frac{2.485}{8}} = 2(0.557) = 1.11$ | **5.61** |
| B | 5.2 | 3 | $2\sqrt{\frac{\ln 12}{3}} = 2\sqrt{\frac{2.485}{3}} = 2(0.910) = 1.82$ | **7.02** ✓ |
| C | 3.8 | 1 | $2\sqrt{\frac{\ln 12}{1}} = 2\sqrt{2.485} = 2(1.577) = 3.15$ | **6.95** |

*($\ln 12 \approx 2.485$)*

**UCB selects App B.** Although App C has the largest uncertainty bonus, App B wins because it combines a solid average (5.2) with meaningful unexplored potential. Pure ε-greedy might have wasted this turn on App C instead.

### Real-World Applications of Bandit Algorithms

| Application | Arms | Reward Signal | Why Bandit? |
|-------------|------|---------------|-------------|
| **Website A/B Testing** | Button text / layouts | Click-through or conversion rate | Reallocate traffic to the winner while still testing |
| **Adaptive Clinical Trials** | Drug dosages / formulations | Patient recovery rate | Minimize patient exposure to inferior treatments |
| **Content Recommendation** | Movies, articles, ads | Watch / click / purchase rate | Personalize in real time without complete data up front |
| **Search Result Ranking** | Ordering of results | User click / dwell time | Continuously improve ranking from live feedback |

**Key insight:** Whenever you face **repeated decisions with uncertain outcomes** and you want to learn from feedback while still performing well, a bandit algorithm applies — no state, no planning required.

---

## 3. Markov Decision Process (MDP)

> *Now let's enrich the bandit problem: actions have consequences that unfold over time, and where you are matters.*

### Motivation

The bandit problem ignores **state** — every decision is independent. But most real problems have **sequential structure**: the situation (state) you're in depends on past actions, and your current action affects future situations.

**Intuitive Example — Chess:**
- The board position is the **state**. Your move is the **action**. The resulting new board position is the **next state**.
- A move that wins a pawn now might lose you the game later. You need to reason about long-term consequences.

**Intuitive Example — Thermostat:**
- The current room temperature is the state. Turning heating on/off is the action. Temperature changes are the transitions. Reaching the target temperature is the reward.

### The Markov Property

An MDP assumes the **Markov Property**: the future depends only on the **current state**, not on the full history of past states and actions.

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

**Intuition:** To decide your next move in chess, you only need to look at the current board — not the entire history of how you got there. The present state captures all relevant information.

### Formalization

An MDP is defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$:

| Component | Symbol&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description |
|-----------|--------------------------------|-------------|
| **State space** | $\mathcal{S}$ | The set of all possible situations the agent can be in |
| **Action space** | $\mathcal{A}$ | The set of all possible actions the agent can take |
| **Transition dynamics** | $\mathcal{T}(s' \mid s, a)$ | Probability of reaching state $s'$ after taking action $a$ in state $s$ |
| **Reward function** | $\mathcal{R}(s, a, s')$ | The reward received after transitioning from $s$ to $s'$ via action $a$ |
| **Discount factor** | $\gamma \in [0, 1]$ | How much future rewards are valued relative to immediate rewards |

**Discount factor intuition:**
- $\gamma = 0$: Agent is completely short-sighted — only cares about immediate reward
- $\gamma = 1$: Agent values all future rewards equally (only valid for **finite-horizon problems** — tasks that have a guaranteed end point, such as one game of chess, one episode of a robot reaching a goal, or one patient treatment cycle; because the episode always terminates in a finite number of steps, the total sum $\sum r_t$ stays bounded even without discounting; in contrast, **infinite-horizon problems** — like a server that runs forever — require $\gamma < 1$ to prevent the sum from diverging to infinity)
- $\gamma = 0.9$: A reward received 10 steps in the future is counted as only $0.9^{10} \approx 35\%$ of its face value — so a future reward of $+100$ contributes only $\approx 35$ to the return $G_t$, whereas the same $+100$ received right now contributes the full $100$

### Running Example: Robot Navigation MDP

To make the formalism concrete, here is a small but complete MDP we will use throughout the rest of this section.

**Scenario:** A robot moves through a 3-room building to reach the Goal room. The Hallway floor is slippery — the robot occasionally slides back to Start.

```
┌─────────┐    Move →    ┌──────────┐  ─── Move (p=0.8) ──▶  ┌──────┐
│  Start  │ ──────────▶  │ Hallway  │                         │ Goal │
│   (S)   │              │   (H)    │  ◀── Slip  (p=0.2) ───  │ (G)  │
└─────────┘              └──────────┘                         └──────┘
```

**MDP Components:**

| Component | Symbol | Value |
|-----------|--------|-------|
| State space | $\mathcal{S}$ | $\{S,\ H,\ G\}$ |
| Action space | $\mathcal{A}$ | $\{\text{Move}\}$ |
| Transition | $\mathcal{T}(H \mid S, \text{Move})$ | $1.0$ — always reaches Hallway |
| Transition | $\mathcal{T}(G \mid H, \text{Move})$ | $0.8$ — reaches Goal 80% of the time |
| Transition | $\mathcal{T}(S \mid H, \text{Move})$ | $0.2$ — slips back to Start 20% of the time |
| Reward | $\mathcal{R}(S, \text{Move}, H)$ | $-1$ (step cost) |
| Reward | $\mathcal{R}(H, \text{Move}, G)$ | $+10$ (goal reached!) |
| Reward | $\mathcal{R}(H, \text{Move}, S)$ | $-1$ (slip cost) |
| Discount | $\gamma$ | $0.9$ |

> **Reading the reward notation** — $\mathcal{R}(\textit{current state},\ \textit{action},\ \textit{next state})$ is the reward signal the environment emits after one transition. For example, $\mathcal{R}(S, \text{Move}, H)$ means *"the agent was in Start, chose Move, and arrived in Hallway."*
>
> **Why is the step cost −1?** Without any cost per step, the agent has no incentive to reach the goal quickly — it could take 1 000 unnecessary detours and still expect the same final reward. Charging −1 for every action makes *shorter paths more valuable*, so the agent is naturally driven to reach the goal as fast as possible. Think of it as a small time-tax: every second the robot is still wandering costs it one point.
>
> **Why is the goal reward +10?** The value 10 is a deliberate design choice to make the goal *clearly worth pursuing* despite the step costs. The key is the **ratio** between the goal reward and the step cost: with +10 and a −1 per step, the agent breaks even at 10 steps — reaching the goal in fewer steps is profitable, in more steps is a loss. If the goal reward were only +2, a 3-step journey would already be a net loss (2 − 3 = −1), and the agent might learn to avoid the goal entirely. In general, set the goal reward large enough that the optimal path yields a comfortable positive return, but not so large that the step costs become negligible and the agent stops caring about efficiency.

$G$ is the **terminal state** — the episode ends once the robot arrives.

### Policy

A **policy** $\pi$ defines the agent's behavior — a mapping from states to actions.

**Deterministic policy:**

$$\pi(s) = a \quad \text{(always take action } a \text{ in state } s\text{)}$$

**Stochastic policy:**

$$\pi(a \mid s) = P(\text{take action } a \mid \text{in state } s)$$

The goal of RL is to find the **optimal policy** $\pi^{*}$ that maximizes the expected cumulative discounted reward from any starting state:

$$\pi^{*} = \arg\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \right]$$

| Notation | Meaning |
|----------|---------|
| $\pi^{*}$ | The **optimal policy** — the best possible strategy for choosing actions |
| $\arg\max_\pi$ | "Find the policy $\pi$ that produces the highest value" — $\arg\max$ returns the *argument* (here, the policy itself) that maximises the expression, not just the maximum value |
| $\mathbb{E}_\pi[\cdots]$ | **Expected value** when the agent follows policy $\pi$ — because transitions can be stochastic, we average over all possible trajectories the agent might experience |
| $\sum_{t=0}^{\infty}$ | Sum over all future time steps, from now ($t=0$) to infinity — captures the full lifetime of the agent |
| $\gamma^t$ | The **discount factor** raised to the power $t$ — rewards $t$ steps in the future are multiplied by $\gamma^t$, making them worth less the further away they are |
| $r_{t+1}$ | The **reward received at time step $t+1$** — it is $t+1$ (not $t$) because the reward is received *after* the agent takes action $a_t$ at time $t$ |

**Value function** — the expected return from state $s$ following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right]$$

| Notation | Meaning |
|----------|---------|
| $V^\pi(s)$ | The **value of state $s$ under policy $\pi$** — a single number answering *"how much total discounted reward can I expect from here if I follow policy $\pi$?"* |
| $\mathbb{E}_\pi[\cdots]$ | **Expected value** under policy $\pi$ — averages the return over all possible future trajectories, since transitions may be stochastic |
| $G_t$ | The **return at time $t$** — the discounted sum of all future rewards from step $t$ onward: $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$ |
| $\mid$ | **"given that"** — a conditional; everything after $\mid$ is the condition we fix |
| $s_t = s$ | The condition: *"the agent is in state $s$ at time $t$"* — we are asking for the expected return specifically from this starting state |

The optimal value function satisfies the **Bellman Optimality Equation**:

$$V^{*}(s) = \max_a \sum_{s'} \mathcal{T}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V^{*}(s') \right]$$

**In plain language:**

> *"The best value I can get from my current state equals: pick the action that gives me the highest expected score, where that score is the immediate reward I receive right now, plus a discounted estimate of how good the state I land in will be."*

Breaking it down piece by piece:

| Part of the equation | What it asks |
|----------------------|-------------|
| $\max_a$ | *"Which action should I take?"* — try every possible action and pick the best one |
| $\sum_{s'} \mathcal{T}(s' \mid s, a)[\cdots]$ | *"The world is uncertain"* — I might land in different next states; average the outcome across all of them, weighted by their probability |
| $\mathcal{R}(s, a, s')$ | *"What do I earn right now?"* — the immediate reward from this transition |
| $\gamma V^{*}(s')$ | *"What is the future worth from where I land?"* — the discounted value of the next state, assuming I keep acting optimally from there |

**The key insight — recursive self-consistency:**

The equation says that the value of the *current* state is defined in terms of the value of the *next* state — which is defined in terms of the state after that, and so on. This is the core idea of **dynamic programming**: a big problem (what is the best long-run outcome from state $s$?) is broken into a smaller one (what is the best immediate action, and then what is the best outcome from the resulting state?).

Think of it like planning a road trip. The best route from your *current city* is: choose the best next city to drive to today, collect whatever you see along the way, and then assume you'll also drive optimally for the rest of the journey from there.

#### Worked Calculation: Value Iteration on the Robot Navigation MDP

Using the Robot Navigation MDP ($\gamma = 0.9$). We solve the Bellman equation repeatedly — updating all states each iteration — until values stop changing.

**Initialize:** $V(S) = V(H) = V(G) = 0$

---

**Iteration 1** — Plug current values into the Bellman equation:

$V(G) = 0$ (terminal state)

$$V(H) = \underbrace{0.8}_{\text{reach Goal}} \bigl[\underbrace{+10}_{\mathcal{R}} + 0.9 \times \underbrace{V(G)}_{0}\bigr] + \underbrace{0.2}_{\text{slip back}} \bigl[\underbrace{-1}_{\mathcal{R}} + 0.9 \times \underbrace{V(S)}_{0}\bigr] \\
= 0.8(10) + 0.2(-1) = 8.0 - 0.2 = \mathbf{7.8}$$

$$V(S) = 1.0 \times \bigl[-1 + 0.9 \times \underbrace{V(H)}_{7.8}\bigr] = -1 + 7.02 = \mathbf{6.02}$$

---

**Iteration 2** — Use the updated values:

$$V(H) = 0.8(10 + 0) + 0.2\bigl(-1 + 0.9 \times \underbrace{V(S)}_{6.02}\bigr) \\
= 8.0 + 0.2(-1 + 5.42) = 8.0 + 0.2(4.42) = 8.0 + 0.88 = \mathbf{8.88}$$

$$V(S) = -1 + 0.9 \times \underbrace{V(H)}_{8.88} = -1 + 7.99 = \mathbf{6.99}$$

---

**Convergence criterion:**

We stop iterating when the largest change across all states in a single iteration falls below a small threshold $\theta$ (called the **tolerance**):

$$\max_s \left| V_{\text{new}}(s) - V_{\text{old}}(s) \right| < \theta$$

In practice, $\theta = 0.01$ or $\theta = 0.001$ is common. Applying it to this example:

| Iteration | $\Delta V(S)$ | $\Delta V(H)$ | $\max \Delta$ | Stop? |
|-----------|--------------|--------------|---------------|-------|
| 1 | $|6.02 - 0| = 6.02$ | $|7.80 - 0| = 7.80$ | 7.80 | No |
| 2 | $|6.99 - 6.02| = 0.97$ | $|8.88 - 7.80| = 1.08$ | 1.08 | No |
| 3 | $|7.15 - 6.99| = 0.16$ | $|9.06 - 8.88| = 0.18$ | 0.18 | No |
| 4 | $|7.18 - 7.15| = 0.03$ | $|9.09 - 9.06| = 0.03$ | 0.03 | No ($\theta=0.01$) |
| 5 | $\approx 0.005$ | $\approx 0.005$ | $\approx 0.005$ | **Yes** ✓ |

**Intuition:** Think of it like a GPS recalculating a route. After each pass, the estimated times change less and less. Once the update is smaller than "1 second difference" (your $\theta$), the GPS considers the route settled and stops recalculating.

**Summary of convergence:**

| Iteration | $V(S)$ | $V(H)$ | $V(G)$ |
|-----------|--------|--------|--------|
| 0 (init) | 0.00 | 0.00 | 0.00 |
| 1 | 6.02 | 7.80 | 0.00 |
| 2 | 6.99 | 8.88 | 0.00 |
| 3 | 7.15 | 9.06 | 0.00 |
| Converged | **7.18** | **9.09** | **0.00** |

**Interpretation:**
- $V^{*}(H) \approx 9.09$ — a robot already in the Hallway can expect ~9.09 total discounted reward by acting optimally.
- $V^{*}(S) \approx 7.18$ — a robot at Start gets slightly less because it pays a −1 step cost before reaching the Hallway.
- The difference $9.09 - 7.18 = 1.91$ reflects exactly the expected cost of the Start → Hallway transition (step cost of −1, discounted one step: $1 \times 0.9 \approx 0.9$, plus the 20% slip risk in the Hallway compounding over time).

### Real-World Applications of MDPs

| Domain | States | Actions | Reward Signal |
|--------|--------|---------|---------------|
| **Robot Navigation** | Map position (x, y, heading) | Move, turn, stop | −1/step, +100 at goal, −50 collision |
| **Inventory Management** | Current stock level | Order quantity | −holding cost, −stockout penalty |
| **Traffic Signal Control** | Vehicle queue lengths per lane | Green/red duration | −average vehicle wait time |
| **Game Playing (Chess / Go)** | Board position | Legal moves | +1 win, −1 loss, 0 draw |
| **Medical Treatment Planning** | Patient health indicators | Treatment decisions | Long-term patient outcome score |
| **Autonomous Driving** | Vehicle position, speed, surroundings | Steer, accelerate, brake | +progress, −collision, −violations |

**Key insight:** Any sequential decision problem where today's action affects tomorrow's situation — and you want to optimize long-term outcomes — can be modeled as an MDP. The Bellman equation is the mathematical backbone for computing the optimal strategy in all of these domains.

---

## 4. Summary

| Concept | Key Idea |
|---------|----------|
| **RL Paradigm** | Agent learns by interacting with an environment, guided by rewards |
| **Agent & Environment** | Agent acts; environment responds with new state and reward |
| **Goal of RL** | Maximize total cumulative (discounted) reward over time |
| **Multi-Armed Bandit** | Simplest RL setting — no states, just repeated action selection |
| **Exploration vs. Exploitation** | Core tension: use known best option vs. discover better ones |
| **Greedy** | Always pick the best-known action; no exploration |
| **$\varepsilon$-Greedy** | Explore randomly with probability $\varepsilon$; otherwise exploit |
| **UCB** | Explore intelligently by targeting high uncertainty actions |
| **MDP** | Formalism for sequential decision-making with states, actions, transitions, rewards |
| **Markov Property** | Future depends only on current state, not history |
| **Policy** | A strategy mapping states to actions; goal is to find the optimal policy $\pi^{*}$ |

---

## 5. Exercise

Test your understanding of the key concepts from this week with a crossword puzzle featuring 20 terms from Reinforcement Learning and Markov Decision Processes.

**WARNING**
- Your test will be ended automatically once you click on the **reveal** button
- The test use only your 1st time taking score.  The later score of the same student will not be used.

[Open Crossword Puzzle](https://claude.ai/public/artifacts/76fe6679-4aa2-441f-b591-a6d297ffeff6)
