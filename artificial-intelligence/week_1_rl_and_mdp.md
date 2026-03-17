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

Where $\gamma \in [0, 1]$ is the **discount factor** that controls how much the agent cares about future rewards versus immediate ones.

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

$$a^* = \arg\max_a Q(a)$$

**Problem:** Never explores. Gets permanently stuck on the first good option it finds.

---

#### Epsilon-Greedy Algorithm ($\varepsilon$-greedy)

A simple fix: with probability $\varepsilon$, pick a **random** action (explore); otherwise, pick the greedy action (exploit).

$$a_t = \begin{cases} \text{random action} & \text{with probability } \varepsilon \\ \arg\max_a Q(a) & \text{with probability } 1 - \varepsilon \end{cases}$$

- Small $\varepsilon$ (e.g., 0.1) → mostly exploit, a little exploration
- Large $\varepsilon$ → more exploration, slower convergence

**Intuition:** Every 1 in 10 meals, force yourself to try a new restaurant. The rest of the time, enjoy your current favorite.

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

| Component | Symbol | Description |
|-----------|--------|-------------|
| **State space** | $\mathcal{S}$ | The set of all possible situations the agent can be in |
| **Action space** | $\mathcal{A}$ | The set of all possible actions the agent can take |
| **Transition dynamics** | $\mathcal{T}(s' \mid s, a)$ | Probability of reaching state $s'$ after taking action $a$ in state $s$ |
| **Reward function** | $\mathcal{R}(s, a, s')$ | The reward received after transitioning from $s$ to $s'$ via action $a$ |
| **Discount factor** | $\gamma \in [0, 1]$ | How much future rewards are valued relative to immediate rewards |

**Discount factor intuition:**
- $\gamma = 0$: Agent is completely short-sighted — only cares about immediate reward
- $\gamma = 1$: Agent values all future rewards equally (only valid for finite-horizon problems)
- $\gamma = 0.9$: A reward 10 steps away is worth $0.9^{10} \approx 35\%$ of an immediate reward

### Policy

A **policy** $\pi$ defines the agent's behavior — a mapping from states to actions.

**Deterministic policy:**
$$\pi(s) = a \quad \text{(always take action } a \text{ in state } s\text{)}$$

**Stochastic policy:**
$$\pi(a \mid s) = P(\text{take action } a \mid \text{in state } s)$$

The goal of RL is to find the **optimal policy** $\pi^*$ that maximizes the expected cumulative discounted reward from any starting state:

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \right]$$

**Value function** — the expected return from state $s$ following policy $\pi$:
$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right]$$

The optimal value function satisfies the **Bellman Optimality Equation**:
$$V^*(s) = \max_a \sum_{s'} \mathcal{T}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V^*(s') \right]$$

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
| **Policy** | A strategy mapping states to actions; goal is to find the optimal policy $\pi^*$ |

---

## 5. Exercise

Test your understanding of the key concepts from this week with a crossword puzzle featuring 20 terms from Reinforcement Learning and Markov Decision Processes.

**WARNING**
- Your test will be ended automatically once you click on the **reveal** button
- The test use only your 1st time taking score.  The later score of the same student will not be used.

[Open Crossword Puzzle](https://claude.ai/public/artifacts/76fe6679-4aa2-441f-b591-a6d297ffeff6)
