# Week 4 — Temporal-Difference Control (Q-Learning)

---

## 1. The Bridge from Week 3: One Missing Piece

> *In Week 3, you learned SARSA — an agent that improves its strategy based on what it actually does. But here's a puzzle: what if the agent could learn the optimal strategy even while behaving cautiously or randomly? What if learning and acting were completely separate concerns?*

### What SARSA Left Unsolved

In Week 3, we introduced **SARSA** — an on-policy TD control algorithm that updates Q-values based on the quintuple $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$. Because the update uses the **actual next action** $a_{t+1}$ chosen by the current policy, SARSA evaluates the policy that is *actually being followed* — including its exploration behavior.

This is a subtle but important limitation.

**Intuitive Example — The Cautious Learner:**

Imagine teaching a student pilot. The trainee always flies very conservatively — gentle turns, slow speed — because they're nervous and exploring. SARSA evaluates the value of being *cautious*, not the value of being *optimal*. The trainee learns the value of their own cautious habits, not the value of the ideal flight path a master pilot would follow.

Ideally, the trainee should be able to practice cautiously for safety, while simultaneously learning the *optimal* strategy — the one a master pilot would use. These are two different things:

- **What I am doing right now** (the cautious exploration policy)
- **What I should ultimately do** (the optimal greedy policy)

**Q-Learning** separates these two concerns entirely. It uses the ε-greedy policy to **explore** the environment, but always learns the value of the **best possible action** at the next state — regardless of what action the agent will actually take.

---

## 2. On-Policy vs. Off-Policy: The Core Distinction

This week's central concept is the difference between **on-policy** and **off-policy** learning. Understanding this distinction is key to understanding Q-Learning.

### Two Policies at Once

In off-policy learning, the agent maintains **two separate policies**:

| Policy | Name | Role |
|--------|------|------|
| **Behavior policy** $b(a \mid s)$ | The policy used to **act** and collect experience | Ensures exploration — often ε-greedy |
| **Target policy** $\pi(a \mid s)$ | The policy being **learned and optimized** | The greedy policy — $\arg\max_a Q(s, a)$ |

**On-policy (SARSA):** The behavior policy and the target policy are the same. You learn the value of what you're actually doing.

**Off-policy (Q-Learning):** The behavior policy and the target policy are different. You act one way (to explore), but you learn the value of acting the optimal way.

### Intuitive Analogy — Learning from Someone Else's Mistakes

**On-policy learning** is like learning to drive by evaluating your own driving. Every update reflects your actual habits — including your mistakes and nervous detours.

**Off-policy learning** is like watching a professional driver race while you sit in the passenger seat — you collect data by observing (or in RL terms, by acting), but what you're learning to evaluate is the *expert's* strategy, not your own cautious behavior.

This gives off-policy methods a significant advantage: **they can reuse experience from any source** — old recordings, random agents, demonstrations by humans — because the learning target is decoupled from who collected the data.

| Property | On-Policy (SARSA) | Off-Policy (Q-Learning) |
|----------|-------------------|------------------------|
| Behavior policy | ε-greedy | ε-greedy |
| Target policy | ε-greedy (same) | Greedy (different) |
| Learns the value of | The ε-greedy policy | The optimal greedy policy |
| Can reuse old data? | ✗ No (data must come from current policy) | ✓ Yes (any data source works) |
| Safety during training | More conservative | Can be more aggressive |

---

## 3. Q-Learning: The Algorithm

> *Q-Learning is arguably the most important algorithm in all of Reinforcement Learning. It is simple, powerful, and the direct ancestor of algorithms like DQN — the system that first beat humans at Atari video games.*

### 3.1 The Q-Learning Update Rule

Q-Learning was introduced by Chris Watkins in 1989 and is remarkably concise. After observing the transition $(s_t, a_t, r_{t+1}, s_{t+1})$, the update is:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Bigl[ \underbrace{r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')}_{\text{Q-Learning target}} - Q(s_t, a_t) \Bigr]$$

Compare this directly with the SARSA update:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Bigl[ \underbrace{r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})}_{\text{SARSA target}} - Q(s_t, a_t) \Bigr]$$

**The one critical difference:**

| Algorithm | Next Q-value used in target | Meaning |
|-----------|-----------------------------|---------|
| **SARSA** | $Q(s_{t+1},\; a_{t+1})$ — the action **actually chosen** by the ε-greedy policy | "What is my current policy worth at the next state?" |
| **Q-Learning** | $\max_{a'} Q(s_{t+1},\; a')$ — the **best possible** action at the next state | "What is the optimal policy worth at the next state?" |

SARSA needs **five** values: $(s, a, r, s', a')$ — it must observe the next action before updating.

Q-Learning needs only **four**: $(s, a, r, s')$ — it doesn't care what action will be taken next. It always imagines taking the *best* action.

### 3.2 Breaking Down the Q-Learning Update

$$Q(s_t, a_t) \leftarrow \underbrace{Q(s_t, a_t)}_{\text{old estimate}} + \alpha \Bigl[ \underbrace{r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')}_{\text{Q-Learning target}} - Q(s_t, a_t) \Bigr]$$

| Term | Meaning |
|------|---------|
| $Q(s_t, a_t)$ | Current estimate: "How valuable do I think it is to take action $a_t$ in state $s_t$?" |
| $r_{t+1}$ | The real reward just received — ground truth for one step |
| $\max_{a'} Q(s_{t+1}, a')$ | Optimistic look-ahead: "What is the best Q-value achievable from the next state?" |
| $r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')$ | The Q-Learning target — a better estimate of the true $Q^*(s_t, a_t)$ |
| $[\text{target} - Q(s_t, a_t)]$ | The **TD error** $\delta_t$ — the surprise signal |
| $\alpha$ | Learning rate — how aggressively we shift toward the new estimate |

**The key insight — "greedy in the limit":** Q-Learning always assumes that from the *next* state onward, the agent will act *optimally* (take the $\max$). This is why it directly approximates the **optimal** Q-function $Q^*(s, a)$, regardless of what policy is actually being used to explore.

### 3.3 The Q-Learning Algorithm

```
Initialize Q(s, a) = 0 for all states s, all actions a
Set α (learning rate), γ (discount), ε (exploration rate)

For each episode:
    Start at initial state s

    For each step until terminal state:
        Choose action a using ε-greedy on Q(s, ·)
            → with probability ε: pick random action
            → with probability 1−ε: pick argmax_a Q(s, a)

        Take action a
        Observe reward r and next state s'

        ┌─────────────────────────────────────────────────────────────┐
        │  Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a') − Q(s, a)] │
        └─────────────────────────────────────────────────────────────┘

        s ← s'

    (Optionally decay ε over episodes for less exploration over time)
```

Notice: the algorithm does **not** need $a'$. After the update, $s$ moves to $s'$, and the next action is chosen fresh at the top of the loop.

---

## 4. Worked Calculation: Q-Learning Step by Step

**Environment:** A $1 \times 4$ grid representing a hallway:

```
[Start]  →  [Room A]  →  [Room B]  →  [Exit]
   S             A             B          E (terminal)
```

**Actions:** `Right` (move forward) or `Left` (move backward). Agent starts at `S`.

**Rewards:**
- Any move: $r = -1$ (time cost)
- Entering `Exit (E)` from `B`: $r = +20$ (success!)
- Moving `Left` from `S`: $r = -5$ (bumps into wall — not allowed, stays at `S`)

**Parameters:** $\gamma = 0.9$, $\alpha = 0.5$, $\varepsilon = 0.2$

**Initial Q-table** (all zeros):

| State | $Q(\cdot, \text{Right})$ | $Q(\cdot, \text{Left})$ |
|-------|--------------------------|-------------------------|
| S | 0.0 | 0.0 |
| A | 0.0 | 0.0 |
| B | 0.0 | 0.0 |

---

### Episode 1

**Step 1:** $s = S$. Both Q-values at $S$ are 0 — random tie-break → choose `Right`.

Observe $r = -1$, arrive at $s' = A$.

**Q-Learning target:**
$$r + \gamma \max_{a'} Q(A, a') = -1 + 0.9 \times \max(0.0, 0.0) = -1 + 0 = -1$$

**Update:**
$$Q(S, \text{Right}) \leftarrow 0 + 0.5(-1 - 0) = \mathbf{-0.5}$$

Q-table now: $Q(S, \text{Right}) = -0.5$

---

**Step 2:** $s = A$. Both $Q(A, \cdot) = 0$ → random → choose `Right`.

Observe $r = -1$, arrive at $s' = B$.

**Q-Learning target:**
$$-1 + 0.9 \times \max(Q(B, \text{Right}), Q(B, \text{Left})) = -1 + 0.9 \times 0 = -1$$

**Update:**
$$Q(A, \text{Right}) \leftarrow 0 + 0.5(-1 - 0) = \mathbf{-0.5}$$

---

**Step 3:** $s = B$. Both $Q(B, \cdot) = 0$ → random → choose `Right`.

Observe $r = +20$, arrive at $s' = E$ (terminal).

**Q-Learning target:**
$$20 + 0.9 \times \max(Q(E, \cdot)) = 20 + 0 = 20 \quad \text{(terminal: all Q-values = 0)}$$

**Update:**
$$Q(B, \text{Right}) \leftarrow 0 + 0.5(20 - 0) = \mathbf{10.0}$$

**End of Episode 1. Q-table:**

| State | $Q(\cdot, \text{Right})$ | $Q(\cdot, \text{Left})$ |
|-------|--------------------------|-------------------------|
| S | −0.5 | 0.0 |
| A | −0.5 | 0.0 |
| B | **10.0** | 0.0 |

---

### Episode 2

**Step 1:** $s = S$. Greedy: $\arg\max = \text{Left}$ (since $Q(S,\text{Left}) = 0 > Q(S,\text{Right}) = -0.5$).

Suppose $\varepsilon$-roll says exploit → choose `Left`. Agent bumps the wall, $r = -5$, stays at $s' = S$.

**Q-Learning target:**
$$-5 + 0.9 \times \max(Q(S, \text{Right}), Q(S, \text{Left})) = -5 + 0.9 \times 0 = -5$$

**Update:**
$$Q(S, \text{Left}) \leftarrow 0 + 0.5(-5 - 0) = \mathbf{-2.5}$$

Now: $Q(S, \text{Right}) = -0.5 > Q(S, \text{Left}) = -2.5$ → greedy at $S$ switches to **Right**!

---

**Step 2:** $s = S$. Greedy: `Right`.

Observe $r = -1$, $s' = A$.

**Q-Learning target:**
$$-1 + 0.9 \times \max(Q(A, \text{Right}), Q(A, \text{Left})) = -1 + 0.9 \times 0 = -1$$

**Update:**
$$Q(S, \text{Right}) \leftarrow -0.5 + 0.5(-1 - (-0.5)) = -0.5 + 0.5(-0.5) = \mathbf{-0.75}$$

---

**Step 3:** $s = A$. Greedy: $\text{Left}$ (since both $Q(A, \cdot) \leq 0$ and $\text{Left} = 0 > \text{Right} = -0.5$).

Suppose agent **exploits** Left → observe $r = -1$, arrive at $s' = S$.

**Q-Learning target:**
$$-1 + 0.9 \times \max(Q(S, \text{Right}), Q(S, \text{Left})) = -1 + 0.9 \times (-0.75) = -1 - 0.675 = -1.675$$

**Update:**
$$Q(A, \text{Left}) \leftarrow 0 + 0.5(-1.675 - 0) = \mathbf{-0.84}$$

Now $Q(A, \text{Right}) = -0.5 > Q(A, \text{Left}) = -0.84$ → greedy at $A$ is now **Right**!

---

**Step 4:** Back at $S$ again. Greedy → `Right`. $r = -1$, $s' = A$.

**Step 5:** At $A$. Greedy → `Right`. $r = -1$, $s' = B$.

**Step 6:** At $B$. Greedy → `Right` (since $Q(B, \text{Right}) = 10.0 \gg 0$). $r = +20$, $s' = E$.

**Q-Learning target:**
$$20 + 0 = 20$$

**Update:**
$$Q(B, \text{Right}) \leftarrow 10.0 + 0.5(20 - 10.0) = 10.0 + 5.0 = \mathbf{15.0}$$

**End of Episode 2. Q-table:**

| State | $Q(\cdot, \text{Right})$ | $Q(\cdot, \text{Left})$ |
|-------|--------------------------|-------------------------|
| S | −0.75 | −2.5 |
| A | −0.5 | −0.84 |
| B | **15.0** | 0.0 |

---

### Watching Q-Values Converge

Each episode, the large reward at `Exit` propagates **backward** one step further:

| Episode | $Q(B, R)$ | $Q(A, R)$ | $Q(S, R)$ |
|---------|-----------|-----------|-----------|
| 0 (init) | 0.0 | 0.0 | 0.0 |
| 1 | 10.0 | −0.5 | −0.5 |
| 2 | 15.0 | ~3.25 | ~0.96 |
| 3 | ~17.5 | ~9.86 | ~5.0 |
| … | … | … | … |
| Converged | **≈ 20.0** | **≈ 17.0** | **≈ 14.3** |

**Verify converged values analytically** (true $Q^*$ with $\gamma = 0.9$, no stochasticity):

$$Q^*(B, \text{Right}) = 20 + 0.9 \times 0 = 20 \checkmark$$
$$Q^*(A, \text{Right}) = -1 + 0.9 \times 20 = -1 + 18 = 17 \checkmark$$
$$Q^*(S, \text{Right}) = -1 + 0.9 \times 17 = -1 + 15.3 = 14.3 \checkmark$$

The **optimal policy** extracted from the converged Q-table: always move `Right` from every state — the agent discovers this purely from experience.

---

## 5. The Exploration vs. Exploitation Dilemma — In Depth

> *Every morning you choose between going to your favourite coffee shop (exploit your best-known option) or trying that new café you walked past yesterday (explore the unknown). If you always go to your favourite, you might never find something better. If you always try new places, you never enjoy your current favourite.*

This dilemma is at the heart of all RL control problems. Q-Learning doesn't solve it — it sidesteps it during *learning* by always updating toward the optimal action. But it still needs a **behavior policy** during data collection that balances exploration and exploitation.

### 5.1 The Problem with Pure Greedy

If the agent always exploits (picks $\arg\max_a Q(s, a)$ every time), certain state-action pairs may **never be visited**. Their Q-values stay at 0 forever — and 0 might look great compared to a correctly learned negative value, tricking the agent into a permanently suboptimal strategy.

**Concrete example:** Three paths from city A to city B:
- Path 1 (known): +5 reward — frequently used, Q(A, Path1) correctly ≈ 5
- Path 2 (never tried): true reward = +15 but Q(A, Path2) = 0 — looks *worse* than Path 1 to greedy agent
- Path 3 (never tried): true reward = −10 but Q(A, Path3) = 0 — also looks like Path 1

A pure greedy agent is trapped on Path 1 forever. It will never discover that Path 2 is three times better.

### 5.2 ε-Greedy (The Standard Approach)

The simplest and most widely used strategy — introduced in Week 1 for bandits, now applied to full RL control:

$$\pi_b(a \mid s) = \begin{cases} \text{uniform random action} & \text{with probability } \varepsilon \\ \arg\max_{a'} Q(s, a') & \text{with probability } 1 - \varepsilon \end{cases}$$

**Decaying ε** is a powerful improvement: start with high exploration, reduce over time as the agent gains knowledge.

$$\varepsilon_t = \max\left(\varepsilon_{\min},\; \varepsilon_0 \times \text{decay}^t\right)$$

A common schedule:

| Episode | $\varepsilon$ | Behavior |
|---------|--------------|---------|
| 1–100 | 1.0 → 0.5 | Mostly random — gathering broad experience |
| 100–500 | 0.5 → 0.1 | Balanced — refining knowledge |
| 500+ | 0.1 (fixed) | Mostly greedy — exploiting learned policy |

**Intuition:** Think of a new employee at a company. Week 1: try everything, ask everyone, observe all processes (high $\varepsilon$). After 6 months: you know what works — mostly use your experience, occasionally try new approaches (low $\varepsilon$).

### 5.3 Upper Confidence Bound (UCB) for Control

Recall UCB from Week 1's bandit problem. The same principle applies to full RL — prefer actions that are either high-value *or* insufficiently explored:

$$a_t = \arg\max_a \left[ Q(s_t, a) + c \sqrt{\frac{\ln t}{N(s_t, a)}} \right]$$

Where $N(s_t, a)$ is the number of times action $a$ has been taken from state $s_t$.

**Advantage over ε-greedy:** UCB explores *intelligently* — it targets the actions we're *uncertain* about, rather than exploring blindly at random. It naturally reduces exploration as $N(s,a)$ grows.

### 5.4 Optimistic Initialization

A surprisingly effective trick: initialize all Q-values to a **large positive value** (e.g., $Q(s, a) = 10$ for all pairs) instead of 0.

**Why it works:** The agent starts believing every action is great. When it tries an action and gets a lower-than-expected reward, that action's Q-value drops and looks *less attractive* than untried actions (which still have their high initial values). This naturally drives systematic exploration of all state-action pairs — no $\varepsilon$ required!

| Strategy | Mechanism | Strength | Weakness |
|----------|-----------|----------|---------|
| **ε-Greedy** | Random exploration probability | Simple, widely used | Explores blindly — may waste time on known-bad actions |
| **Decaying ε** | Reduces exploration over time | Better long-term performance | Requires careful tuning of decay schedule |
| **UCB** | Bonus for under-explored actions | Intelligent, targeted exploration | More complex; hard to extend to deep RL |
| **Optimistic Init** | High initial Q-values drive curiosity | No extra hyperparameters | Only works well in stationary environments |

---

## 6. SARSA vs. Q-Learning: The Cliff Walking Showdown

> *The most illuminating comparison between SARSA and Q-Learning is the "Cliff Walking" problem — a scenario where one algorithm's caution makes it safe, and the other's optimism makes it fast but risky.*

### 6.1 The Cliff Walking Environment

A classic benchmark from Sutton & Barto. The agent navigates a $4 \times 12$ grid:

```
┌────────────────────────────────────────┐
│  .  .  .  .  .  .  .  .  .  .  .  .  │  Row 3 (top)
│  .  .  .  .  .  .  .  .  .  .  .  .  │  Row 2
│  .  .  .  .  .  .  .  .  .  .  .  .  │  Row 1
│  S  C  C  C  C  C  C  C  C  C  C  G  │  Row 0 (bottom)
└────────────────────────────────────────┘
  S = Start,  C = Cliff (r = −100),  G = Goal,  . = Safe (r = −1)
```

- **Start (S):** Bottom-left corner
- **Goal (G):** Bottom-right corner
- **The Cliff:** All cells between S and G on the bottom row
- **Rewards:** −1 per step, −100 for stepping onto the Cliff (and reset to S), 0 at Goal

**Actions:** Up, Down, Left, Right

### 6.2 Two Different Paths Emerge

After training, the two algorithms consistently converge on **different strategies**:

```
SARSA path (safe but longer):          Q-Learning path (optimal but risky):
┌────────────────────────────────┐     ┌────────────────────────────────┐
│  →  →  →  →  →  →  →  →  →  ↓ │     │                                │
│  ↑                          ↓ │     │                                │
│  ↑                          ↓ │     │                                │
│  S  C  C  C  C  C  C  C  C  G │     │  S  →  →  →  →  →  →  →  →  G │
└────────────────────────────────┘     └────────────────────────────────┘
  Takes longer route away from cliff     Walks right along the cliff edge
  Expected reward ≈ −13 per episode      Expected reward ≈ −13 optimal
  (but safe — rarely falls off)          (but risky — sometimes falls off)
```

**Why the difference?**

| Algorithm | Strategy | Reason |
|-----------|----------|--------|
| **SARSA** | Safe path (away from cliff) | On-policy: evaluates the ε-greedy behavior, which includes random exploration that **could accidentally walk onto the cliff**. SARSA learns that being near the cliff is dangerous *given that you sometimes act randomly* |
| **Q-Learning** | Optimal path (along cliff) | Off-policy: always targets the best action ($\max$), ignoring the risk of random exploration. Learns the optimal path assuming perfect greedy behavior — which includes walking right next to the cliff |

**The deeper insight:** SARSA learns to be safe *relative to the current ε-greedy policy*, including its random mistakes. Q-Learning learns to be optimal *relative to the greedy policy*, ignoring exploration noise.

This is not a bug — it's a feature of each algorithm's design:
- **SARSA is safer** when the agent must perform well during training (e.g., a physical robot that breaks if it falls)
- **Q-Learning finds the true optimal** when training safety is less critical, and you want the best converged policy (e.g., video game AI)

### 6.3 Side-by-Side Update Comparison

One concrete transition to illustrate the difference:

**Setup:** Agent is in state $s$, takes action `Right`, gets $r = -1$, arrives at $s'$ (near the cliff).

At $s'$: $Q(s', \text{Right}) = -90$ (near cliff!), $Q(s', \text{Up}) = -5$ (safe direction).

The ε-greedy policy at $s'$ would choose `Up` (greedy) 90% of the time, but `Right` (random) 10% of the time.

| | SARSA | Q-Learning |
|--|-------|------------|
| **Next action $a'$ used in update** | $a_{t+1}$ from ε-greedy — could be `Right` (−90) | $\max_{a'} Q(s', a')$ — always `Up` (−5) |
| **Target value contribution** | $0.9 \times \underbrace{(-90)}_{10\%\text{ chance}}$ or $0.9 \times \underbrace{(-5)}_{90\%\text{ chance}}$ → **expected ≈ −13.5** | $0.9 \times (-5) = -4.5$ (always the best) |
| **Q(s, Right) updated toward** | **Lower value** (because risky next state under ε-greedy) | **Higher value** (ignores exploration risk) |
| **Result** | Agent learns to avoid $s'$ — stay away from the cliff | Agent happily approaches $s'$ — optimal ignoring noise |

---

## 7. Convergence of Q-Learning

### 7.1 Theoretical Guarantee

Q-Learning is guaranteed to converge to $Q^*$ (the true optimal Q-function) under two conditions:

**Condition 1 — GLIE (Greedy in the Limit with Infinite Exploration):**

Every state-action pair $(s, a)$ must be visited infinitely many times:

$$\sum_{t=1}^{\infty} \mathbf{1}[s_t = s, a_t = a] = \infty \quad \text{for all } s, a$$

In practice: use a slowly decaying $\varepsilon$ so every action gets tried eventually.

**Condition 2 — Robbins-Monro step-size conditions:**

The learning rate $\alpha_t$ must satisfy:

$$\sum_{t=1}^{\infty} \alpha_t = \infty \qquad \text{and} \qquad \sum_{t=1}^{\infty} \alpha_t^2 < \infty$$

**Intuition for these conditions:** The first sum being infinite ensures the algorithm never "gives up" — it always incorporates new information. The second sum being finite ensures the learning rate shrinks enough that the noise in the updates averages out. In practice, a fixed $\alpha$ (like 0.1 or 0.5) works very well even though it technically violates the second condition.

### 7.2 How Fast Does Q-Learning Converge?

**Factors that speed up convergence:**
- **More episodes** — obvious, but episodes must be diverse (explore well)
- **Higher α** — learns faster but can oscillate; find the sweet spot
- **Lower γ** — shorter effective planning horizon, faster propagation but suboptimal long-term reasoning
- **Smaller state/action spaces** — fewer Q-values to estimate

**A practical convergence check:** Plot the average reward per episode over training. If Q-Learning is working, you'll see a characteristic curve: low performance early (lots of exploration), then a rapid improvement, then a plateau near the optimal policy.

```
Average reward per episode
      │                                              ░░░░░░░░
      │                                        ░░░░░░
      │                                   ░░░░░
      │                              ░░░░░
      │                        ░░░░░░
  ────┼───────────────░░░░░░░░░░────────────────────────────── Episodes
      0           100         500        1000
      ↑ Random exploration   ↑ Q-values stabilizing   ↑ Converged
```

---

## 8. Extensions and Connections

### 8.1 Double Q-Learning — Fixing the Maximization Bias

Q-Learning has a subtle flaw: the $\max$ operator introduces **optimism bias**.

**The problem:** Suppose at a new state $s'$, all actions have true value 0, but due to random noise in early estimates, some Q-values are positive. Taking the $\max$ will always pick the noisiest-positive one, systematically overestimating the value of $s'$.

**Double Q-Learning** (van Hasselt, 2010) fixes this by maintaining **two separate Q-tables** and decoupling action selection from value estimation:

$$\text{Standard Q-Learning target: } r + \gamma \cdot Q\bigl(s',\; \arg\max_{a'} Q(s', a')\bigr)$$

$$\text{Double Q-Learning target: } r + \gamma \cdot Q_B\bigl(s',\; \arg\max_{a'} Q_A(s', a')\bigr)$$

- $Q_A$ selects which action is best
- $Q_B$ evaluates how good that action actually is

Using separate tables for selection and evaluation removes the systematic optimism. This technique became critical in **DQN** (Deep Q-Network, Week 5).

### 8.2 Expected SARSA — The Middle Ground

**Expected SARSA** replaces the SARSA target $Q(s', a')$ with the **expected** Q-value under the current policy:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \Bigl[ r + \gamma \sum_{a'} \pi(a' \mid s') Q(s', a') - Q(s, a) \Bigr]$$

For an ε-greedy policy:
$$\sum_{a'} \pi(a' \mid s') Q(s', a') = \frac{\varepsilon}{|\mathcal{A}|} \sum_{a'} Q(s', a') + (1-\varepsilon) \max_{a'} Q(s', a')$$

**Where it sits:** Expected SARSA is between SARSA (on-policy) and Q-Learning (fully off-policy):

```
SARSA                Expected SARSA               Q-Learning
(most on-policy) ──────────────────────────── (fully off-policy)
High variance      Lower variance, less bias      Lowest variance
Safest during      Good all-rounder               Optimal converged
training                                          policy
```

### 8.3 The Road to Deep RL

Tabular Q-Learning — the algorithm from this week — breaks down when the state space is large. Consider:

| Environment | State space size | Tabular Q-table size |
|-------------|-----------------|---------------------|
| 3-room robot (Week 1) | 3 states | 3 rows |
| GridWorld 5×5 | 25 states | 25 rows |
| **Atari Pong** | ~$10^{26}$ pixel arrangements | **Impossible to store** |
| **Autonomous driving** | Continuous sensor readings | **Infinite** |

The solution: replace the Q-table with a **neural network** that approximates $Q(s, a; \theta)$. This is **Deep Q-Network (DQN)** — the topic of Week 5. Q-Learning is literally the same algorithm; we just swap the lookup table for a neural network and add stabilization tricks.

---

## 9. Real-World Applications

| Application | Algorithm | How It Works |
|-------------|-----------|-------------|
| **Atari game playing** (DeepMind, 2013) | Q-Learning + DQN | Agent plays 49 Atari games from raw pixels; DQN learns $Q^*$ using a convolutional neural network as the Q-function |
| **AlphaGo / AlphaZero** | Q-Learning + Monte Carlo Tree Search | Q-values guide where to search; MCTS refines them with rollouts |
| **Robot arm control** (OpenAI) | Q-Learning variants | Learns to grasp objects by trial and error in simulation |
| **Elevator scheduling** | Tabular Q-Learning | Dispatches elevators to minimize passenger wait times — a classic RL benchmark |
| **Network packet routing** | Q-Learning | Routers learn optimal forwarding rules by observing network latency |
| **Drug dosage optimization** | Q-Learning | Learns treatment regimens from patient outcome data — off-policy allows using historical clinical records |
| **Dialogue systems** | Q-Learning | Conversational agents learn to steer dialogue toward goal completion (booking a flight, resolving a complaint) |

**Why Q-Learning dominates in practice:**

Off-policy learning means Q-Learning can learn from **any data source** — historical records, simulations, human demonstrations, or even experience from other agents. This "sample reuse" is enormously valuable in expensive real-world settings where every interaction costs time, money, or physical wear.

---

## 10. Summary

| Concept | Key Idea |
|---------|----------|
| **On-Policy** | Behavior and target policy are the same (SARSA) — learns the value of what you're doing |
| **Off-Policy** | Behavior and target policy differ (Q-Learning) — learns the optimal value while exploring |
| **Behavior policy** | The policy used to collect experience — usually ε-greedy |
| **Target policy** | The policy being optimized — greedy ($\arg\max$) in Q-Learning |
| **Q-Learning update** | $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ |
| **SARSA update** | $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$ |
| **The $\max$ operator** | Q-Learning always bootstraps from the best next action — this is what makes it off-policy and optimal |
| **ε-Greedy** | Explore randomly with probability $\varepsilon$; exploit with $1-\varepsilon$ |
| **Decaying ε** | Gradually reduce exploration as the agent gains confidence |
| **GLIE** | Convergence condition: every state-action pair must be visited infinitely often |
| **Cliff Walking** | SARSA learns the safer path; Q-Learning learns the optimal (riskier) path |
| **Maximization bias** | $\max$ over noisy estimates causes overestimation — fixed by Double Q-Learning |
| **Expected SARSA** | A middle ground: uses expected Q-value under the policy instead of a sampled action |
| **Tabular → Deep RL** | Replace the Q-table with a neural network → DQN (Week 5) |

---

## 11. Exercises

### Conceptual Questions

1. **What is the one mathematical difference** between the SARSA update and the Q-Learning update? Explain in one sentence what that difference means algorithmically and why it makes Q-Learning off-policy.

2. **In the Cliff Walking example**, after full convergence, Q-Learning finds the optimal path (along the cliff edge). Yet during training, Q-Learning performs *worse* than SARSA on average reward per episode. Why? And when training is done and both agents act greedily, which one performs better?

3. **Explain the maximization bias** using an analogy. Suppose you're evaluating 5 new restaurants you've never been to. Their true quality is all identical (average), but your first guesses about each are random due to limited information. Why would always picking the highest-rated restaurant lead to systematically disappointed experiences over time?

4. **A self-driving car** is learning to navigate a busy intersection. Would you prefer to train it with SARSA or Q-Learning? Justify your answer by connecting to the on-policy/off-policy distinction and the consequences of exploration during training.

---

### Calculation Problems

**Problem 1 — Q-Learning Update**

Current Q-table:

| State | $Q(\cdot, \text{Forward})$ | $Q(\cdot, \text{Turn})$ |
|-------|----------------------------|-------------------------|
| X | 4.0 | 7.0 |
| Y | 2.0 | 9.0 |

The agent is in state $X$, chooses action `Turn` (greedy), receives reward $r = +3$, and transitions to state $Y$.

Parameters: $\alpha = 0.4$, $\gamma = 0.9$

(a) What is the Q-Learning target?

(b) What is the TD error $\delta$?

(c) What is the updated $Q(X, \text{Turn})$?

---

**Problem 2 — SARSA vs. Q-Learning Comparison**

Same Q-table and transition as Problem 1. The agent is at $X$, chose `Turn`, arrives at $Y$ with $r = +3$.

At $Y$, the ε-greedy policy (with $\varepsilon = 0.2$) selects `Forward` as the next action $a'$.

(a) Compute the SARSA target (using $a' = \text{Forward}$).

(b) Compute the SARSA updated $Q(X, \text{Turn})$.

(c) Compare with your Q-Learning answer from Problem 1. Which update is larger and why?

---

**Problem 3 — Convergence Tracing**

A robot can be in states $\{A, B, \text{Goal}\}$. The only reward is $+10$ upon entering Goal from B. All else gives $r = 0$. $\gamma = 1.0$ (no discounting), $\alpha = 1.0$ (replace old value entirely).

Initial Q-table: all zeros.

Trace **two complete episodes** of Q-Learning where the robot always goes $A \to B \to \text{Goal}$:

(a) After Episode 1, what are $Q(B, \text{Right})$ and $Q(A, \text{Right})$?

(b) After Episode 2, what are $Q(B, \text{Right})$ and $Q(A, \text{Right})$?

(c) What are the true optimal values $Q^*(A, \text{Right})$ and $Q^*(B, \text{Right})$ with $\gamma = 1.0$? Is the agent converging in the right direction?

---

**Problem 4 — Exploration Decision**

At step $t = 20$, the agent uses decaying ε-greedy with $\varepsilon_0 = 1.0$ and decay rate $= 0.9$ per episode (current episode = 4).

(a) Calculate $\varepsilon$ at episode 4 (after 3 full decay steps from $\varepsilon_0$).

(b) The agent is in state $s$ with $Q(s, A) = 3.5$ and $Q(s, B) = 2.1$. A random draw gives $u = 0.12$. What action does the agent take?

(c) If the draw had been $u = 0.07$, what action would be taken, and which arm would be explored?

---

### Answer Key

**Problem 1:**
- (a) Q-Learning target $= r + \gamma \max_{a'} Q(Y, a') = 3 + 0.9 \times \max(2.0, 9.0) = 3 + 0.9 \times 9 = 3 + 8.1 = \mathbf{11.1}$
- (b) $\delta = 11.1 - 7.0 = \mathbf{+4.1}$
- (c) $Q(X, \text{Turn}) \leftarrow 7.0 + 0.4 \times 4.1 = 7.0 + 1.64 = \mathbf{8.64}$

**Problem 2:**
- (a) SARSA target $= r + \gamma Q(Y, \text{Forward}) = 3 + 0.9 \times 2.0 = 3 + 1.8 = \mathbf{4.8}$
- (b) $Q(X, \text{Turn}) \leftarrow 7.0 + 0.4(4.8 - 7.0) = 7.0 + 0.4(-2.2) = 7.0 - 0.88 = \mathbf{6.12}$
- (c) Q-Learning (8.64) > SARSA (6.12). Q-Learning used $\max_{a'} Q(Y, \cdot) = Q(Y, \text{Turn}) = 9.0$ — the best possible next action. SARSA used the actually-chosen action $Q(Y, \text{Forward}) = 2.0$, which was a worse move. Q-Learning is more optimistic because it always imagines the agent will act perfectly from the next state.

**Problem 3:**

*Episode 1:*

- Step B→Goal: target $= 10 + 1.0 \times 0 = 10$; $Q(B, R) \leftarrow 0 + 1.0(10 - 0) = \mathbf{10}$
- Step A→B: target $= 0 + 1.0 \times \max Q(B, \cdot) = 0 + 10 = 10$; $Q(A, R) \leftarrow 0 + 1.0(10 - 0) = \mathbf{10}$

(a) After Episode 1: $Q(B, \text{Right}) = \mathbf{10}$, $Q(A, \text{Right}) = \mathbf{10}$

*Episode 2:*

- Step B→Goal: target $= 10$; $Q(B, R) \leftarrow 10 + 1.0(10 - 10) = \mathbf{10}$ (unchanged)
- Step A→B: target $= 0 + 10 = 10$; $Q(A, R) \leftarrow 10 + 1.0(10 - 10) = \mathbf{10}$ (unchanged)

(b) After Episode 2: $Q(B, \text{Right}) = \mathbf{10}$, $Q(A, \text{Right}) = \mathbf{10}$

(c) True values with $\gamma = 1$: $Q^*(B, R) = 10$, $Q^*(A, R) = 0 + 1.0 \times 10 = 10$. The agent converged in one episode because $\alpha = 1.0$ replaces the old value entirely — rapid with a deterministic environment.

**Problem 4:**
- (a) $\varepsilon = 1.0 \times 0.9^3 = 1.0 \times 0.729 = \mathbf{0.729}$
- (b) $u = 0.12 < \varepsilon = 0.729$ → **Explore** → random action (could be either A or B)
- (c) $u = 0.07 < 0.729$ → **Explore** → random action. The agent would randomly select one of the two actions. If it selected B, B is explored (the non-greedy action).

---

*Coming up in Week 5 — Deep Q-Networks (DQN): we'll take Q-Learning and give it a neural network "brain," enabling agents to master Atari games directly from raw pixel inputs — one of the landmark achievements in modern AI.*
