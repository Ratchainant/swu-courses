# Week 3 — Model-Free Prediction & Control

---

## 1. Why "Model-Free"? The Real-World Problem

> *Imagine you're dropped into a foreign city with no map, no GPS, and no guidebook. How do you find the best route to the hotel? You walk, you observe, you try different streets, and you gradually learn which paths are fast and which are dead ends — entirely from your own experience.*

### The Limits of Dynamic Programming

In Week 2, we learned that **Dynamic Programming (DP)** can compute the exact optimal policy — but only when the agent knows the **complete MDP**: every transition probability $\mathcal{T}(s' \mid s, a)$ and every reward $\mathcal{R}(s, a, s')$.

This is a luxury that almost never exists in practice.

**Think about these scenarios:**
- A robot learning to walk has no mathematical formula for how muscles, joints, and gravity interact.
- A stock-trading agent has no model of how news events will move the market tomorrow.
- A medical AI deciding treatment protocols can't know exactly how a specific patient's body will respond.

In all these cases, the agent is **blind to the rules** — it can only observe what actually *happens* after each action.

**Model-Free RL** is the answer: learn the value of states and actions **directly from experience** — from real sequences of states, actions, and rewards — without ever needing to know the transition probabilities $\mathcal{T}$ or reward function $\mathcal{R}$ in advance.

| Feature | Dynamic Programming | Model-Free RL |
|---------|--------------------|--------------------|
| Needs $\mathcal{T}(s' \mid s, a)$? | ✓ Yes (required) | ✗ No |
| Needs $\mathcal{R}(s, a, s')$? | ✓ Yes (required) | ✗ No |
| Learns from? | Mathematical computation | Real experience (episodes) |
| Works in complex environments? | ✗ Only when rules are known | ✓ Yes |

---

## 2. Two Problems: Prediction and Control

Before diving into algorithms, we distinguish two fundamental tasks in RL:

**Prediction** — *"How good is it to follow this policy?"*

> Given a fixed policy $\pi$, estimate the value function $V^\pi(s)$ for every state. The agent evaluates how well a given strategy performs.

**Control** — *"What is the best policy?"*

> Find the optimal policy $\pi^*$ by improving it iteratively. The agent not only evaluates but also optimizes its strategy.

**Intuitive Analogy:**
- **Prediction** is like a student calculating their expected final exam grade given how much they currently study.
- **Control** is like that student then adjusting their study habits to maximize the grade.

This week covers **model-free** versions of both. We start with Monte Carlo methods, then Temporal-Difference learning.

---

## 3. Monte Carlo Methods

> *A weather forecaster doesn't know tomorrow's weather with certainty — but if they watch 1,000 similar weather patterns from history and see that 750 lead to rain, they conclude there's a 75% chance of rain. Monte Carlo does exactly this — it learns from complete experience.*

### 3.1 The Core Idea

**Monte Carlo (MC)** methods learn value functions by averaging the **actual returns** observed after visiting a state across many **complete episodes**.

An **episode** is one full run from a starting state to a terminal state — like one game of chess from start to checkmate/draw, or one trip from home to work.

**Key insight:** The value of a state $V(s)$ is the expected return from that state. So if we observe the actual return $G_t$ every time we visit state $s$ across many episodes, the average of those returns converges to $V^\pi(s)$.

$$V^\pi(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}$$

Where:
- $N(s)$ = number of times state $s$ has been visited across all episodes
- $G_t^{(i)}$ = the actual discounted return observed starting from state $s$ in episode $i$

### 3.2 Intuitive Example — Scoring a Board Game Position

Imagine a simplified board game. A player always starts at position **A**, moves through positions **B**, **C**, and finally **WIN** (terminal) or **LOSE** (terminal).

You want to learn: *"How valuable is it to be in position B?"*

You play 5 complete games from position B onward and record the total reward received in each game:

| Game | Sequence from B | Total Return $G$ from B |
|------|-----------------|-------------------------|
| 1 | B → C → WIN (+10) | +8.1 (discounted) |
| 2 | B → C → LOSE (−5) | −3.95 |
| 3 | B → C → WIN (+10) | +8.1 |
| 4 | B → C → WIN (+10) | +8.1 |
| 5 | B → C → LOSE (−5) | −3.95 |

After 5 games:

$$V(B) \approx \frac{8.1 + (-3.95) + 8.1 + 8.1 + (-3.95)}{5} = \frac{16.4}{5} = \mathbf{3.28}$$

After 50 games, 500 games, this average will converge to the true $V^\pi(B)$. No knowledge of transition probabilities was needed — only lived experience.

### 3.3 First-Visit vs. Every-Visit MC

Within a single episode, a state may be visited **multiple times**. There are two ways to handle this:

| Variant | Rule | Characteristic |
|---------|------|----------------|
| **First-Visit MC** | Only count the return from the **first** visit to $s$ in each episode | Unbiased estimate, statistically cleaner |
| **Every-Visit MC** | Count the return from **every** visit to $s$ in each episode | More data points, can be slightly biased but converges |

**Example:** In one episode, the agent visits state $C$ twice: at step $t=3$ and $t=7$.

- **First-Visit MC** records only the return from $t=3$.
- **Every-Visit MC** records returns from both $t=3$ and $t=7$.

In practice, **First-Visit MC** is more common and is what we'll use in our calculations.

### 3.4 The Running Average Update

Computing the full average from scratch after every episode is wasteful. Instead, we use an **incremental update rule** — the same mathematical trick used to compute a running average efficiently.

After observing a new return $G$, update the value estimate:

$$V(s) \leftarrow V(s) + \frac{1}{N(s)} \bigl(G - V(s)\bigr)$$

Or, using a fixed **learning rate** $\alpha$ (which forgets old data gradually — useful in non-stationary environments):

$$V(s) \leftarrow V(s) + \alpha \bigl(G - V(s)\bigr)$$

**Intuition:** The term $(G - V(s))$ is the **prediction error** — the gap between what actually happened ($G$) and what was expected ($V(s)$). We nudge the estimate in the direction of reality, scaled by $\alpha$.

Think of it like adjusting a restaurant rating. If your expected rating for a restaurant is 3.5/5 and your new visit gives a 5/5 experience, you update your belief upward — but not all the way to 5, because past experiences matter too.

---

### 3.5 Worked Calculation: First-Visit Monte Carlo Prediction

**Environment:** A student navigating a study day. States: **Morning (M)**, **Afternoon (A)**, **Evening (E)**, **Done (D)** (terminal). The student follows a fixed policy: always study.

**Reward structure:**
- Each hour of productive study: $+2$
- Feeling burned out at the end of the day: $-5$
- Successfully completing all work: $+10$

**Discount factor:** $\gamma = 0.9$

**Initial estimates:** $V(M) = V(A) = V(E) = 0$, $V(D) = 0$

We observe **3 complete episodes**:

---

**Episode 1:** M → A → E → D(complete)

Rewards: $r_1 = +2$ (M→A), $r_2 = +2$ (A→E), $r_3 = +10$ (E→D complete)

Compute returns working **backwards** from the terminal state:

$$G(E) = r_3 = +10$$

$$G(A) = r_2 + \gamma \cdot G(E) = 2 + 0.9 \times 10 = 2 + 9 = \mathbf{11}$$

$$G(M) = r_1 + \gamma \cdot G(A) = 2 + 0.9 \times 11 = 2 + 9.9 = \mathbf{11.9}$$

Update values (learning rate $\alpha = 0.5$):

$$V(E) \leftarrow 0 + 0.5(10 - 0) = \mathbf{5.0}$$
$$V(A) \leftarrow 0 + 0.5(11 - 0) = \mathbf{5.5}$$
$$V(M) \leftarrow 0 + 0.5(11.9 - 0) = \mathbf{5.95}$$

---

**Episode 2:** M → A → E → D(burned out)

Rewards: $r_1 = +2$, $r_2 = +2$, $r_3 = -5$ (burned out)

$$G(E) = -5$$
$$G(A) = 2 + 0.9 \times (-5) = 2 - 4.5 = \mathbf{-2.5}$$
$$G(M) = 2 + 0.9 \times (-2.5) = 2 - 2.25 = \mathbf{-0.25}$$

Update values (from Episode 1 estimates):

$$V(E) \leftarrow 5.0 + 0.5(-5 - 5.0) = 5.0 - 5.0 = \mathbf{0.0}$$
$$V(A) \leftarrow 5.5 + 0.5(-2.5 - 5.5) = 5.5 - 4.0 = \mathbf{1.5}$$
$$V(M) \leftarrow 5.95 + 0.5(-0.25 - 5.95) = 5.95 - 3.1 = \mathbf{2.85}$$

---

**Episode 3:** M → A → E → D(complete)

Same rewards as Episode 1: $G(E)=10,\; G(A)=11,\; G(M)=11.9$

$$V(E) \leftarrow 0.0 + 0.5(10 - 0.0) = \mathbf{5.0}$$
$$V(A) \leftarrow 1.5 + 0.5(11 - 1.5) = 1.5 + 4.75 = \mathbf{6.25}$$
$$V(M) \leftarrow 2.85 + 0.5(11.9 - 2.85) = 2.85 + 4.525 = \mathbf{7.375}$$

**Summary after 3 episodes:**

| State | After Ep 1 | After Ep 2 | After Ep 3 |
|-------|-----------|-----------|-----------|
| M | 5.95 | 2.85 | **7.375** |
| A | 5.5 | 1.5 | **6.25** |
| E | 5.0 | 0.0 | **5.0** |

**Interpretation:** Values are fluctuating because we have only seen 3 episodes. With hundreds of episodes, the averages stabilize to the true $V^\pi(s)$. Notice the estimates reflect the "risky" nature of the policy — about half the time the student burns out.

### 3.6 Strengths and Weaknesses of Monte Carlo

| Strength | Weakness |
|----------|----------|
| ✓ No model needed ($\mathcal{T}$, $\mathcal{R}$ unknown is fine) | ✗ Must wait for the **complete episode** to end before updating |
| ✓ Unbiased estimate of $V^\pi(s)$ | ✗ High **variance** — episodes can be very different from each other |
| ✓ Simple to implement | ✗ Cannot handle **continuing tasks** (no terminal state) |
| ✓ Only visits states on the actual path | ✗ Learning is slow — updates happen once per episode |

**The waiting problem is real.** In a chess game, you must play the entire game before learning anything. In a real robot, you must let it run until it crashes or succeeds before updating. This can be very slow.

**Temporal-Difference learning**, developed next, solves exactly this problem.

---

## 4. Temporal-Difference Learning

> *A meteorologist doesn't wait for the entire year to pass before updating tomorrow's forecast. As soon as new data arrives — this morning's humidity reading — they update their prediction. TD learning does the same for RL agents.*

### 4.1 The Core Idea

**Temporal-Difference (TD) Learning** is the single most important idea in modern RL. It combines the best features of Monte Carlo and Dynamic Programming:

- Like **Monte Carlo**: learns directly from experience without needing a model.
- Like **Dynamic Programming**: updates estimates **immediately**, without waiting for the episode to end — by using an estimate of the *next* state's value.

This "update now using another estimate" idea is called **bootstrapping**.

### 4.2 The TD(0) Update Rule

The simplest TD algorithm, **TD(0)**, updates the value of a state after each single step:

$$V(s_t) \leftarrow V(s_t) + \alpha \bigl[\underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD target}} - \underbrace{V(s_t)}_{\text{current estimate}}\bigr]$$

The key components:

| Term | Name | Meaning |
|------|------|---------|
| $V(s_t)$ | Current estimate | What the agent currently thinks state $s_t$ is worth |
| $r_{t+1}$ | Immediate reward | The actual reward just received |
| $V(s_{t+1})$ | Bootstrap estimate | The agent's current guess of the next state's value |
| $r_{t+1} + \gamma V(s_{t+1})$ | **TD target** | A better, one-step-improved estimate of $V(s_t)$ |
| $\delta_t = \text{TD target} - V(s_t)$ | **TD error** | The surprise — how wrong the current estimate was |

**The TD error $\delta_t$** is the heartbeat of all TD methods. It measures: *"How much better or worse did things turn out compared to what I expected?"*

### 4.3 Intuitive Example — Commuting to Work

You commute from home to office every day. You want to estimate how long the drive takes from each waypoint.

You leave home at 8:00 AM. Your current estimates:

| Location | Estimated minutes to office |
|----------|----------------------------|
| Home (H) | 45 min |
| Highway (HW) | 30 min |
| City Center (CC) | 15 min |
| Office (O) | 0 min (terminal) |

**Today:** You leave Home, drive 10 minutes, arrive at Highway. Traffic is light! You now think the rest of the drive from Highway will take only **20 min** (instead of your old estimate of 30).

The **TD update** for Home:

$$V(\text{Home}) \leftarrow 45 + \alpha\bigl[(10 + V(\text{Highway}_{\text{new}})) - 45\bigr]$$

Using $\alpha = 0.5$ and the improved Highway estimate of 20 min:

$$V(\text{Home}) \leftarrow 45 + 0.5\bigl[(10 + 20) - 45\bigr] = 45 + 0.5(-15) = \mathbf{37.5 \text{ min}}$$

You've updated your commute estimate from 45 to 37.5 minutes — **before reaching the office** — because you used the intermediate observation (arriving at Highway quickly) plus your updated estimate for the remaining journey. That's bootstrapping.

---

### 4.4 Worked Calculation: TD(0) Step by Step

**Environment:** A delivery robot navigating from **Warehouse (W)** → **Street (S)** → **Customer (C)** (terminal). The robot follows a fixed policy.

**Rewards:**
- W → S: $r = -1$ (travel cost)
- S → C: $r = +10$ (successful delivery)

**Discount factor:** $\gamma = 0.9$, **Learning rate:** $\alpha = 0.5$

**Initial estimates:** $V(W) = 0$, $V(S) = 0$, $V(C) = 0$

---

**Episode 1, Step 1:** State $s_t = W$, action = move, observe $r = -1$, arrive at $s_{t+1} = S$

**TD error:**
$$\delta = r + \gamma V(S) - V(W) = -1 + 0.9 \times 0 - 0 = \mathbf{-1}$$

**Update $V(W)$:**
$$V(W) \leftarrow 0 + 0.5 \times (-1) = \mathbf{-0.5}$$

Current estimates: $V(W) = -0.5$, $V(S) = 0$

---

**Episode 1, Step 2:** State $s_t = S$, action = move, observe $r = +10$, arrive at $s_{t+1} = C$ (terminal)

**TD error:**
$$\delta = r + \gamma V(C) - V(S) = 10 + 0.9 \times 0 - 0 = \mathbf{+10}$$

**Update $V(S)$:**
$$V(S) \leftarrow 0 + 0.5 \times 10 = \mathbf{5.0}$$

End of Episode 1. Current estimates: $V(W) = -0.5$, $V(S) = 5.0$

---

**Episode 2, Step 1:** State $s_t = W$, arrive at $s_{t+1} = S$, $r = -1$

Now $V(S)$ has been updated to 5.0, so the TD error is different:

$$\delta = -1 + 0.9 \times 5.0 - (-0.5) = -1 + 4.5 + 0.5 = \mathbf{4.0}$$

**Update $V(W)$:**
$$V(W) \leftarrow -0.5 + 0.5 \times 4.0 = -0.5 + 2.0 = \mathbf{1.5}$$

---

**Episode 2, Step 2:** State $s_t = S$, $r = +10$, $s_{t+1} = C$

$$\delta = 10 + 0 - 5.0 = \mathbf{5.0}$$

$$V(S) \leftarrow 5.0 + 0.5 \times 5.0 = \mathbf{7.5}$$

---

**Tracking value convergence across episodes:**

| Episode | $V(W)$ after ep | $V(S)$ after ep |
|---------|----------------|----------------|
| 0 (init) | 0.0 | 0.0 |
| 1 | −0.5 | 5.0 |
| 2 | 1.5 | 7.5 |
| 3 | 4.3 | 8.75 |
| … | … | … |
| Converged | **≈ 7.1** | **≈ 8.0** |

**Verify the converged values make sense:**

The true value of $V(S)$ can be calculated analytically:

$$V^{\pi}(S) = 0.9^0 \times 10 = 10 \quad \text{(one step away, discounted by } \gamma^1\text{ — wait, we earn +10 immediately)}$$

Actually: $V^{\pi}(S) = 1.0 \times [+10 + 0.9 \times 0] = +10$ (if no stochasticity). Our estimate converges toward this. The gap from 10 is due to $\alpha=0.5$ and only a few episodes — with more episodes, it approaches the true value.

$$V^{\pi}(W) = 1.0 \times [-1 + 0.9 \times V(S)] = -1 + 0.9 \times 10 = \mathbf{8.0}$$

The TD algorithm is slowly "propagating" the delivery reward backward through the states — first $S$ learns it's near a big reward, then $W$ learns that $S$ is valuable, so $W$ itself becomes valuable. This **credit assignment through bootstrapping** is the magic of TD.

---

### 4.5 Why Does TD Work Without Waiting?

This is the deep insight: TD doesn't need the complete return $G_t = r_1 + \gamma r_2 + \gamma^2 r_3 + \cdots$ to update. It uses the **one-step look ahead**:

$$\text{TD target} = r_{t+1} + \gamma V(s_{t+1})$$

It's like saying: *"I don't know how the rest of the journey goes, but I just received one real data point ($r_{t+1}$), and I trust my current estimate of where I'll be next ($V(s_{t+1})$) — that's enough to make progress."*

This is **bootstrapping**: using your own (imperfect) estimates to improve your estimates. Over many steps, the estimates correct themselves iteratively.

---

## 5. Monte Carlo vs. Temporal-Difference: A Deep Comparison

> *MC is like studying for an exam by completing full practice papers. TD is like updating your understanding after every question you get back. Both teach you the same material, but they have very different styles.*

### 5.1 Bias and Variance

These two properties fundamentally characterize the tradeoff between MC and TD:

**Bias** — Is the estimate systematically wrong in a particular direction?

- **MC is unbiased**: the return $G_t$ is the actual cumulative reward, so averaging it gives the true $V^\pi(s)$ in the long run.
- **TD is biased**: the TD target $r + \gamma V(s')$ uses $V(s')$, which is an imperfect estimate. The bias gradually disappears as $V(s')$ improves.

**Variance** — How much do estimates fluctuate between different samples?

- **MC has high variance**: the return $G_t$ depends on the entire episode, which can be highly variable (one game of chess can unfold very differently from another).
- **TD has low variance**: it only depends on one transition $r_{t+1}$, which is much more stable.

| Property | Monte Carlo | TD(0) |
|----------|-------------|-------|
| **Bias** | None (unbiased) | Initial bias (disappears over time) |
| **Variance** | High (full episode return) | Low (single-step transition) |
| **Update timing** | End of episode only | After every single step |
| **Handles continuing tasks?** | ✗ No | ✓ Yes |
| **Convergence** | To $V^\pi$ (guaranteed) | To $V^\pi$ (guaranteed with proper $\alpha$) |
| **Data efficiency** | Less efficient | More efficient |

### 5.2 The Backup Diagram

A powerful way to visualize what each method "looks at" when updating:

```
Monte Carlo — Full Trajectory Backup:
             s_t
              |  (r_{t+1})
             s_{t+1}
              |  (r_{t+2})
             s_{t+2}
              |  ...
             s_T   ← Terminal: actual G_t computed here, then propagated back

TD(0) — One-Step Backup:
             s_t
              |  (r_{t+1})
             s_{t+1}  ← Stop here. Use V(s_{t+1}) as a proxy for the rest.
```

MC looks all the way to the end of the episode.
TD looks just one step ahead — and uses its own estimate for the rest.

### 5.3 The Spectrum: n-Step TD

There's actually a **spectrum** between MC and TD(0), called **n-step TD**:

$$G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})$$

| $n$ | Name | Behavior |
|-----|------|---------|
| $n = 1$ | TD(0) | Minimal lookahead, low variance, biased |
| $n = 2$ | 2-step TD | Slightly more lookahead |
| $n = 5$ | 5-step TD | Balancing variance and bias |
| $n = \infty$ | Monte Carlo | Full episode, unbiased, high variance |

**TD($\lambda$)** (covered in later courses) elegantly combines all $n$-step returns using a parameter $\lambda \in [0, 1]$, giving practitioners a single knob to tune the bias-variance tradeoff.

---

## 6. Model-Free Control: From Prediction to Improvement

> *Knowing how good your current strategy is (prediction) is only half the story. The real goal is to find a BETTER strategy (control). How do we do this without knowing the environment's rules?*

### 6.1 The Control Problem

So far, we've assumed the agent follows a **fixed policy** $\pi$ and tries to estimate $V^\pi(s)$. But in control, the agent must also **improve** the policy.

In Dynamic Programming, we did **policy improvement** using the known model:

$$\pi'(s) = \arg\max_a \sum_{s'} \mathcal{T}(s' \mid s, a) \bigl[\mathcal{R}(s, a, s') + \gamma V(s')\bigr]$$

But without $\mathcal{T}$ and $\mathcal{R}$, we **can't** evaluate this expression!

**The solution:** Learn $Q(s, a)$ instead of $V(s)$.

### 6.2 Why We Need Q(s, a) for Model-Free Control

The **action-value function** $Q^\pi(s, a)$ is the expected return from state $s$, taking action $a$ first, then following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$$

**Crucial advantage:** Improving the policy from $Q$ requires *no model*:

$$\pi'(s) = \arg\max_a Q(s, a)$$

We simply pick the action with the highest $Q$-value — no need to know $\mathcal{T}$ or $\mathcal{R}$.

**Intuition:** Think of $Q(s, a)$ as a restaurant menu with scores. To choose the best dish (action) at a restaurant (state), you just pick the highest-scored item on the menu — you don't need to know the recipe ($\mathcal{T}$) or the chef's cost structure ($\mathcal{R}$).

### 6.3 Generalized Policy Iteration (GPI)

Both MC control and TD control use the same alternating loop called **Generalized Policy Iteration (GPI)**:

```
┌─────────────────────────────────────┐
│                                     │
│   Policy Evaluation                 │
│   Estimate Q^π(s,a) from experience │
│            ↓                        │
│   Policy Improvement                │
│   π'(s) = argmax_a Q(s,a)          │
│            ↓                        │
│   (repeat until convergence)        │
│            ↑_____________________   │
│                                     │
└─────────────────────────────────────┘
```

The key question is: **how do we do the evaluation step?** MC and TD differ here.

### 6.4 The Exploration Problem: ε-Greedy Policies

There's a trap: if we always follow the greedy policy $\pi(s) = \arg\max_a Q(s, a)$, we **never explore** — so some actions will never be tried, and their $Q$-values will stay at zero forever. We might miss the best actions entirely!

**Solution:** Use **ε-greedy** policy during control:

$$\pi(a \mid s) = \begin{cases} \text{random action} & \text{with probability } \varepsilon \\ \arg\max_a Q(s, a) & \text{with probability } 1 - \varepsilon \end{cases}$$

This ensures all state-action pairs continue to be visited (called **GLIE** — Greedy in the Limit with Infinite Exploration). A common strategy is to start with a large $\varepsilon$ (lots of exploration early) and gradually reduce it as the agent gains experience.

---

## 7. SARSA — On-Policy TD Control

> *"SARSA" sounds like a word but it's actually an acronym that describes exactly what data each update uses: the current State, the Action taken, the Reward received, the next State, and the next Action.*

### 7.1 The SARSA Update

**SARSA** (State-Action-Reward-State-Action) is the on-policy TD control algorithm. It extends TD(0) to update **Q-values** instead of state values:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \bigl[\underbrace{r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})}_{\text{SARSA target}} - Q(s_t, a_t)\bigr]$$

The name SARSA comes from the quintuple $(s_t,\; a_t,\; r_{t+1},\; s_{t+1},\; a_{t+1})$ — these five quantities are all that's needed for one update.

**What "on-policy" means:** SARSA evaluates and improves the **same policy** that is being used to collect experience (the ε-greedy policy). This is important — the Q-value of $(s_{t+1}, a_{t+1})$ uses the **actual next action** chosen by the current policy.

### 7.2 The SARSA Algorithm

```
Initialize Q(s, a) = 0 for all states s, actions a
Set ε (e.g., 0.1), α (e.g., 0.5), γ (e.g., 0.9)

For each episode:
    Start at initial state s
    Choose action a from Q using ε-greedy

    For each step until terminal state:
        Take action a, observe reward r and next state s'
        Choose next action a' from Q(s', ·) using ε-greedy

        Update: Q(s, a) ← Q(s, a) + α[r + γ·Q(s', a') − Q(s, a)]

        s ← s'
        a ← a'
```

---

### 7.3 Worked Calculation: SARSA on a Grid

**Environment:** A $1 \times 3$ grid: **[Start]** → **[Middle]** → **[Goal]**

Actions: **Right** (move forward) or **Stay** (do nothing, waste a turn)

**Rewards:**
- Any step: $r = -1$ (time cost — the agent is penalized for taking time)
- Reaching Goal: $r = +10$
- Staying: $r = -2$ (extra penalty for wasting time)

**Discount factor:** $\gamma = 0.9$, **Learning rate:** $\alpha = 0.5$, **$\varepsilon = 0.1$**

**Initial Q-table** (all zeros):

| State | $Q(\cdot, \text{Right})$ | $Q(\cdot, \text{Stay})$ |
|-------|--------------------------|-------------------------|
| Start | 0.0 | 0.0 |
| Middle | 0.0 | 0.0 |

---

**Episode 1, Step 1:**

- Current state: $s = \text{Start}$
- **Choose action** with ε-greedy: both $Q$ values are 0, so random tie-break → choose **Right**
- Take **Right**, observe $r = -1$, arrive at $s' = \text{Middle}$
- **Choose next action** $a'$ from $s'$: both $Q(M, \cdot) = 0$, random → choose **Right**

**SARSA update for $Q(\text{Start}, \text{Right})$:**

$$Q(S, R) \leftarrow 0 + 0.5\bigl[-1 + 0.9 \times Q(M, R) - 0\bigr] = 0.5 \times (-1 + 0) = \mathbf{-0.5}$$

Updated Q-table: $Q(\text{Start}, \text{Right}) = -0.5$

---

**Episode 1, Step 2:**

- Current state: $s = \text{Middle}$, current action: $a = \text{Right}$ (already chosen as $a'$)
- Take **Right**, observe $r = +10$, arrive at $s' = \text{Goal}$ (terminal)
- At terminal: $Q(\text{Goal}, \cdot) = 0$, next action $a'$ is irrelevant

**SARSA update for $Q(\text{Middle}, \text{Right})$:**

$$Q(M, R) \leftarrow 0 + 0.5\bigl[10 + 0.9 \times 0 - 0\bigr] = 0.5 \times 10 = \mathbf{5.0}$$

**End of Episode 1:**

| State | $Q(\cdot, \text{Right})$ | $Q(\cdot, \text{Stay})$ |
|-------|--------------------------|-------------------------|
| Start | −0.5 | 0.0 |
| Middle | **5.0** | 0.0 |

---

**Episode 2, Step 1:**

- $s = \text{Start}$
- ε-greedy: with probability $1-\varepsilon = 0.9$, choose greedy → $\arg\max_a Q(\text{Start}, a) = \text{Stay}$ (since $Q(S, \text{Stay}) = 0.0 > Q(S, \text{Right}) = -0.5$)
- Actually, let's say the agent exploits and chooses **Stay**
- Take **Stay**, observe $r = -2$, remain at $s' = \text{Start}$
- Choose $a'$ ε-greedy from Start: still greedy → **Stay**

**SARSA update for $Q(\text{Start}, \text{Stay})$:**

$$Q(S, \text{Stay}) \leftarrow 0 + 0.5\bigl[-2 + 0.9 \times Q(S, \text{Stay}) - 0\bigr] = 0.5(-2 + 0) = \mathbf{-1.0}$$

Updated Q-table: $Q(\text{Start}, \text{Stay}) = -1.0$

Now: $Q(\text{Start}, \text{Right}) = -0.5 > Q(\text{Start}, \text{Stay}) = -1.0$

The greedy action at Start switches back to **Right**!

---

**Episode 2, Step 2 onwards** — agent chooses Right, goes to Middle, goes to Goal, collecting $+10$.

After several episodes, the Q-values converge:

| State | $Q(\cdot, \text{Right})$ | $Q(\cdot, \text{Stay})$ | Optimal Action |
|-------|--------------------------|-------------------------|----------------|
| Start | **≈ 7.1** | ≈ −4.5 | Right ✓ |
| Middle | **≈ 9.5** | ≈ −2.0 | Right ✓ |

The agent has discovered the optimal policy: **always go Right** — without ever being told the transition probabilities or reward function.

---

## 8. Key Concepts Unified: The TD Error as a Learning Signal

One of the most elegant aspects of TD learning is that the **TD error** $\delta_t$ appears everywhere in modern RL and even in neuroscience:

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

| $\delta_t$ | Meaning | Agent's response |
|-----------|---------|-----------------|
| $\delta_t > 0$ | Better than expected! | Increase value of $s_t$ (positive surprise) |
| $\delta_t = 0$ | Exactly as expected | No update needed |
| $\delta_t < 0$ | Worse than expected | Decrease value of $s_t$ (negative surprise) |

**Neuroscience Connection:** In 1997, Schultz, Dayan, and Montague discovered that **dopamine neurons** in the brain fire in a pattern almost identical to the TD error signal. When something good happens unexpectedly, dopamine spikes (large positive $\delta$). When something good was expected but didn't happen, dopamine dips (negative $\delta$). RL and the brain use the same algorithm for learning!

---

## 9. Real-World Applications

| Algorithm | Domain | How It's Used |
|-----------|--------|---------------|
| **Monte Carlo** | Game playing (Go, Poker) | Evaluate board positions by simulating many random complete games (rollouts) |
| **Monte Carlo** | Finance (option pricing) | Estimate asset values by simulating thousands of possible market scenarios |
| **TD Learning** | Robot navigation | Update positional value estimates after each sensor reading — no need to complete the route |
| **TD Learning** | Recommendation systems | Update user preference scores after each click, without waiting for the user's entire session |
| **SARSA** | Autonomous driving (cautious) | Learn safe policies — on-policy means the agent considers the consequences of its own cautious behavior |
| **SARSA** | Resource scheduling | Adapt scheduling policies based on actual job completion patterns observed in real-time |

**An important distinction — SARSA vs. Q-Learning (next week):**

SARSA is **on-policy** — it learns the value of the policy it's actually using. This makes it **safer** in risky environments. If you're learning to drive, you want to learn the value of your cautious driving style, not the value of a fearless racer. Q-Learning (next week) learns the **optimal** policy regardless of what the agent actually does — more powerful but potentially more dangerous during training.

---

## 10. Summary

| Concept | Key Idea |
|---------|----------|
| **Model-Free** | Learn from experience alone — no need for $\mathcal{T}$ or $\mathcal{R}$ |
| **Prediction** | Estimate $V^\pi(s)$ or $Q^\pi(s,a)$ for a fixed policy |
| **Control** | Find the optimal policy $\pi^*$ by alternating evaluation and improvement |
| **Monte Carlo** | Average actual returns from complete episodes — unbiased but high variance |
| **TD(0)** | Update after every step using one-step bootstrapping — fast but initially biased |
| **Bootstrapping** | Using your own current estimates to improve your estimates, step by step |
| **TD error** $\delta_t$ | The surprise signal: actual outcome minus expected outcome |
| **SARSA** | On-policy TD control — evaluates the policy being followed using $(s, a, r, s', a')$ tuples |
| **ε-greedy** | Explore randomly with probability $\varepsilon$ to ensure all actions are tried |
| **GPI** | General framework: alternate between policy evaluation and policy improvement |
| **Bias-Variance tradeoff** | MC is unbiased with high variance; TD is biased but low variance |
| **n-step TD** | A spectrum between TD(0) and MC, tunable by the lookahead depth $n$ |

---

## 11. Exercises

### Conceptual Questions

1. **Why can't Dynamic Programming solve the robot vacuum cleaner problem?** What information would the DP algorithm need that the vacuum cleaner doesn't have access to?

2. **Explain the bias-variance tradeoff** in your own words using the restaurant analogy from this week (rating estimation). When would you prefer MC over TD, and vice versa?

3. **The TD error is $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$.** What happens to the TD error when the algorithm has fully converged? What does this tell you about the converged value function?

4. **Why does SARSA need to observe the "next action" $a_{t+1}$** before updating, while Q-Learning (next week) does not? What is the consequence of this design choice?

---

### Calculation Problems

**Problem 1 — Monte Carlo Return Calculation**

An agent follows a policy through one complete episode: $s_0 \to s_1 \to s_2 \to s_T$

Rewards: $r_1 = -2$, $r_2 = +4$, $r_3 = +8$ (terminal)

Discount factor: $\gamma = 0.8$

(a) Calculate $G_0$ (the return from $s_0$).

(b) Calculate $G_1$ (the return from $s_1$).

(c) If the current estimate is $V(s_0) = 3.0$ and $\alpha = 0.4$, what is the updated $V(s_0)$ after this episode?

---

**Problem 2 — TD(0) Update**

A robot is in state $A$. Current estimates: $V(A) = 5$, $V(B) = 8$

The robot takes an action, receives reward $r = +3$, and transitions to state $B$.

Learning rate: $\alpha = 0.3$, discount factor: $\gamma = 0.9$

(a) Calculate the TD target.

(b) Calculate the TD error $\delta$.

(c) Calculate the updated $V(A)$.

---

**Problem 3 — SARSA Q-Table Update**

You have the following Q-table:

| State | Q(·, Left) | Q(·, Right) |
|-------|------------|-------------|
| X | 3.0 | 5.0 |
| Y | 7.0 | 2.0 |

The agent is in state $X$, chooses action **Right**, receives reward $r = -1$, moves to state $Y$, and then chooses action **Left** (from the ε-greedy policy).

Learning rate: $\alpha = 0.5$, discount factor: $\gamma = 0.9$

(a) Write out the SARSA quintuple $(s, a, r, s', a')$ for this transition.

(b) Calculate the SARSA target.

(c) Calculate the updated $Q(X, \text{Right})$.

---

### Answer Key

**Problem 1:**
- (a) $G_0 = -2 + 0.8(4) + 0.8^2(8) = -2 + 3.2 + 5.12 = \mathbf{6.32}$
- (b) $G_1 = 4 + 0.8(8) = 4 + 6.4 = \mathbf{10.4}$
- (c) $V(s_0) \leftarrow 3.0 + 0.4(6.32 - 3.0) = 3.0 + 1.33 = \mathbf{4.33}$

**Problem 2:**
- (a) TD target $= r + \gamma V(B) = 3 + 0.9 \times 8 = 3 + 7.2 = \mathbf{10.2}$
- (b) $\delta = 10.2 - 5 = \mathbf{+5.2}$ (better than expected!)
- (c) $V(A) \leftarrow 5 + 0.3 \times 5.2 = 5 + 1.56 = \mathbf{6.56}$

**Problem 3:**
- (a) $(X,\; \text{Right},\; -1,\; Y,\; \text{Left})$
- (b) SARSA target $= r + \gamma \cdot Q(Y, \text{Left}) = -1 + 0.9 \times 7.0 = -1 + 6.3 = \mathbf{5.3}$
- (c) $Q(X, \text{Right}) \leftarrow 5.0 + 0.5(5.3 - 5.0) = 5.0 + 0.15 = \mathbf{5.15}$

---

*Coming up in Week 4 — Q-Learning: we'll take everything from this week and introduce an off-policy control algorithm that can find the optimal policy even while behaving sub-optimally, giving us the foundation for the most famous algorithm in RL.*
