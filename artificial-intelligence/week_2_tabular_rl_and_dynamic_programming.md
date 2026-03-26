# Week 2 — Tabular RL & Dynamic Programming

---

## 1. The "Perfect World" Assumption

> *A chess engine doesn't learn chess from scratch by trial and error — it knows the rules perfectly and uses them to plan millions of moves ahead. How do we build an agent that does the same?*

### Planning vs. Learning

In Week 1, we framed Reinforcement Learning as learning by trial and error — the agent *discovers* good actions by living through rewards and penalties. But this raises a natural question:

**What if the agent already knows the complete MDP — every state, every action, every transition probability, and every reward?**

When the MDP is fully known, the agent doesn't need to *experience* anything. It can **reason** and **plan** mathematically — computing the exact optimal policy without taking a single real action. This is called **Dynamic Programming (DP)**.

**Intuitive Example — Chess Engine vs. Self-Taught Player:**
- A **chess engine** (like Stockfish) has access to the full rules of chess. It simulates millions of possible future board positions without moving a single piece, and computes the best move — pure planning.
- A **self-taught player** with no rulebook must play thousands of games, win and lose, to gradually improve — pure learning.
- Both can become strong players, but the engine reaches expertise far faster *because it knows the rules*.

**Intuitive Example — GPS Routing:**
- A GPS app knows the complete map: all roads, distances, speed limits, and traffic patterns — the full "MDP."
- It doesn't need to send you down random roads hoping to stumble on the fastest route. It computes the optimal path directly.
- An agent navigating without a map — relying on experience — is like a driver learning routes the hard way.

### When Is the Complete MDP Known?

| Setting | MDP Known? | Approach | Example |
|---------|-----------|----------|---------|
| Game AI (Chess, Go) | ✓ Yes (by design) | Dynamic Programming | Chess engines |
| Inventory Management | ✓ Yes (if demand is modeled) | Policy Iteration | Supply chain optimization |
| Robot Arm (controlled lab) | Partially | DP + estimation | Industrial automation |
| **Robot Vacuum Cleaner** | **✗ No (room layout & obstacles unknown)** | **Model-Free RL** | **Roomba-style coverage** |
| Stock Trading | ✗ No (market is unpredictable) | Model-Free RL | Algorithmic trading |
| Human medical treatment | ✗ No (patient variation) | Model-Free RL | Personalized medicine |

**Why is each case classified this way?**

**✓ Game AI (Chess, Go) — MDP fully known.**
The rules of Chess and Go are written down precisely. Every legal move, every resulting board position, and every win/loss outcome is determined by the rules — there is no hidden randomness. The programmer can enumerate all transitions $\mathcal{T}(s' \mid s, a)$ and rewards $\mathcal{R}(s, a, s')$ exactly. Because the complete model is available before the first game is played, the engine can plan purely by computation — no experience required.

**✓ Inventory Management — MDP known (if demand is modeled).**
A company decides how many units to reorder each day. The states (stock level), actions (order size), step costs (holding fees), and stockout penalties are all defined by the business itself. Customer demand can be estimated from historical sales data and fitted to a probability distribution — giving an explicit $\mathcal{T}$. As long as the demand model is a reasonable approximation of reality, the full MDP can be written down and solved with DP.

**◑ Robot Arm (controlled lab) — Partially known.**
In a controlled factory environment, the joint angles, motor dynamics, and physics equations are largely known from engineering specs. However, small real-world imperfections — subtle friction variations, sensor noise, tool wear — mean the true $\mathcal{T}$ is only approximately known. Practitioners often use the known physics to build a *model* and then fine-tune with real experience to correct the residual errors. This "hybrid" approach sits between pure DP and pure model-free RL.

**✗ Robot Vacuum Cleaner — MDP unknown.**
The vacuum enters a room it has never seen: the floor plan, furniture positions, and obstacle layout are all unknown at the start. Even if the vacuum could build an internal map on the fly, the map will have gaps and errors, so $\mathcal{T}$ is never fully known. The reward is also multi-faceted — it cares about *coverage* (visit every tile), *speed* (drain as little battery as possible), and *collision avoidance* (don't bump into chair legs). A modern vacuum like Roomba therefore uses a model-free learning strategy: start with a random or heuristic exploration policy, receive rewards proportional to newly cleaned area minus battery spent, and iteratively refine behavior across many cleaning runs. Once enough experience accumulates, the vacuum can even build a partial map and switch to a *model-based* approach for that specific room — an example of **transitioning from model-free to model-based RL** as knowledge is acquired.

**✗ Stock Trading — MDP unknown.**
Stock prices depend on millions of interacting factors: investor sentiment, geopolitical events, competitor earnings, and macroeconomic shifts — none of which can be captured in a fixed transition function $\mathcal{T}$. Even if a model were fitted to historical data, the market evolves constantly and yesterday's model is wrong today. Because the environment is fundamentally non-stationary and unpredictable, the agent must learn entirely from live experience without relying on a pre-specified model.

**✗ Human Medical Treatment — MDP unknown.**
Every patient responds differently to treatment due to genetic makeup, lifestyle, and disease progression. There is no fixed $\mathcal{T}(s' \mid s, a)$ that reliably maps a patient's condition and a drug dosage to a future health state. Clinical trials can give population-level estimates, but for any individual patient the true dynamics remain uncertain. The agent (treatment protocol) must learn from patient outcomes over time, making it a model-free problem where the "environment" is the unique biology of each person.

**The take-away:** Dynamic Programming is exact and powerful — but requires complete knowledge of the environment. In Weeks 3–4, we will tackle the real world, where the MDP is unknown and the agent must learn from experience alone.

---

## 2. The Tabular Setting

> *Before spreadsheets were replaced by databases, accountants stored every number explicitly in a table — one row per account, one column per month. Tabular RL does the same for an agent's values.*

### What "Tabular" Means

A problem is called **tabular** when:
1. The **state space** $\mathcal{S}$ is **finite** — there are a countable number of distinct situations.
2. The **action space** $\mathcal{A}$ is **finite** — there are a countable number of possible actions.
3. Values for every state (or every state-action pair) are stored **explicitly in a table**.

This is in contrast to **deep RL** (Weeks 5–7), where the state space is enormous (every pixel arrangement of a video game frame) and a neural network *approximates* values instead.

### The State-Value Function V(s)

The **state-value function** $V(s)$ is a table where each entry answers one question:

> *"Standing in state $s$ and following policy $\pi$ from here onward, how much total discounted reward can I expect?"*

$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \;\Big|\; s_t = s \right]$$

For a 4-state grid world, this table might look like:

| State | $V(s)$ |
|-------|--------|
| A | 8.0 |
| B | 10.0 |
| C | 10.0 |
| D (Goal) | 0.0 |

### The Action-Value Function Q(s, a) — The Q-Table

The **action-value function** $Q(s, a)$ is more granular — it answers:

> *"In state $s$, if I take action $a$ right now (even if $\pi$ wouldn't suggest it), then follow $\pi$ forever after, how much total reward do I expect?"*

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s,\ a_t = a \right]$$

For the same 4-state grid, the Q-table has one row per state and one column per action:

| State | Q(s, Up) | Q(s, Down) | Q(s, Left) | Q(s, Right) |
|-------|----------|------------|------------|-------------|
| A | 6.2 | **8.0** ← best | 6.2 | 8.0 |
| B | 8.0 | **10.0** ← best | 6.2 | 8.0 |
| C | 6.2 | 8.0 | 8.0 | **10.0** ← best |
| D | — | — | — | — |

### The Bridge Between V and Q

$$Q^\pi(s, a) = \sum_{s'} \mathcal{T}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V^\pi(s') \right]$$

$$V^\pi(s) = Q^\pi(s,\ \pi(s)) \quad \text{(for a deterministic policy)}$$

$$V^*(s) = \max_a\, Q^*(s, a) \quad \text{(for the optimal policy)}$$

**Intuitive Example — Ordering at a Restaurant:**
- $V(\text{restaurant})$ = *how satisfied you expect to be overall if you go to this restaurant and order as you normally do.*
- $Q(\text{restaurant},\, \text{"pasta"})$ = *how satisfied you'd be if you specifically ordered the pasta today, then resumed your usual habits at future meals.*
- Your overall satisfaction $V$ equals the satisfaction from your best order — the best $Q$ over all dishes.

---

## 3. Running Example: The Grid World

> *A robot must navigate a small room to reach its charging dock.*

We will use the following example throughout Sections 4–8 so that every algorithm produces directly comparable results.

**Scenario:** A robot moves through a 2×2 grid to reach the charging dock at position D. Moving into a wall costs 1 point but keeps the robot in place. Reaching the goal earns +10.

```
┌──────────┬──────────┐
│    A     │    B     │
│  (Start) │          │
├──────────┼──────────┤
│    C     │    D     │
│          │  (Goal)  │
└──────────┴──────────┘
```

**MDP Components:**

| Component | Symbol | Value |
|-----------|--------|-------|
| State space | $\mathcal{S}$ | $\{A,\ B,\ C,\ D\}$ |
| Action space | $\mathcal{A}$ | $\{\uparrow\text{Up},\ \downarrow\text{Down},\ \leftarrow\text{Left},\ \rightarrow\text{Right}\}$ |
| Discount factor | $\gamma$ | $0.9$ |
| Goal reward | $\mathcal{R}(\cdot, \cdot, D)$ | $+10$ |
| Step / wall cost | all other transitions | $-1$ |
| Terminal state | $D$ | Episode ends upon arrival |

**Transition Table (deterministic — no stochastic slipping this time):**

| From | Action | Moves To | Reward |
|------|--------|----------|--------|
| A | Up | A (wall) | −1 |
| A | Down | C | −1 |
| A | Left | A (wall) | −1 |
| A | Right | B | −1 |
| B | Up | B (wall) | −1 |
| B | Down | D ← Goal! | **+10** |
| B | Left | A | −1 |
| B | Right | B (wall) | −1 |
| C | Up | A | −1 |
| C | Down | C (wall) | −1 |
| C | Left | C (wall) | −1 |
| C | Right | D ← Goal! | **+10** |
| D | — | terminal | — |

> **Why do wall actions still cost −1?** The robot expends energy attempting the move even though it doesn't go anywhere. This discourages the robot from blindly bumping into walls — just as a delivery driver shouldn't repeatedly try to turn onto a closed road.

---

## 4. Policy Evaluation — "How Good Is This Policy?"

> *You can't improve a strategy you haven't measured. Policy Evaluation gives you the yardstick.*

### The Bellman Expectation Equation

To evaluate a **given** policy $\pi$, we use the **Bellman Expectation Equation** — the recursive relationship between a state's value and its successors' values *under $\pi$*:

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} \mathcal{T}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V^\pi(s') \right]$$

For a **deterministic policy** this simplifies to:

$$V^\pi(s) = \sum_{s'} \mathcal{T}(s' \mid s,\ \pi(s)) \left[ \mathcal{R}(s,\ \pi(s),\ s') + \gamma V^\pi(s') \right]$$

**Contrast with Week 1:** The Bellman *Optimality* Equation uses $\max_a$ and yields the best possible values. The Bellman *Expectation* Equation uses the fixed actions prescribed by $\pi$ — it simply *measures* the policy, not improves it.

| Equation | Purpose | Key Operator |
|----------|---------|--------------|
| Bellman Expectation | Measure a specific policy $\pi$ | $\sum_a \pi(a \mid s)$ or $\pi(s)$ (deterministic) |
| Bellman Optimality | Find the optimal policy | $\max_a$ |

### Iterative Policy Evaluation Algorithm

Because $V^\pi(s)$ is defined in terms of other $V^\pi(s')$ values, we solve it **iteratively**: start with all zeros, then repeatedly apply the Bellman Expectation equation until the values stop changing.

```
Algorithm: Iterative Policy Evaluation
────────────────────────────────────────────
Input:  MDP (S, A, T, R, γ), policy π, tolerance θ

Initialize V(s) ← 0  for all s ∈ S

Repeat:
    Δ ← 0
    For each non-terminal s ∈ S:
        v_old ← V(s)
        V(s) ← Σ_{s'} T(s'|s,π(s)) [R(s,π(s),s') + γ · V(s')]
        Δ ← max(Δ, |v_old − V(s)|)
Until Δ < θ

Return V
```

**Intuitive Analogy — Word-of-Mouth in a Town:**
Imagine each state as a person who starts with zero information about how good their situation is. Each round, every person asks their "neighbors" (the states they transition to) how good *their* situations are and updates their own estimate. After enough rounds, everyone's estimate settles to the true value.

### Worked Calculation: Evaluating a Suboptimal Policy

**Given policy $\pi_0$:**

| State | Action | Comment |
|-------|--------|---------|
| A | Right → B | Sends robot toward B |
| B | Up → B (wall!) | Robot bounces off wall repeatedly |
| C | Right → D | Good — one step from goal |
| D | — | terminal |

Notice: $\pi_0$ sends the robot from A into B, where it will **bounce off the top wall forever** under Up. This is deliberately terrible policy that we will fix later.

**Analytical Solution** (since the MDP is small and deterministic):

$$V^{\pi_0}(D) = 0 \quad \text{(terminal)}$$

For state $B$ — $\pi_0$ says Up → wall → stays at $B$, incurring −1 and cycling back:

$$V^{\pi_0}(B) = -1 + \gamma \cdot V^{\pi_0}(B)$$

$$V^{\pi_0}(B) \,(1 - 0.9) = -1 \quad \Longrightarrow \quad \boxed{V^{\pi_0}(B) = -10}$$

> **Why −10?** Hitting the wall forever generates an infinite series of −1 penalties. With $\gamma = 0.9$ this converges to the geometric series: $-1 - 0.9 - 0.81 - \cdots = -1/(1 - 0.9) = -10$. The robot is trapped in a losing loop.

For state $A$ — $\pi_0$ says Right → moves to $B$:

$$V^{\pi_0}(A) = -1 + \gamma \cdot V^{\pi_0}(B) = -1 + 0.9 \times (-10) = \boxed{-10}$$

For state $C$ — $\pi_0$ says Right → moves to $D$ (Goal!):

$$V^{\pi_0}(C) = +10 + \gamma \cdot V^{\pi_0}(D) = +10 + 0 = \boxed{+10}$$

**Value Table for $\pi_0$:**

| State | $V^{\pi_0}(s)$ | Interpretation |
|-------|---------------|----------------|
| A | **−10** | Leads straight into the wall-bouncing trap |
| B | **−10** | Stuck hitting the wall forever |
| C | **+10** | One step from the Goal |
| D | **0** | Terminal; no future reward |

**Iterative verification** (synchronous updates, showing convergence of $V(A)$ and $V(B)$)

| Round | $V(A)$ | $V(B)$ | $V(C)$ | $\max \Delta$ |
|-------|--------|--------|--------|---------------|
| 0 (init) | 0 | 0 | 0 | — |
| 1 | $-1 + 0.9(0) = -1$ | $-1 + 0.9(0) = -1$ | $+10$ | 10 |
| 2 | $-1 + 0.9(-1) = -1.9$ | $-1 + 0.9(-1) = -1.9$ | $+10$ | 0.9 |
| 3 | $-1 + 0.9(-1.9) = -2.71$ | $-1 + 0.9(-1.9) = -2.71$ | $+10$ | 0.81 |
| ⋮ | ⋮ | ⋮ | ⋮ | ⋮ |
| ∞ | **−10** | **−10** | **+10** | **0** ✓ |

Each round adds one more discounted $-1$ to $A$ and $B$, tracing the geometric series toward $-10$.

---

## 5. The Action-Value (Q) Function

> *V tells you how good a state is. Q tells you how good a specific action is — before you commit to it.*

### From V to Q via One-Step Lookahead

Given $V^\pi$, you can compute $Q^\pi$ for any state-action pair by "looking one step ahead":

$$Q^\pi(s, a) = \sum_{s'} \mathcal{T}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V^\pi(s') \right]$$

This answers: *"If I deviate from $\pi$ for exactly one step by taking action $a$, then resume following $\pi$ from wherever I land — what total reward do I expect?"*

**Intuitive Example — Job Offer:**
- $V^\pi(\text{current job})$ = your expected career satisfaction if you stay and follow your usual career decisions.
- $Q^\pi(\text{current job},\ \text{"accept new offer"})$ = your expected career satisfaction if you take this specific new position today, then resume your usual career decisions from there.

$Q$ is the language of **decision-making in the moment**: it answers "should I take this specific action right now?"

### Worked Calculation: Q-Values Under $\pi_0$

Using $V^{\pi_0}$: $V(A)=-10$, $V(B)=-10$, $V(C)=10$, $V(D)=0$

**All Q-values for state B:**

| Action from B | Moves To | Reward | Q formula | Q Value |
|---------------|---------|--------|-----------|---------|
| Up (wall) | B | −1 | $-1 + 0.9 \times (-10)$ | **−10** ← current $\pi_0$ action |
| **Down** | D (Goal!) | +10 | $+10 + 0.9 \times 0$ | **+10** ✓ best |
| Left | A | −1 | $-1 + 0.9 \times (-10)$ | **−10** |
| Right (wall) | B | −1 | $-1 + 0.9 \times (-10)$ | **−10** |

**All Q-values for state A:**

| Action from A | Moves To | Reward | Q formula | Q Value |
|---------------|---------|--------|-----------|---------|
| Up (wall) | A | −1 | $-1 + 0.9 \times (-10)$ | **−10** |
| **Down** | C | −1 | $-1 + 0.9 \times 10$ | **+8** ✓ best |
| Left (wall) | A | −1 | $-1 + 0.9 \times (-10)$ | **−10** |
| Right | B | −1 | $-1 + 0.9 \times (-10)$ | **−10** ← current $\pi_0$ action |

**All Q-values for state C:**

| Action from C | Moves To | Reward | Q formula | Q Value |
|---------------|---------|--------|-----------|---------|
| Up | A | −1 | $-1 + 0.9 \times (-10)$ | **−10** |
| Down (wall) | C | −1 | $-1 + 0.9 \times 10$ | **+8** |
| Left (wall) | C | −1 | $-1 + 0.9 \times 10$ | **+8** |
| **Right** | D (Goal!) | +10 | $+10 + 0.9 \times 0$ | **+10** ✓ best ← current $\pi_0$ action |

**Critical insight:** Even under a terrible policy like $\pi_0$, the Q-table reveals the hidden path to success. *B→Down* and *A→Down* are dramatically superior to what $\pi_0$ recommends — we just need a mechanism to act on this information.

---

## 6. Policy Improvement — "Can We Do Better?"

> *Once you know how good each action is, the path forward is obvious: choose the better ones.*

### Greedy Policy Improvement

Given the Q-values of a policy $\pi$, we construct an improved policy $\pi'$ by choosing the action with the **highest Q-value** at every state:

$$\pi'(s) = \arg\max_a\, Q^\pi(s, a)$$

### The Policy Improvement Theorem

This greedy step is backed by a guarantee:

> **Policy Improvement Theorem:** If $\pi'$ is the greedy policy with respect to $V^\pi$, then $V^{\pi'}(s) \geq V^\pi(s)$ for every state $s$.

**Proof sketch:** At every state, the greedy policy earns at least as much expected return as $\pi$ does, because we directly chose the action whose Q-value is highest — and $Q^\pi(s, \pi'(s)) \geq Q^\pi(s, \pi(s)) = V^\pi(s)$. By unrolling this one-step advantage recursively across all future states, the inequality propagates to a full improvement in $V$.

If $\pi'(s) = \pi(s)$ for all states — i.e., the greedy policy makes no changes — then $V^{\pi'} = V^\pi$ and the Bellman Optimality Equation is satisfied. The policy is **optimal**.

### Worked Calculation: Improving $\pi_0$ → $\pi_1$

| State | $\pi_0$ action (Q value) | Best Q-value found | New $\pi_1$ action | Δ Value |
|-------|--------------------------|--------------------|--------------------|---------|
| A | Right (Q = **−10**) | Down (Q = **+8**) | **Down** | +18 |
| B | Up (Q = **−10**) | Down (Q = **+10**) | **Down** | +20 |
| C | Right (Q = **+10**) | Right (Q = **+10**) | Right (no change) | 0 |

**Improved policy $\pi_1$:** $A \to \text{Down},\; B \to \text{Down},\; C \to \text{Right}$

The Policy Improvement Theorem guarantees immediately — without computing anything else — that $V^{\pi_1}(s) \geq V^{\pi_0}(s)$ for all states. We already know the improvement will be massive.

---

## 7. Policy Iteration — "Alternate Until Optimal"

> *Evaluate your strategy. Improve your strategy. Repeat — until the strategy doesn't change.*

### The Algorithm

**Policy Iteration** weaves evaluation and improvement into a single loop that is provably guaranteed to converge to the optimal policy:

```
Algorithm: Policy Iteration
────────────────────────────────────────────
Input: MDP (S, A, T, R, γ)

1. Initialize:  π(s) ← some arbitrary action  for all s ∈ S

2. Repeat:
    (a) POLICY EVALUATION
        Compute V^π fully using Iterative Policy Evaluation

    (b) POLICY IMPROVEMENT
        policy_stable ← True
        For each s ∈ S:
            old_action ← π(s)
            π(s)       ← argmax_a  Q^π(s, a)      (greedy update)
            If old_action ≠ π(s):
                policy_stable ← False

    (c) If policy_stable:
            return π, V^π                          (OPTIMAL — done!)
```

**Convergence:** Each improvement step either strictly increases $V(s)$ for at least one state, or the policy is already optimal. Since there are at most $|\mathcal{A}|^{|\mathcal{S}|}$ possible deterministic policies (a finite number), the loop must terminate.

**Intuitive Example — Restaurant Owner Refining a Menu:**
1. **Evaluate:** Survey diners' satisfaction with the current menu → compute $V^\pi$ per dish.
2. **Improve:** For each dish, compare Q-values: "What if I swapped this out for something else?" — replace every dish with a strictly better alternative wherever one exists.
3. **Repeat:** The menu can only improve. Stop when no swap makes diners happier.

### Worked Calculation: Full Policy Iteration on the Grid World

---

#### Iteration 1

**(a) Policy Evaluation of $\pi_0 = \{A \to R,\; B \to U,\; C \to R\}$:**

From Section 4:

$$V^{\pi_0}(A) = -10, \quad V^{\pi_0}(B) = -10, \quad V^{\pi_0}(C) = +10, \quad V^{\pi_0}(D) = 0$$

**(b) Policy Improvement** — compute Q-values using $V^{\pi_0}$ and take the greedy action:

| State | Current $\pi_0$ | Q(Up) | Q(Down) | Q(Left) | Q(Right) | Best action | $\pi_1$ |
|-------|-----------------|-------|---------|---------|---------|-------------|---------|
| A | Right | −10 | **+8** | −10 | −10 | **Down** | Down |
| B | Up | −10 | **+10** | −10 | −10 | **Down** | Down |
| C | Right | −10 | +8 | +8 | **+10** | Right | Right |

> Q calculations for reference (using $V^{\pi_0}$):
> - $Q(A,\text{Down}) = -1 + 0.9 \times V(C) = -1 + 9 = 8$
> - $Q(B,\text{Down}) = +10 + 0.9 \times V(D) = +10$
> - $Q(C,\text{Up}) = -1 + 0.9 \times V(A) = -1 + (-9) = -10$
> - $Q(C,\text{Down/Left}) = -1 + 0.9 \times V(C) = -1 + 9 = 8$ (wall — stays at C)

$\pi_1 = \{A \to \text{Down},\; B \to \text{Down},\; C \to \text{Right}\}$

Policy changed → continue.

---

#### Iteration 2

**(a) Policy Evaluation of $\pi_1 = \{A \to \text{Down},\; B \to \text{Down},\; C \to \text{Right}\}$:**

$$V^{\pi_1}(D) = 0$$

$$V^{\pi_1}(C) = +10 + 0.9 \times V^{\pi_1}(D) = +10$$

$$V^{\pi_1}(B) = +10 + 0.9 \times V^{\pi_1}(D) = +10$$

$$V^{\pi_1}(A) = -1 + 0.9 \times V^{\pi_1}(C) = -1 + 9 = +8$$

| State | $V^{\pi_1}(s)$ |
|-------|---------------|
| A | **8** |
| B | **10** |
| C | **10** |
| D | 0 |

**(b) Policy Improvement** — compute Q-values using $V^{\pi_1}$:

| State | Current $\pi_1$ | Q(Up) | Q(Down) | Q(Left) | Q(Right) | Best action | $\pi_2$ |
|-------|-----------------|-------|---------|---------|---------|-------------|---------|
| A | Down | $-1+0.9(8)=6.2$ | $\mathbf{-1+0.9(10)=8}$ | $-1+0.9(8)=6.2$ | $-1+0.9(10)=8$ | Down (tied) | **Down** |
| B | Down | $-1+0.9(10)=8$ | $\mathbf{+10+0=10}$ | $-1+0.9(8)=6.2$ | $-1+0.9(10)=8$ | **Down** | Down |
| C | Right | $-1+0.9(8)=6.2$ | $-1+0.9(10)=8$ | $-1+0.9(10)=8$ | $\mathbf{+10+0=10}$ | **Right** | Right |

No actions changed → **policy_stable = True → Policy Iteration converges!** ✓

**Optimal Policy $\pi^*$:**

| State | Action | Path to Goal | Steps |
|-------|--------|--------------|-------|
| A | Down → C → D | A → C → D | 2 |
| B | Down → D | B → D | 1 |
| C | Right → D | C → D | 1 |
| D | terminal | — | 0 |

**Optimal Values:** $V^*(A) = 8$, $V^*(B) = 10$, $V^*(C) = 10$, $V^*(D) = 0$

---

**Progress summary across Policy Iterations:**

| Iteration | $V(A)$ | $V(B)$ | $V(C)$ | Policy |
|-----------|--------|--------|--------|--------|
| $\pi_0$ (init) | −10 | −10 | +10 | A→R, B→U, C→R |
| $\pi_1$ | +8 | +10 | +10 | A→D, B→D, C→R ← big improvement |
| $\pi_2$ | +8 | +10 | +10 | A→D, B→D, C→R ← no change ✓ |

---

## 8. Value Iteration — "Compress Evaluation into One Step"

> *Why evaluate a policy fully before improving it, when you can fuse both steps into a single operation?*

### The Idea

In Policy Iteration, the **evaluation phase** requires its own convergence loop before we can improve. Value Iteration skips the notion of maintaining an explicit policy altogether. Instead, it directly applies the **Bellman Optimality Equation** as a repeated update rule:

$$V_{k+1}(s) = \max_a \sum_{s'} \mathcal{T}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma\, V_k(s') \right]$$

After convergence, the greedy policy with respect to $V^*$ is extracted in one final step:

$$\pi^*(s) = \arg\max_a \sum_{s'} \mathcal{T}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma\, V^*(s') \right]$$

```
Algorithm: Value Iteration
────────────────────────────────────────────
Input: MDP (S, A, T, R, γ), tolerance θ

Initialize V(s) ← 0  for all s ∈ S

Repeat:
    Δ ← 0
    For each non-terminal s ∈ S:
        v_old ← V(s)
        V(s)  ← max_a  Σ_{s'} T(s'|s,a) [R(s,a,s') + γ · V(s')]
        Δ ← max(Δ, |v_old − V(s)|)
Until Δ < θ

Extract: π*(s) ← argmax_a  Σ_{s'} T(s'|s,a) [R(s,a,s') + γ · V(s')]
Return V*, π*
```

**Key difference from Policy Evaluation:** The $\max_a$ replaces the fixed policy action. Each sweep simultaneously evaluates *and* improves in one pass.

### Worked Calculation: Value Iteration on the Grid World

**Initialize:** $V_0(A) = V_0(B) = V_0(C) = V_0(D) = 0$

---

**Iteration 1** — Apply Bellman Optimality using $V_0 = 0$ everywhere:

$$V_1(D) = 0 \quad \text{(terminal)}$$

$$V_1(C) = \max \begin{cases} Q(C,\uparrow)= -1 + 0.9 \times V_0(A) = -1 \\ Q(C,\downarrow)= -1 + 0.9 \times V_0(C) = -1 \quad \text{(wall)} \\ Q(C,\leftarrow)= -1 + 0.9 \times V_0(C) = -1 \quad \text{(wall)} \\ Q(C,\rightarrow)= +10 + 0.9 \times V_0(D) = \mathbf{+10} \end{cases} = \mathbf{10}$$

$$V_1(B) = \max \begin{cases} Q(B,\uparrow)= -1 + 0.9 \times V_0(B) = -1 \quad \text{(wall)} \\ Q(B,\downarrow)= +10 + 0.9 \times V_0(D) = \mathbf{+10} \\ Q(B,\leftarrow)= -1 + 0.9 \times V_0(A) = -1 \\ Q(B,\rightarrow)= -1 + 0.9 \times V_0(B) = -1 \quad \text{(wall)} \end{cases} = \mathbf{10}$$

$$V_1(A) = \max \begin{cases} Q(A,\uparrow)= -1 + 0.9 \times V_0(A) = -1 \quad \text{(wall)} \\ Q(A,\downarrow)= -1 + 0.9 \times V_0(C) = -1 \\ Q(A,\leftarrow)= -1 + 0.9 \times V_0(A) = -1 \quad \text{(wall)} \\ Q(A,\rightarrow)= -1 + 0.9 \times V_0(B) = -1 \end{cases} = \mathbf{-1}$$

> All actions from A tie at −1 after Iteration 1 — the agent can't "see" the goal yet because it is two hops away and all $V_0$ values are zero. The goal's reward will propagate to A in the next iteration.

---

**Iteration 2** — Use $V_1$: $V_1(A)=-1$, $V_1(B)=10$, $V_1(C)=10$, $V_1(D)=0$

$$V_2(C) = \max \begin{cases} -1 + 0.9 \times V_1(A) = -1 + 0.9(-1) = -1.9 \\ -1 + 0.9 \times V_1(C) = -1 + 9 = 8 \quad \text{(wall)} \\ -1 + 0.9 \times V_1(C) = 8 \quad \text{(wall)} \\ +10 + 0.9 \times V_1(D) = \mathbf{10} \end{cases} = \mathbf{10}$$

$$V_2(B) = \max \begin{cases} -1 + 0.9 \times V_1(B) = -1 + 9 = 8 \quad \text{(wall)} \\ +10 + 0.9 \times V_1(D) = \mathbf{10} \\ -1 + 0.9 \times V_1(A) = -1 + (-0.9) = -1.9 \\ -1 + 0.9 \times V_1(B) = 8 \quad \text{(wall)} \end{cases} = \mathbf{10}$$

$$V_2(A) = \max \begin{cases} -1 + 0.9 \times V_1(A) = -1.9 \quad \text{(wall)} \\ -1 + 0.9 \times V_1(C) = -1 + 9 = \mathbf{8} \\ -1 + 0.9 \times V_1(A) = -1.9 \quad \text{(wall)} \\ -1 + 0.9 \times V_1(B) = -1 + 9 = \mathbf{8} \end{cases} = \mathbf{8}$$

The goal's value has now propagated two hops away to reach state A!

---

**Iteration 3** — Check for convergence using $V_2$: $V_2(A)=8$, $V_2(B)=10$, $V_2(C)=10$, $V_2(D)=0$

$$V_3(A) = \max \begin{cases} -1.9 \quad (\text{wall}) \\ -1 + 0.9 \times V_2(C) = -1 + 9 = 8 \\ -1.9 \quad (\text{wall}) \\ -1 + 0.9 \times V_2(B) = -1 + 9 = 8 \end{cases} = 8 \quad \text{(no change)}$$

$$\max \Delta = 0 < \theta \quad \Longrightarrow \quad \textbf{Converged!}\ ✓$$

**Convergence Summary:**

| Iter | $V(A)$ | $V(B)$ | $V(C)$ | $V(D)$ | $\max \Delta$ |
|------|--------|--------|--------|--------|---------------|
| 0 | 0 | 0 | 0 | 0 | — |
| 1 | **−1** | **10** | **10** | 0 | 10 |
| 2 | **8** | 10 | 10 | 0 | 9 |
| 3 | 8 | 10 | 10 | 0 | **0** ✓ |

**Extract Optimal Policy** — greedy with respect to $V^*$:

$$\pi^*(A) = \arg\max_a Q(A, a) = \text{Down or Right} \quad (Q = 8 \text{ for both})$$

$$\pi^*(B) = \text{Down} \quad (Q = 10)$$

$$\pi^*(C) = \text{Right} \quad (Q = 10)$$

This matches the result from Policy Iteration exactly — both algorithms find the same optimal policy and the same optimal values, confirming correctness. ✓

> **Why does Value Iteration converge in only 3 iterations here?** The grid world has a "diameter" of 2 — any state can reach the goal in at most 2 steps. After 2 Bellman sweeps, the goal's reward has propagated to every reachable state, and no further updates are needed.

---

## 9. Policy Iteration vs. Value Iteration

> *Two roads to the same destination — which one should you take?*

### Side-by-Side Comparison

| Feature | Policy Iteration | Value Iteration |
|---------|----------------|-----------------|
| **Core loop** | Evaluate policy fully, then improve | Apply Bellman Optimality directly |
| **Policy representation** | Explicitly maintained throughout | Implicit; extracted only at the end |
| **Per-iteration cost** | High (full evaluation loop inside) | Low (single Bellman sweep) |
| **Outer iterations to converge** | Fewer (policy changes are decisive) | More (values trickle in gradually) |
| **Useful policy available mid-run** | ✓ Yes — always has a valid policy | ✗ No — must wait until $V$ converges |
| **Convergence guarantee** | ✓ Yes | ✓ Yes |
| **Finds optimal policy** | ✓ Yes | ✓ Yes |

**Analogy — Preparing for an Exam:**
- **Policy Iteration:** Study one subject completely → take a practice test → identify weaknesses → revise your study plan → repeat. Each cycle is intensive and informed, but you always know your current strategy.
- **Value Iteration:** Run rapid flashcard drills across all topics → scan for your weakest areas after each round → stop once everything is solid. Each round is quick, but you don't have a "study plan" until the very end.

### Rule of Thumb

| Situation | Preferred Algorithm |
|-----------|---------------------|
| States are few and evaluation is fast | Policy Iteration |
| Values propagate slowly across many states | Value Iteration |
| A deployable policy is needed before full convergence | Policy Iteration |
| Simplicity of implementation is the priority | Value Iteration |

In practice, both are equivalent in correctness and serve primarily as conceptual foundations. Real-world problems are almost never small enough for tabular DP to be applied directly — they motivate everything that follows in this course.

---

## 10. Limitations: The Curse of Dimensionality

> *Everything works perfectly on our 4-state grid world — until the state space explodes.*

### The Core Problem

Dynamic Programming is exact and guaranteed to find the optimal policy. But it requires iterating over **every single state** at every sweep. When the state space is enormous, this becomes computationally impossible.

**Intuitive Example — Backgammon:**
- Backgammon has approximately $10^{20}$ possible board positions.
- A tabular $V$-table would need $10^{20}$ entries. At 8 bytes per float, that's $8 \times 10^{20}$ bytes ≈ **800 million terabytes** of memory.

**Intuitive Example — Robot Arm:**
- A robot arm with 6 joints, each with 1000 discrete angular positions, has $1000^6 = 10^{18}$ states.
- Add joint velocities and the state space becomes effectively continuous — infinite.

### The Exponential Explosion

If each feature of your state takes $k$ discrete values and your state is described by $n$ features, the state space has $k^n$ entries:

| Environment | State Features | Values/Feature | State Space Size | Tabular? |
|-------------|---------------|----------------|-----------------|---------|
| Grid World (2×2) | 1 (position) | 4 | **4** | ✓ Easy |
| Grid World (10×10) | 1 (position) | 100 | **100** | ✓ Easy |
| Grid World (100×100) | 1 (position) | 10 000 | **10 000** | ✓ Feasible |
| Robot arm (6 joints) | 6 | 100 angle bins/joint | **$10^{12}$** | ✗ Too large |
| Atari game (pixels) | 84×84 pixels | 256 grayscale | **$256^{7056}$** | ✗ Impossible |

### What Comes Next?

The solution is **function approximation** — replacing the explicit value table with a **parametric model** (typically a neural network) that takes a state as input and outputs an approximate value. Instead of storing $10^{18}$ exact numbers, we store a few million neural network weights that generalize across similar states.

| Approach | State Space | Storage | Accuracy |
|----------|------------|---------|----------|
| Tabular DP (this week) | Small, discrete | Exact table | Exact |
| Linear function approximation | Medium | Weight vector | Approximate |
| Deep Neural Network (DQN, Week 5) | Large / continuous | Network parameters | Approximate |

Understanding tabular DP deeply — as you have this week — is not just an exercise in theory. Every approximate method (DQN, PPO, etc.) is fundamentally trying to solve the same Bellman equations, just without being able to store every state explicitly. The intuition you've built here is the foundation for everything in the second half of this course.

---

## 11. Summary

| Concept | Key Idea |
|---------|----------|
| **Dynamic Programming** | Plan optimally when the full MDP is known — no experience needed |
| **Tabular Setting** | Finite states/actions; values stored explicitly in tables |
| **State-Value Function $V(s)$** | Expected total discounted return from $s$ following policy $\pi$ |
| **Action-Value Function $Q(s,a)$** | Expected return after taking action $a$ in $s$, then following $\pi$ |
| **Bellman Expectation Equation** | Recursive definition of $V^\pi$; used to *measure* a given policy |
| **Bellman Optimality Equation** | Recursive definition of $V^*$; uses $\max_a$ to find the best policy |
| **Policy Evaluation** | Compute $V^\pi$ for a fixed policy by iterating Bellman Expectation |
| **Policy Improvement Theorem** | The greedy policy is guaranteed to be at least as good as the original |
| **Policy Iteration** | Repeatedly evaluate and improve until the policy stabilizes → optimal |
| **Value Iteration** | Apply Bellman Optimality directly each sweep; extract policy at convergence |
| **Curse of Dimensionality** | State space grows exponentially with features; tabular DP breaks at scale |

---

## 12. Exercise

Test your understanding of Tabular RL and Dynamic Programming from this week.

**WARNING**
- Your test will be ended automatically once you click on the **reveal** button
- The test uses only your 1st time taking score. Later scores of the same student will not be used.

[Open Exercise — Coming Soon]
