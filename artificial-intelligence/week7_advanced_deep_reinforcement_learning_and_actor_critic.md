# Week 7 — Advanced Deep RL & Actor-Critic

---

## 1. The Grand Synthesis: Why We Need Both a Head and a Heart

> *A chess grandmaster doesn't just follow instinct (pure policy) or compute every possible move (pure value). They use trained intuition to decide which moves deserve deep calculation — and deep calculation to sharpen that intuition. Actor-Critic works the same way: one network acts, the other evaluates, and both improve together.*

### 1.1 Picking Up the Threads

Over the past three weeks, we have built two powerful but incomplete families of algorithms:

**Value-Based Methods (DQN, Q-Learning — Weeks 4–5):**
- Learn the Q-function and derive a policy as $\arg\max_a Q(s,a)$
- Very sample efficient (experience replay, off-policy)
- ✗ Struggle with continuous actions; produce deterministic policies only

**Policy Gradient Methods (REINFORCE — Week 6):**
- Learn the policy directly as $\pi(a \mid s;\, \boldsymbol{\theta})$
- Handle continuous actions natively; naturally stochastic
- ✗ High variance — require enormous numbers of complete episodes to converge

The weaknesses of each family are precisely the strengths of the other. The question is: can we combine them?

**Actor-Critic** is that combination. It simultaneously maintains:

| Component | Network | Role | Fixes |
|-----------|---------|------|-------|
| **Actor** | Policy network $\pi(a \mid s;\, \boldsymbol{\theta})$ | Decides what to do | Handles continuous actions, stochastic policies |
| **Critic** | Value network $V(s;\, \boldsymbol{w})$ | Evaluates how good each situation is | Provides low-variance advantage estimates, replacing noisy MC returns |

The critic does not control actions — it exists solely to give the actor a more informative, lower-variance learning signal. The actor does not evaluate states — it exists solely to produce good actions.

### 1.2 A Motivating Analogy: The Startup Team

Imagine a startup where:
- The **CEO (Actor)** makes all the decisions — which markets to enter, which products to build. They act, they commit.
- The **CFO (Critic)** evaluates the company's financial position — not to make product decisions, but to tell the CEO "that quarter went 20% above our forecast" or "this division is underperforming by $3M."

The CEO uses the CFO's analysis to refine future decisions. The CFO uses the CEO's actions to update the financial model. Both improve together, each playing to their strengths.

---

## 2. The Critic: From Monte Carlo Returns to TD Advantage

> *The key innovation of Actor-Critic over REINFORCE: instead of waiting for a complete episode to compute a noisy return $G_t$, the critic provides an immediate, low-variance advantage estimate after every single step.*

### 2.1 Recall: REINFORCE's Variance Problem

In Week 6, REINFORCE updated the policy using the full Monte Carlo return:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot G_t \cdot \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta})$$

And we showed that adding a baseline $b(s_t)$ helps:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot (G_t - b(s_t)) \cdot \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta})$$

The optimal baseline is $V^\pi(s_t)$ — the value function — giving the advantage $A_t = G_t - V^\pi(s_t)$.

The problem: $G_t$ requires the **complete episode** to compute. And even with the baseline, a single $G_t$ is still a random variable — high variance across episodes.

### 2.2 The TD Advantage Estimate

The critic's key contribution: **replace the Monte Carlo return $G_t$ with a one-step TD estimate**:

$$\hat{A}_t = r_{t+1} + \gamma V(s_{t+1};\, \boldsymbol{w}) - V(s_t;\, \boldsymbol{w})$$

This is the **TD error** $\delta_t$ — the same quantity from Week 3! It is simultaneously:
- A signal for updating the **critic** (reduce the TD error — make $V$ more accurate)
- A signal for updating the **actor** (scale the policy gradient — reinforce actions that produced positive surprises)

Breaking it down:

| Term | Meaning |
|------|---------|
| $r_{t+1}$ | Actual reward just received — one step of ground truth |
| $\gamma V(s_{t+1};\, \boldsymbol{w})$ | Critic's estimate of future value from the next state |
| $V(s_t;\, \boldsymbol{w})$ | Critic's current estimate of how good the current state was |
| $\hat{A}_t = \delta_t$ | **Advantage estimate**: how much better/worse than expected was this step? |

**Why is this better than the MC return $G_t$?**

- $G_t$ depends on the entire remaining episode — every random action, every stochastic transition, every reward for potentially hundreds of steps. Enormous variance.
- $\hat{A}_t$ depends on just **one reward $r_{t+1}$** plus the critic's smooth estimate $V(s_{t+1})$. Much lower variance.
- We can update **after every step**, without waiting for the episode to end.

**The tradeoff:** We introduced some bias — $V(s_{t+1};\, \boldsymbol{w})$ is an imperfect estimate (the critic isn't perfect). But the bias-variance tradeoff strongly favors this in practice, especially early in training.

**Intuition — The Sports Coach:**

In REINFORCE, a basketball coach gives feedback only after the game ends: "You played terribly today" or "Great game!" — but the player doesn't know which specific plays were good or bad.

In Actor-Critic, the coach gives feedback after every play: "That pass was better than your average — it opened up a scoring lane" or "That shot was below expectations given how open you were." The player learns much faster with this real-time, comparative feedback.

### 2.3 The n-Step Return: A Spectrum

Just as with TD vs. Monte Carlo in Week 3, there is a spectrum between pure TD (1-step) and pure MC (full episode):

$$G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n};\, \boldsymbol{w})$$

$$\hat{A}_t^{(n)} = G_t^{(n)} - V(s_t;\, \boldsymbol{w})$$

| $n$ | Name | Bias | Variance | Notes |
|-----|------|------|----------|-------|
| 1 | TD advantage | Higher | Lowest | Standard Actor-Critic |
| 2–5 | n-step advantage | Medium | Medium | Good balance |
| $\infty$ | MC advantage | Lowest | Highest | Pure REINFORCE with baseline |

### 2.4 Generalized Advantage Estimation (GAE)

Rather than choosing a single $n$, **GAE** (Schulman et al., 2015) elegantly combines all n-step estimates using an exponential weighting parameter $\lambda \in [0, 1]$:

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}$$

where $\delta_{t+k} = r_{t+k+1} + \gamma V(s_{t+k+1}) - V(s_{t+k})$ is the TD error at step $t+k$.

| $\lambda$ | Behavior | Reduces to |
|-----------|----------|-----------|
| $\lambda = 0$ | Only the immediate TD error | 1-step TD advantage |
| $\lambda = 1$ | Full sum of all future TD errors | Monte Carlo advantage |
| $\lambda = 0.95$ | Mostly long-horizon, slight bias reduction | **PPO's default setting** |

**Intuition:** GAE is like a weighted opinion poll. You trust your immediate observation most (weight 1), discount the next step slightly ($\gamma\lambda$), discount two steps ahead more ($(\gamma\lambda)^2$), and so on — collecting information from the future but trusting closer signals more.

---

## 3. The Actor-Critic Update Rules

### 3.1 Two Networks, Two Losses

Actor-Critic maintains two separate neural networks trained simultaneously:

**Critic Loss** — minimize the error in value prediction (standard supervised regression):

$$\mathcal{L}_{\text{critic}}(\boldsymbol{w}) = \mathbb{E}_t \left[ \left( r_{t+1} + \gamma V(s_{t+1};\, \boldsymbol{w}) - V(s_t;\, \boldsymbol{w}) \right)^2 \right] = \mathbb{E}_t \left[ \delta_t^2 \right]$$

The critic is trained to make $V(s;\, \boldsymbol{w})$ as accurate as possible — reducing the TD error.

**Actor Loss** — maximize expected return using the critic's advantage estimate (gradient ascent):

$$\mathcal{L}_{\text{actor}}(\boldsymbol{\theta}) = -\mathbb{E}_t \left[ \hat{A}_t \cdot \log \pi(a_t \mid s_t;\, \boldsymbol{\theta}) \right]$$

The negative sign converts gradient ascent on $J(\boldsymbol{\theta})$ to gradient descent on the loss (standard in deep learning frameworks).

**Entropy Bonus** — an optional but important regularization term added to the actor loss:

$$\mathcal{L}_{\text{entropy}}(\boldsymbol{\theta}) = -\mathbb{E}_t \left[ H(\pi(\cdot \mid s_t;\, \boldsymbol{\theta})) \right] = \mathbb{E}_t \left[ \sum_a \pi(a \mid s_t) \log \pi(a \mid s_t) \right]$$

Adding $-\beta H(\pi)$ to the actor loss **encourages exploration** by penalizing policies that become too deterministic too fast. Without it, the policy might prematurely commit to suboptimal actions before the critic has learned good value estimates.

**Combined Loss:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{actor}} + c_1 \mathcal{L}_{\text{critic}} - c_2 \mathcal{L}_{\text{entropy}}$$

Where $c_1$ and $c_2$ are weighting coefficients ($c_1 \approx 0.5$, $c_2 \approx 0.01$ in practice).

### 3.2 The Advantage Actor-Critic (A2C) Algorithm

A2C is the synchronous, simplified version of the Actor-Critic framework:

```
╔══════════════════════════════════════════════════════════════════════╗
║               Advantage Actor-Critic (A2C)                          ║
╠══════════════════════════════════════════════════════════════════════╣
║ Initialize actor π(a|s; θ) and critic V(s; w) with random weights   ║
║                                                                      ║
║ For each training iteration:                                         ║
║                                                                      ║
║   ① COLLECT experience by running n parallel environments            ║
║     for T steps each, following current policy π(·|·; θ)            ║
║     → Collect: {sₜ, aₜ, rₜ₊₁, sₜ₊₁} for all steps                ║
║                                                                      ║
║   ② COMPUTE advantages:                                              ║
║     Âₜ = rₜ₊₁ + γ V(sₜ₊₁; w) − V(sₜ; w)    [TD advantage]        ║
║     (or use n-step returns or GAE)                                   ║
║                                                                      ║
║   ③ COMPUTE LOSSES:                                                  ║
║     L_actor  = −mean[ Âₜ · log π(aₜ|sₜ; θ) ]                     ║
║     L_critic = mean[ (rₜ₊₁ + γV(sₜ₊₁;w) − V(sₜ;w))² ]           ║
║     L_entropy = −β · mean[ H(π(·|sₜ; θ)) ]                         ║
║     L_total  = L_actor + c₁ · L_critic + L_entropy                 ║
║                                                                      ║
║   ④ UPDATE both networks simultaneously:                             ║
║     θ ← θ − α · ∇_θ L_actor                                        ║
║     w ← w − α · ∇_w L_critic                                       ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 4. Worked Calculation: A2C One Update Step

**Setup:** A robot in a maze. State is described by 2 features; 2 actions: `Forward` and `Turn`.

**Current networks:**
- Critic: $V(s;\, \boldsymbol{w})$ — currently outputs $V(s_t) = 3.5$ for the current state
- Actor: $\pi(a \mid s;\, \boldsymbol{\theta})$ — currently outputs $\pi(\text{Forward} \mid s_t) = 0.7$, $\pi(\text{Turn} \mid s_t) = 0.3$

**One step of experience:**

The agent is at state $s_t$, takes action `Forward` ($a_t = \text{Forward}$), receives reward $r_{t+1} = +2$, arrives at $s_{t+1}$.

Critic evaluates the next state: $V(s_{t+1};\, \boldsymbol{w}) = 5.0$

**Parameters:** $\gamma = 0.9$, $\alpha_{\text{actor}} = 0.01$, $\alpha_{\text{critic}} = 0.05$

---

### Step 1 — Compute the TD Advantage

$$\hat{A}_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t) = 2 + 0.9 \times 5.0 - 3.5 = 2 + 4.5 - 3.5 = \mathbf{+3.0}$$

**Interpretation:** The outcome was $+3.0$ better than the critic expected. Taking action `Forward` was a positive surprise — the robot moved to a much better state than it was in.

---

### Step 2 — Compute the Critic Loss and Update

$$\mathcal{L}_{\text{critic}} = \hat{A}_t^2 = (3.0)^2 = 9.0$$

Gradient descent on the critic: $V(s_t)$ should increase toward the better estimate:

The critic's update target for $s_t$ is: $r_{t+1} + \gamma V(s_{t+1}) = 2 + 4.5 = 6.5$

$$V(s_t;\, \boldsymbol{w}) \leftarrow V(s_t) + \alpha_{\text{critic}} \times \hat{A}_t = 3.5 + 0.05 \times 3.0 = 3.5 + 0.15 = \mathbf{3.65}$$

The critic's estimate for the current state rises from 3.5 to 3.65 — learning that $s_t$ is slightly more valuable than previously thought, because `Forward` from here leads to a great next state.

---

### Step 3 — Compute the Score Function (Actor Gradient Direction)

For a softmax policy, the score function for the taken action `Forward`:

$$\frac{\partial \log \pi(\text{Forward} \mid s_t)}{\partial \theta_{\text{Forward}}} = 1 - \pi(\text{Forward} \mid s_t) = 1 - 0.7 = +0.3$$

$$\frac{\partial \log \pi(\text{Forward} \mid s_t)}{\partial \theta_{\text{Turn}}} = -\pi(\text{Turn} \mid s_t) = -0.3$$

---

### Step 4 — Compute the Actor Update

$$\Delta\theta_{\text{Forward}} = \alpha_{\text{actor}} \times \hat{A}_t \times \frac{\partial \log\pi}{\partial \theta_{\text{Forward}}} = 0.01 \times 3.0 \times 0.3 = \mathbf{+0.009}$$

$$\Delta\theta_{\text{Turn}} = \alpha_{\text{actor}} \times \hat{A}_t \times \frac{\partial \log\pi}{\partial \theta_{\text{Turn}}} = 0.01 \times 3.0 \times (-0.3) = \mathbf{-0.009}$$

**Result:** The score for `Forward` increases slightly; the score for `Turn` decreases slightly. The policy shifts toward choosing `Forward` more often — because the critic confirmed this was a better-than-expected outcome.

---

### Step 5 — The Feedback Loop

| Network | Before | After | Direction |
|---------|--------|-------|-----------|
| Critic: $V(s_t)$ | 3.5 | 3.65 | ↑ Learned $s_t$ is more valuable |
| Actor: $\pi(\text{Forward})$ | 70% | ~70.9% | ↑ `Forward` reinforced |
| Actor: $\pi(\text{Turn})$ | 30% | ~29.1% | ↓ `Turn` slightly discouraged |

One step. Two networks. Both improved. No complete episode required.

---

## 5. The Dangerous Step: When Policy Updates Go Wrong

> *Imagine you're teaching a dog new tricks. If your feedback is too harsh and too frequent, the dog gets confused and scared — it forgets everything it knew. Overly large policy gradient steps cause the same catastrophic forgetting in neural network policies.*

### 5.1 The "Cliff Edge" of Policy Optimization

Standard policy gradient (REINFORCE and A2C) uses an update of the form:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \hat{A}_t \cdot \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta})$$

The problem: if the advantage $\hat{A}_t$ is large or the gradient is steep, the update step can be **enormous**. The policy can change so drastically that it falls off a "performance cliff":

```
Policy performance J(θ)
      │
  ████│                  ← We're here, doing well
  ███ │
  ██  │        ← One catastrophically large gradient step
  █   │
  ▄   │                          ← We're now here — collapsed!
──────┼──────────────────────────────── θ (policy parameters)
      │
```

After a catastrophically large update, the new policy might assign near-zero probability to actions that used to work well. The experience collected under the old policy is now useless for bootstrapping further learning. Recovery from such a collapse can take thousands of episodes.

**Why is this worse in RL than supervised learning?**

In supervised learning, the training data is fixed — a bad gradient step doesn't change the dataset. In RL, the data is generated by the policy itself. If the policy collapses, it starts generating catastrophically bad experience, which makes the next gradient step even worse. It can spiral.

### 5.2 Trust Region Policy Optimization (TRPO)

**TRPO** (Schulman et al., 2015) formalizes the safety constraint: only update the policy within a **trust region** — a region of parameter space where we can trust our approximation of $J(\boldsymbol{\theta})$.

The constraint is expressed as a bound on the **KL divergence** between old and new policies:

$$\text{maximize}_{\boldsymbol{\theta}} \quad \hat{\mathbb{E}}_t \left[ \frac{\pi(a_t \mid s_t;\, \boldsymbol{\theta})}{\pi(a_t \mid s_t;\, \boldsymbol{\theta}_{\text{old}})} \hat{A}_t \right] \quad \text{subject to} \quad \hat{\mathbb{E}}_t \left[ D_{\text{KL}}(\pi_{\text{old}} \| \pi_{\text{new}}) \right] \leq \delta$$

**KL divergence** $D_{\text{KL}}(P \| Q) = \sum_a P(a) \log \frac{P(a)}{Q(a)}$ measures how different two distributions are:
- $D_{\text{KL}} = 0$: identical distributions
- $D_{\text{KL}}$ large: policies have changed substantially

**Intuition:** TRPO says "maximize the objective, but don't let the policy change so much that it becomes unrecognizable." It's like learning to ski — take the steepest line available, but only within the safe section of the slope.

**TRPO's problem:** Enforcing the KL constraint requires solving a **constrained optimization problem** (using conjugate gradients and a line search) — computationally expensive and difficult to implement correctly, especially with large neural networks.

This motivated the search for a simpler, equally effective approach — which led to **PPO**.

---

## 6. Proximal Policy Optimization (PPO)

> *PPO is arguably the most important practical RL algorithm in use today. It powers the training of ChatGPT, Claude, Gemini, and virtually every modern RLHF system. It is elegant, stable, and remarkably easy to implement — yet matches or exceeds TRPO's performance.*

### 6.1 The Probability Ratio

PPO's key quantity is the **probability ratio** — how much more (or less) likely the current policy is to take the same action compared to the old policy that collected the experience:

$$r_t(\boldsymbol{\theta}) = \frac{\pi(a_t \mid s_t;\, \boldsymbol{\theta})}{\pi(a_t \mid s_t;\, \boldsymbol{\theta}_{\text{old}})}$$

| Value of $r_t$ | Meaning |
|---------------|---------|
| $r_t = 1$ | Policy unchanged — same probability of taking $a_t$ |
| $r_t > 1$ | New policy more likely to take $a_t$ (policy shifted toward $a_t$) |
| $r_t < 1$ | New policy less likely to take $a_t$ (policy shifted away from $a_t$) |
| $r_t \gg 1$ or $r_t \ll 1$ | Policy changed drastically — **dangerous territory** |

The standard policy gradient objective, rewritten using this ratio (this is called the **surrogate objective**):

$$\mathcal{L}^{\text{CPI}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t \left[ r_t(\boldsymbol{\theta}) \cdot \hat{A}_t \right]$$

This objective naturally increases $r_t$ when $\hat{A}_t > 0$ (make the action more likely if it was good) and decreases $r_t$ when $\hat{A}_t < 0$ (make the action less likely if it was bad). But without any constraint, this can lead to arbitrarily large $r_t$ — the same instability as before.

### 6.2 The PPO Clipping Objective

PPO's elegant solution: **clip** the probability ratio so it can never deviate too far from 1:

$$\mathcal{L}^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t \left[ \min\!\left( r_t(\boldsymbol{\theta}) \cdot \hat{A}_t,\;\; \text{clip}(r_t(\boldsymbol{\theta}),\, 1-\varepsilon,\, 1+\varepsilon) \cdot \hat{A}_t \right) \right]$$

Where $\varepsilon$ is the clipping hyperparameter (typically $\varepsilon = 0.2$).

The `clip` function:

$$\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) = \begin{cases} 1-\varepsilon & \text{if } r_t < 1-\varepsilon \\ r_t & \text{if } 1-\varepsilon \leq r_t \leq 1+\varepsilon \\ 1+\varepsilon & \text{if } r_t > 1+\varepsilon \end{cases}$$

The `min` of the two terms ensures PPO takes the **pessimistic (conservative) estimate** at every point.

### 6.3 Understanding the Clipping: Four Cases

Let's build intuition by analyzing all four combinations of advantage sign and ratio size:

---

**Case 1: $\hat{A}_t > 0$ (good action) and $r_t < 1+\varepsilon$ (ratio within bounds)**

Both terms in the `min` give similar values. The update proceeds normally — the policy shifts toward $a_t$.

---

**Case 2: $\hat{A}_t > 0$ (good action) and $r_t > 1+\varepsilon$ (policy moved too far toward $a_t$)**

- Unclipped term: $r_t \cdot \hat{A}_t$ — large and positive (wants to keep going)
- Clipped term: $(1+\varepsilon) \cdot \hat{A}_t$ — smaller (capped)
- $\min$ selects the **clipped term** — gradient is **zero beyond the clip boundary**

The clipping acts as a **wall**: once the policy has already moved sufficiently toward $a_t$, no further gradient pushes in that direction. The update stops.

---

**Case 3: $\hat{A}_t < 0$ (bad action) and $r_t > 1-\varepsilon$ (ratio within bounds)**

Normal update — policy shifts away from $a_t$.

---

**Case 4: $\hat{A}_t < 0$ (bad action) and $r_t < 1-\varepsilon$ (policy already moved far away from $a_t$)**

- Unclipped term: $r_t \cdot \hat{A}_t$ — $r_t$ small and $\hat{A}_t$ negative → small product (less negative)
- Clipped term: $(1-\varepsilon) \cdot \hat{A}_t$ — fixed and negative
- $\min$ selects the **clipped term** — gradient is **zero beyond the clip boundary**

Again, clipping acts as a wall: once the policy has already moved sufficiently away from the bad action, no further gradient pushes.

**The key insight:** The clipping region $[1-\varepsilon, 1+\varepsilon]$ defines a **trust region in probability space**. PPO enforces this trust region via a simple element-wise clip operation — no constrained optimization, no conjugate gradients, no line searches required. It's a few lines of code.

### 6.4 Visual Summary of the PPO Clipping Objective

```
      L^CLIP
        ▲
        │            ┌──────────────────── clipped (no gradient here)
        │            │
        │         ╱──┘
        │        ╱   ← slope = Â_t (positive advantage)
        │       ╱
────────┼──────╱──────────────────── r_t (ratio)
        │   1-ε    1    1+ε
        │
   Case: Â_t > 0
   Gradient exists only in [1-ε, 1+ε].
   Beyond 1+ε: clipped — no incentive to push ratio further.

      L^CLIP
        ▲
        │
────────┼──────────────────────────── r_t
        │      1-ε   1    1+ε
        │       └──╲
        │           ╲ ← slope = Â_t (negative advantage)
        │            └──────────────── clipped (no gradient here)
        │
   Case: Â_t < 0
   Beyond 1-ε: clipped — no incentive to push ratio further.
```

### 6.5 The Full PPO Objective

Combining actor loss, critic loss, and entropy bonus:

$$\mathcal{L}^{\text{PPO}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t \left[ \mathcal{L}^{\text{CLIP}}_t(\boldsymbol{\theta}) - c_1 \mathcal{L}^{\text{VF}}_t(\boldsymbol{\theta}) + c_2 H\bigl[\pi(\cdot \mid s_t;\, \boldsymbol{\theta})\bigr] \right]$$

Where:
- $\mathcal{L}^{\text{CLIP}}_t$ = clipped policy gradient (actor)
- $\mathcal{L}^{\text{VF}}_t = (V(s_t;\, \boldsymbol{\theta}) - V_t^{\text{target}})^2$ = value function loss (critic)
- $H[\pi]$ = policy entropy (exploration bonus)
- $c_1 \approx 0.5$, $c_2 \approx 0.01$

### 6.6 The PPO Algorithm

```
╔══════════════════════════════════════════════════════════════════════╗
║             Proximal Policy Optimization (PPO)                      ║
╠══════════════════════════════════════════════════════════════════════╣
║ Initialize policy+value network π(a|s;θ), V(s;θ) [shared backbone] ║
║ Set ε = 0.2, T (rollout length), K (epochs), M (mini-batches)      ║
║                                                                      ║
║ For each iteration:                                                  ║
║                                                                      ║
║   ① COLLECT ROLLOUT (T steps in N parallel environments):           ║
║     Run current policy π_old → collect {sₜ, aₜ, rₜ, sₜ₊₁}        ║
║     Save old log-probs: log π_old(aₜ|sₜ)                           ║
║                                                                      ║
║   ② COMPUTE ADVANTAGES using GAE:                                    ║
║     δₜ = rₜ + γV(sₜ₊₁) − V(sₜ)                                    ║
║     Âₜ = Σₖ (γλ)ᵏ δₜ₊ₖ                                            ║
║     Normalize: Â ← (Â − mean(Â)) / (std(Â) + 1e-8)               ║
║                                                                      ║
║   ③ UPDATE for K epochs on mini-batches of size M:                  ║
║     For each mini-batch:                                             ║
║       Compute ratio: rₜ(θ) = π(aₜ|sₜ;θ) / π_old(aₜ|sₜ)          ║
║       Clipped loss: L^CLIP = min(rₜÂₜ, clip(rₜ,1-ε,1+ε)Âₜ)      ║
║       Value loss: L^VF = (V(sₜ;θ) − Vₜ_target)²                   ║
║       Entropy: H = -Σₐ π(a|sₜ) log π(a|sₜ)                        ║
║       Total: L = -L^CLIP + c₁·L^VF - c₂·H                         ║
║       Gradient step: θ ← θ − α·∇_θL                                ║
║                                                                      ║
║   ④ θ_old ← θ  (update old policy for next iteration)              ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Key PPO design choices:**

| Feature | Value | Purpose |
|---------|-------|---------|
| Clipping parameter $\varepsilon$ | 0.1–0.3 (typically 0.2) | Trust region size |
| K epochs per iteration | 3–10 | Reuse each rollout multiple times |
| Rollout length T | 128–2048 steps | More steps → better advantage estimates |
| GAE $\lambda$ | 0.95 | Balance bias-variance in advantages |
| Advantage normalization | Yes | Stabilizes gradient magnitudes across iterations |
| Shared backbone | Often yes | Efficient: one network for both $\pi$ and $V$ |

---

## 7. Worked Calculation: PPO Clipping in Action

**Setup:** A language model fine-tuning scenario (foreshadowing RLHF). The model must choose between two possible sentence continuations.

**Old policy** (before this iteration's update):
$$\pi_{\text{old}}(a_1 \mid s) = 0.3, \quad \pi_{\text{old}}(a_2 \mid s) = 0.7$$

**Current policy** (after some gradient steps in this iteration):
$$\pi_{\text{new}}(a_1 \mid s) = 0.55, \quad \pi_{\text{new}}(a_2 \mid s) = 0.45$$

**Clipping parameter:** $\varepsilon = 0.2$

**Scenario A:** Action $a_1$ was taken. Advantage $\hat{A} = +3.0$ (good action).

---

**Step 1 — Probability ratio:**

$$r(a_1) = \frac{\pi_{\text{new}}(a_1 \mid s)}{\pi_{\text{old}}(a_1 \mid s)} = \frac{0.55}{0.30} = \mathbf{1.833}$$

**Step 2 — Clipping bounds:** $[1 - 0.2,\; 1 + 0.2] = [0.8,\; 1.2]$

$r = 1.833 > 1.2$ → **ratio is outside the trust region** (policy moved too aggressively toward $a_1$)

**Step 3 — Compute both terms of the min:**

$$r_t \cdot \hat{A} = 1.833 \times 3.0 = 5.499$$

$$\text{clip}(r_t, 0.8, 1.2) \cdot \hat{A} = 1.2 \times 3.0 = 3.6$$

**Step 4 — Apply min:**

$$\mathcal{L}^{\text{CLIP}} = \min(5.499,\; 3.6) = \mathbf{3.6}$$

**Interpretation:** Although the policy strongly favors $a_1$ now (ratio = 1.833), the objective is capped at $3.6$ — not $5.499$. The gradient with respect to $\boldsymbol{\theta}$ at this point is **zero** beyond the clip (the policy has already moved enough). No further push toward $a_1$ will occur from this sample.

---

**Scenario B:** Same state. Now action $a_2$ was taken. Advantage $\hat{A} = -2.0$ (bad action).

$$r(a_2) = \frac{\pi_{\text{new}}(a_2 \mid s)}{\pi_{\text{old}}(a_2 \mid s)} = \frac{0.45}{0.70} = \mathbf{0.643}$$

$r = 0.643 < 0.8$ → **ratio outside trust region** (policy has already moved away from $a_2$)

$$r_t \cdot \hat{A} = 0.643 \times (-2.0) = -1.286$$

$$\text{clip}(r_t, 0.8, 1.2) \cdot \hat{A} = 0.8 \times (-2.0) = -1.6$$

$$\mathcal{L}^{\text{CLIP}} = \min(-1.286,\; -1.6) = \mathbf{-1.6}$$

**Interpretation:** Although the bad action $a_2$ has already been suppressed significantly (from 70% to 45%), the objective is capped. The gradient is zero — **no further punishment** of $a_2$ from this sample.

---

**Scenario C:** A fresh sample with ratio within bounds. Action $a_3$, $r = 1.05$, $\hat{A} = +2.5$.

$r = 1.05 \in [0.8, 1.2]$ → **within trust region**

$$r_t \cdot \hat{A} = 1.05 \times 2.5 = 2.625$$

$$\text{clip}(r_t, 0.8, 1.2) \cdot \hat{A} = 1.05 \times 2.5 = 2.625 \quad (\text{no clipping — same value})$$

$$\mathcal{L}^{\text{CLIP}} = \min(2.625, 2.625) = \mathbf{2.625}$$

**Gradient is active** — the policy will be updated normally. The full gradient signal flows through.

---

**Summary across the three scenarios:**

| Scenario | Ratio $r_t$ | $\hat{A}$ | Clipped? | $\mathcal{L}^{\text{CLIP}}$ | Gradient active? |
|----------|------------|----------|---------|---------------------------|-----------------|
| A | 1.833 | +3.0 | Yes ($r > 1.2$) | 3.6 | ✗ No — already moved far enough |
| B | 0.643 | −2.0 | Yes ($r < 0.8$) | −1.6 | ✗ No — already moved far enough |
| C | 1.05 | +2.5 | No | 2.625 | ✓ Yes — update normally |

PPO only applies gradient pressure where the policy still has room to improve — it stops pushing once the policy has moved to the boundary of the trust region.

---

## 8. PPO for RLHF: Training Language Models with Human Feedback

> *The most impactful application of everything in this course: PPO is the algorithm that transformed raw language models into helpful, harmless, and honest AI assistants. ChatGPT, Claude, and Gemini all have PPO in their DNA.*

### 8.1 The RLHF Pipeline

**Reinforcement Learning from Human Feedback (RLHF)** is a three-stage process:

```
Stage 1: Supervised Fine-Tuning (SFT)
─────────────────────────────────────────────────────
Pre-trained LLM  →  Fine-tune on high-quality demos  →  SFT Model
(next-token prediction)    (imitation learning)         π_SFT

Stage 2: Reward Model Training
─────────────────────────────────────────────────────
Prompt → Generate K responses → Humans rank them
       → Train Reward Model r_φ(prompt, response)
         to predict human preferences

Stage 3: PPO Fine-Tuning
─────────────────────────────────────────────────────
LLM Policy (actor)  +  Value Head (critic)
         ↓ generate response
Reward Model r_φ scores it
         ↓
PPO updates LLM weights to maximize reward
         ↓ (repeat thousands of times)
RLHF-aligned model: helpful, harmless, honest
```

### 8.2 The Language Model as a Policy

Every concept from this week maps directly onto the LLM setting:

| RL Concept | LLM Setting |
|-----------|-------------|
| State $s_t$ | The prompt + all tokens generated so far |
| Action $a_t$ | The next token to generate (vocabulary size ~50,000) |
| Policy $\pi(a \mid s;\, \boldsymbol{\theta})$ | The language model's softmax over the vocabulary |
| Reward $r$ | Human preference score (from the reward model) |
| Episode | One complete prompt-response pair |
| Actor network | The language model being fine-tuned |
| Critic network | A separate value head added to the LLM |

### 8.3 The KL Penalty — Preventing Reward Hacking

A critical addition for LLM fine-tuning: a **KL divergence penalty** between the fine-tuned policy and the original SFT policy:

$$r_{\text{total}}(s, a) = r_\phi(s, a) - \beta \cdot D_{\text{KL}}\bigl(\pi_{\boldsymbol{\theta}}(\cdot \mid s) \,\|\, \pi_{\text{SFT}}(\cdot \mid s)\bigr)$$

**Why this is essential — Reward Hacking:**

Without the KL penalty, the PPO agent quickly learns to **game the reward model** — finding degenerate responses that score highly on the reward model but are nonsensical or unhelpful to humans. For example, the model might learn to generate repetitive praise ("Great question! Excellent! Amazing!") because a poorly trained reward model assigns high scores to enthusiastic language.

The KL penalty says: "Maximize human preference reward, but don't drift too far from the original SFT model." It's a regularization that keeps the language coherent and anchored to the pre-training distribution.

**Tuning $\beta$:**
- $\beta$ too small → reward hacking, incoherent outputs
- $\beta$ too large → the model barely changes from SFT, ignores human feedback
- Typical $\beta \approx 0.01$ to $0.05$

### 8.4 Why PPO Specifically?

Other RL algorithms were considered for RLHF. PPO won because:

1. **Stability**: The clipping mechanism prevents catastrophic forgetting of language capabilities accumulated during pre-training — vital when the policy has billions of parameters.
2. **Sample efficiency**: Multiple epochs on each rollout reduce the number of expensive human annotations needed.
3. **Simplicity**: Compared to TRPO, PPO requires no second-order optimization — scales to 100B+ parameter models.
4. **On-policy**: Ensures the reward model evaluates outputs from the current policy distribution, not stale old outputs.

---

## 9. Asynchronous Advantage Actor-Critic (A3C)

> *What if multiple students each study a different chapter simultaneously, then share their notes? A3C applies exactly this parallelism to policy gradient learning.*

### 9.1 The Idea

**A3C** (Mnih et al., 2016) runs multiple **asynchronous workers** — each worker has its own copy of the environment and its own local copy of the networks. Workers independently collect experience, compute gradients, and push gradient updates to a **global shared network**:

```
┌────────────────────────────────────────────────────┐
│           Global Network (θ_global, w_global)      │
│   ← Gradient updates pushed from all workers       │
│   → Weight copies pulled by all workers            │
└──────────────┬──────────────────┬──────────────────┘
               │                  │
    ┌──────────▼──┐         ┌─────▼────────┐
    │  Worker 1   │         │   Worker N   │
    │  Env copy   │   ...   │   Env copy   │
    │  Local θ,w  │         │   Local θ,w  │
    └─────────────┘         └──────────────┘
```

**Key benefits:**

| Benefit | Explanation |
|---------|-------------|
| **Decorrelated experience** | Each worker explores different parts of the environment simultaneously — gradients are naturally diverse |
| **Faster wall-clock time** | Parallelism replaces the replay buffer for decorrelation |
| **No replay buffer needed** | The asynchronous updates naturally break temporal correlations |
| **CPU-efficient** | Works on CPU clusters, not just GPUs |

**A2C vs. A3C:** A2C is the **synchronous** version — all workers collect experience simultaneously, then update together. A2C is simpler, more reproducible, and works better with GPUs. A3C is the original asynchronous version. In practice today, A2C + PPO is the dominant choice.

---

## 10. The Complete Landscape: Comparing All Methods

We now have the full picture of model-free RL methods from Weeks 3–7:

| Algorithm | Type | Action Space | Policy | Key Strength | Key Weakness |
|-----------|------|-------------|--------|-------------|-------------|
| **Monte Carlo** | Value-based | Discrete | Implicit | Unbiased | High variance, must wait for episode end |
| **Q-Learning** | Value-based, off-policy | Discrete | Implicit, deterministic | Sample efficient, can reuse data | Discrete actions only |
| **DQN** | Value-based, off-policy | Discrete | Implicit, deterministic | Scales to large state spaces | Discrete actions, no continuous control |
| **REINFORCE** | Policy gradient, on-policy | Any | Explicit, stochastic | Correct gradient; continuous actions | Very high variance, slow |
| **REINFORCE + baseline** | Policy gradient, on-policy | Any | Explicit, stochastic | Lower variance | Still MC — must complete episode |
| **A2C** | Actor-Critic, on-policy | Any | Explicit, stochastic | Step-by-step updates, low variance | On-policy — no data reuse |
| **PPO** | Actor-Critic, on-policy | Any | Explicit, stochastic | Stable, reuses data (K epochs), RLHF | More hyperparameters |
| **TRPO** | Actor-Critic, on-policy | Any | Explicit, stochastic | Theoretically sound trust region | Computationally expensive |

**The modern consensus:** For most practical applications today, **PPO** is the starting point. It is:
- Stable enough for LLM fine-tuning (billions of parameters)
- Efficient enough for robotics (real hardware at risk)
- Simple enough to implement correctly in ~100 lines of PyTorch
- Expressive enough for both discrete and continuous action spaces

---

## 11. Real-World Applications

| Application | Algorithm | Why Actor-Critic / PPO |
|-------------|-----------|----------------------|
| **ChatGPT / Claude / Gemini alignment** (OpenAI, Anthropic, Google) | PPO + RLHF | Stable fine-tuning of 100B+ parameter LLMs; KL penalty prevents reward hacking |
| **OpenAI Five (Dota 2)** | PPO | 5-agent team coordination; continuous game state; defeated world champions |
| **AlphaGo / AlphaStar** | A3C + policy gradient | Actor-Critic provides move probabilities; critic guides MCTS |
| **Robot locomotion** (ETH Zürich, Boston Dynamics) | PPO | Continuous joint control; learns to walk/run on real hardware after sim training |
| **Drone racing** (Autonomous flight faster than human champions) | PPO | Continuous 3D thrust control; millisecond decisions |
| **Protein folding optimization** (DeepMind) | Actor-Critic variants | Continuous molecular conformation space |
| **Industrial robot manipulation** | PPO (sim-to-real) | 6-DOF arm control; trained in simulation, deployed on hardware |
| **Code generation RL** (GitHub Copilot variants) | PPO + RLHF | Reward = code execution success; aligns coding LLMs with functional correctness |

---

## 12. Summary

| Concept | Key Idea |
|---------|----------|
| **Actor-Critic** | Two networks: actor $\pi(a \mid s;\,\boldsymbol{\theta})$ acts; critic $V(s;\,\boldsymbol{w})$ evaluates |
| **TD Advantage** | $\hat{A}_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ — immediate, low-variance signal for actor updates |
| **Entropy bonus** | Adds $-\beta H(\pi)$ to loss — prevents premature determinism, encourages exploration |
| **GAE** | Generalized Advantage Estimation — exponentially weighted sum of TD errors; $\lambda$ tunes bias-variance |
| **A2C** | Synchronous Actor-Critic with advantage estimates; stable, parallelizable |
| **The cliff edge** | Large policy gradient steps can catastrophically collapse the policy — must constrain step size |
| **KL divergence** | $D_{\text{KL}}(P \| Q)$ measures policy change — used in TRPO as a hard constraint |
| **TRPO** | Trust region via KL constraint — correct but computationally expensive |
| **Probability ratio** | $r_t(\boldsymbol{\theta}) = \pi_{\text{new}}(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$ — measures policy shift |
| **PPO clipping** | $\mathcal{L}^{\text{CLIP}} = \min(r_t \hat{A}_t,\; \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t)$ — trust region via clipping |
| **Trust region effect** | Clipping zeros the gradient once the ratio exceeds $[1-\varepsilon, 1+\varepsilon]$ — no further push |
| **PPO K epochs** | Reuse each rollout for multiple gradient steps without leaving the trust region |
| **RLHF pipeline** | SFT → Reward Model → PPO fine-tuning; aligns LLMs with human preferences |
| **KL penalty in RLHF** | $r_{\text{total}} = r_\phi - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{SFT}})$ — prevents reward hacking |
| **A3C** | Asynchronous workers push gradients to global network — decorrelates experience without replay buffer |
| **PPO dominates** | Balances stability, efficiency, and simplicity — the default algorithm for modern deep RL |

---

## 13. Exercises

### Conceptual Questions

1. **The Actor-Critic architecture fixes the high variance of REINFORCE.** Explain precisely which component of REINFORCE is replaced, what it is replaced with, and why this reduces variance while introducing bias. Is this tradeoff worthwhile?

2. **The TD advantage** $\hat{A}_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ is described as a signal for both the actor and the critic. Explain how the same quantity is used to update each network, and what each update is trying to accomplish.

3. **TRPO constrains the KL divergence** between old and new policy. PPO clips the probability ratio instead. Explain in your own words why these two mechanisms achieve a similar effect — both preventing catastrophically large policy updates — but PPO does so with much less computational cost.

4. **In RLHF**, why is the KL penalty $-\beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{SFT}})$ added to the reward? Give a concrete example of what "reward hacking" looks like in an LLM context, and explain how the KL penalty prevents it.

5. **A robot learning to walk** uses PPO with $\varepsilon = 0.2$. After one rollout, some step has a probability ratio of 1.45 and a positive advantage. Does PPO update the policy for this step? What about a step with ratio 1.15? What does this asymmetry achieve?

---

### Calculation Problems

**Problem 1 — TD Advantage Estimation**

A critic network produces: $V(s_t) = 6.0$, $V(s_{t+1}) = 8.5$.
The agent takes an action and receives $r_{t+1} = +3$.

Parameters: $\gamma = 0.9$.

(a) Compute the TD advantage $\hat{A}_t$.

(b) Is this a positive or negative advantage? What should the actor do in response?

(c) Compute the critic's update target and the critic loss $\mathcal{L}_{\text{critic}} = \hat{A}_t^2$.

---

**Problem 2 — A2C Actor Update**

Using the advantage from Problem 1 ($\hat{A}_t = +4.65$). The actor's current policy in state $s_t$: $\pi(\text{Left}) = 0.4$, $\pi(\text{Right}) = 0.6$. The agent took action `Right`.

Learning rate: $\alpha = 0.02$.

(a) Compute the score function $\partial \log\pi(\text{Right}) / \partial \theta_{\text{Right}}$ and $\partial \log\pi(\text{Right}) / \partial \theta_{\text{Left}}$.

(b) Compute $\Delta\theta_{\text{Right}}$ and $\Delta\theta_{\text{Left}}$.

(c) In which direction does $\pi(\text{Right})$ change? Does this make intuitive sense given the positive advantage?

---

**Problem 3 — PPO Clipping Calculation**

For the following three transitions sampled from a rollout, compute $\mathcal{L}^{\text{CLIP}}$ for each. Use $\varepsilon = 0.2$.

| Transition | $\pi_{\text{new}}(a \mid s)$ | $\pi_{\text{old}}(a \mid s)$ | $\hat{A}$ |
|-----------|------------------------------|------------------------------|-----------|
| 1 | 0.60 | 0.40 | +2.0 |
| 2 | 0.20 | 0.50 | −3.0 |
| 3 | 0.35 | 0.30 | +1.5 |

For each transition:
(a) Compute the ratio $r_t = \pi_{\text{new}} / \pi_{\text{old}}$.
(b) Compute $r_t \cdot \hat{A}$ and $\text{clip}(r_t, 0.8, 1.2) \cdot \hat{A}$.
(c) Compute $\mathcal{L}^{\text{CLIP}} = \min(\cdot)$ and state whether the gradient is active.

---

**Problem 4 — GAE Computation**

An episode has 3 steps. The critic estimates: $V(s_0) = 2.0$, $V(s_1) = 4.0$, $V(s_2) = 3.0$, $V(s_3) = 0$ (terminal). Rewards: $r_1 = +1$, $r_2 = +2$, $r_3 = +8$.

Parameters: $\gamma = 0.9$, $\lambda = 0.95$.

(a) Compute the TD errors $\delta_0$, $\delta_1$, $\delta_2$.

(b) Compute the GAE advantages $\hat{A}_0^{\text{GAE}}$, $\hat{A}_1^{\text{GAE}}$, $\hat{A}_2^{\text{GAE}}$ using:
$$\hat{A}_t^{\text{GAE}} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

(c) Compare $\hat{A}_0^{\text{GAE}}$ with $\lambda=0.95$ to the pure 1-step TD advantage $\delta_0$. What is the effect of including the longer-horizon terms?

---

### Answer Key

**Problem 1:**

(a) $\hat{A}_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t) = 3 + 0.9 \times 8.5 - 6.0 = 3 + 7.65 - 6.0 = \mathbf{+4.65}$

(b) **Positive advantage** — the outcome exceeded the critic's expectations. The actor should **increase** the probability of the action taken in state $s_t$.

(c) Critic target $= r_{t+1} + \gamma V(s_{t+1}) = 3 + 7.65 = 10.65$. Critic loss $= \hat{A}_t^2 = 4.65^2 = \mathbf{21.62}$

---

**Problem 2:**

(a) Score function for taken action `Right`:
$\partial\log\pi(\text{Right})/\partial\theta_{\text{Right}} = 1 - \pi(\text{Right}) = 1 - 0.6 = \mathbf{+0.4}$
$\partial\log\pi(\text{Right})/\partial\theta_{\text{Left}} = -\pi(\text{Left}) = \mathbf{-0.4}$

(b) $\Delta\theta_{\text{Right}} = 0.02 \times 4.65 \times 0.4 = \mathbf{+0.0372}$
$\Delta\theta_{\text{Left}} = 0.02 \times 4.65 \times (-0.4) = \mathbf{-0.0372}$

(c) $\pi(\text{Right})$ **increases** (positive $\Delta\theta_{\text{Right}}$ → higher softmax score → higher probability). Yes — the advantage was positive (+4.65), meaning taking `Right` led to a better outcome than the critic predicted. The actor correctly increases the probability of `Right`.

---

**Problem 3:**

**Transition 1:** $r_t = 0.60/0.40 = 1.50$; $\hat{A} = +2.0$

- $r_t \hat{A} = 1.50 \times 2.0 = 3.0$
- $\text{clip}(1.50, 0.8, 1.2) \times 2.0 = 1.2 \times 2.0 = 2.4$
- $\mathcal{L}^{\text{CLIP}} = \min(3.0, 2.4) = \mathbf{2.4}$ — **Gradient inactive** (clipped, $r > 1.2$)

**Transition 2:** $r_t = 0.20/0.50 = 0.40$; $\hat{A} = -3.0$

- $r_t \hat{A} = 0.40 \times (-3.0) = -1.2$
- $\text{clip}(0.40, 0.8, 1.2) \times (-3.0) = 0.8 \times (-3.0) = -2.4$
- $\mathcal{L}^{\text{CLIP}} = \min(-1.2, -2.4) = \mathbf{-2.4}$ — **Gradient inactive** (clipped, $r < 0.8$)

**Transition 3:** $r_t = 0.35/0.30 = 1.167$; $\hat{A} = +1.5$

- $r_t \hat{A} = 1.167 \times 1.5 = 1.75$
- $\text{clip}(1.167, 0.8, 1.2) \times 1.5 = 1.167 \times 1.5 = 1.75$ (no clipping — $r \in [0.8, 1.2]$)
- $\mathcal{L}^{\text{CLIP}} = \min(1.75, 1.75) = \mathbf{1.75}$ — **Gradient active** ✓

---

**Problem 4:**

(a) $\gamma\lambda = 0.9 \times 0.95 = 0.855$

$\delta_0 = r_1 + \gamma V(s_1) - V(s_0) = 1 + 0.9(4.0) - 2.0 = 1 + 3.6 - 2.0 = \mathbf{+2.6}$

$\delta_1 = r_2 + \gamma V(s_2) - V(s_1) = 2 + 0.9(3.0) - 4.0 = 2 + 2.7 - 4.0 = \mathbf{+0.7}$

$\delta_2 = r_3 + \gamma V(s_3) - V(s_2) = 8 + 0.9(0) - 3.0 = \mathbf{+5.0}$

(b) Working backwards:

$\hat{A}_2^{\text{GAE}} = \delta_2 = \mathbf{5.0}$

$\hat{A}_1^{\text{GAE}} = \delta_1 + (\gamma\lambda)\delta_2 = 0.7 + 0.855 \times 5.0 = 0.7 + 4.275 = \mathbf{4.975}$

$\hat{A}_0^{\text{GAE}} = \delta_0 + (\gamma\lambda)\delta_1 + (\gamma\lambda)^2\delta_2 = 2.6 + 0.855(0.7) + 0.855^2(5.0)$
$= 2.6 + 0.5985 + 0.7310 \times 5.0 = 2.6 + 0.5985 + 3.655 = \mathbf{6.854}$

(c) Pure 1-step TD: $\hat{A}_0^{\text{1-step}} = \delta_0 = +2.6$

GAE ($\lambda=0.95$): $\hat{A}_0^{\text{GAE}} = +6.854$

The longer-horizon terms add $+4.254$ to the advantage estimate. This reflects the large terminal reward of $+8$ at $s_3$ — information that the 1-step TD estimate at $t=0$ couldn't see because it only looks one step ahead. GAE "sees" the future $+8$ and correctly attributes significant credit to the action taken at $t=0$, making the policy gradient update much more informative.

---

*This completes Part 1 of the course — Reinforcement Learning. In Week 8, we turn to Part 2: LLMs and Agentic AI. We'll look under the hood of the Transformer architecture and see how the systems we just trained with PPO (like ChatGPT and Claude) are structured — beginning the journey from RL agents to fully autonomous AI systems.*
