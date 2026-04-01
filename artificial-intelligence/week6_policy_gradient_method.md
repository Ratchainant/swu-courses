# Week 6 — Policy Gradient Methods

---

## 1. The Limits of Thinking in Values

> *A jazz musician doesn't compute a "value score" for every possible note before playing. They've trained their instincts so deeply that their fingers move directly to the right note — without any intermediate calculation. Policy Gradient methods teach RL agents to do the same: act directly, without the detour through values.*

### 1.1 What DQN Cannot Do

In Weeks 4 and 5, we built powerful value-based agents. Q-Learning and DQN learn to answer the question *"How good is action $a$ in state $s$?"*, then act greedily by picking the highest-valued action. This works beautifully — until it doesn't.

**Fundamental Limitation 1 — Discrete actions only.**

DQN requires computing $\arg\max_a Q(s, a)$ — which means enumerating and comparing all possible actions. For a finite set like $\{\text{Left}, \text{Right}, \text{Jump}\}$, this is trivial. But many real problems have **continuous action spaces**:

| Problem | Action space | Why $\arg\max$ fails |
|---------|-------------|---------------------|
| Robot arm | Joint torques: $[-10\,\text{Nm}, +10\,\text{Nm}]^6$ | Infinite actions; can't enumerate |
| Autonomous car | Steering angle + throttle (continuous) | Same — uncountably many choices |
| Drone flight | 4 rotor thrust values (continuous) | $\arg\max$ over $\mathbb{R}^4$ is meaningless |
| Drug dosage | Dosage in $[0, 100\,\text{mg}]$ | Continuous; best value is a real number |

You cannot build a Q-table — or even a Q-network with one output per action — when actions are real-valued vectors.

**Fundamental Limitation 2 — Deterministic policies are sometimes suboptimal.**

DQN's greedy policy is **deterministic**: given state $s$, always take action $a^* = \arg\max_a Q(s, a)$. But there are situations where the **optimal policy is stochastic** — randomly mixing between actions.

**Intuitive Example — Rock-Paper-Scissors:**

If you always play the same move, a smart opponent will figure it out and beat you every time. The only Nash-optimal strategy is to randomize uniformly — play each action with probability 1/3. A deterministic policy *cannot* represent this; a stochastic policy does it naturally.

**Intuitive Example — Partially Observable Environments:**

When the agent can't fully observe the state (e.g., poker — you can't see the opponent's cards), randomizing your actions is necessary to prevent opponents from predicting your behavior. The optimal *behavioral* strategy is explicitly stochastic.

**Fundamental Limitation 3 — The policy is implicit.**

In DQN, the policy is not stored anywhere explicitly. It's *implied* by $\arg\max_a Q(s, a)$. This makes it hard to interpret, constrain, or fine-tune the policy directly. We can't say "I want the agent to prefer turning left slightly more than turning right in this particular state" without modifying the entire Q-function.

### 1.2 The Policy Gradient Idea

Instead of learning a value function and deriving a policy from it, **Policy Gradient** methods learn the policy **directly**:

$$\pi(a \mid s;\, \boldsymbol{\theta})$$

This is a **parameterized policy** — a function (usually a neural network with weights $\boldsymbol{\theta}$) that takes a state $s$ as input and outputs a **probability distribution over actions**.

The agent's goal is to find $\boldsymbol{\theta}$ that **maximizes expected return**. Since the policy is differentiable with respect to $\boldsymbol{\theta}$, we can do **gradient ascent** directly on the objective — nudging $\boldsymbol{\theta}$ to make good actions more probable and bad actions less probable.

**The central shift in thinking:**

| Value-Based (DQN) | Policy Gradient |
|-------------------|----------------|
| Learn $Q(s, a)$ — how good each action is | Learn $\pi(a \mid s;\, \boldsymbol{\theta})$ — how often to pick each action |
| Policy derived indirectly: $\pi = \arg\max_a Q$ | Policy is the primary object of learning |
| Must enumerate actions to act | Samples directly from the learned distribution |
| Deterministic policy | Stochastic policy — naturally handles randomness |
| Fails on continuous actions | Works natively on continuous actions |

---

## 2. Parameterizing a Policy

> *A policy is just a rule: "given this situation, what should I do?" We need to represent that rule in a way a computer can learn and refine. That's parameterization.*

### 2.1 What Parameterization Means

A **parameterized policy** $\pi(a \mid s;\, \boldsymbol{\theta})$ is a function that maps any state $s$ to a probability distribution over actions, controlled by learnable parameters $\boldsymbol{\theta}$.

Think of it as a dial: $\boldsymbol{\theta}$ is the dial's position. Turning the dial changes the policy — makes the agent more aggressive or cautious, more likely to go left or right. Our job is to find the dial position that maximizes total reward.

**Desirable properties:**
1. **Differentiable**: we must be able to compute $\nabla_{\boldsymbol{\theta}} \pi(a \mid s;\, \boldsymbol{\theta})$ — otherwise gradient methods don't apply.
2. **Proper distribution**: $\sum_a \pi(a \mid s;\, \boldsymbol{\theta}) = 1$ and $\pi(a \mid s;\, \boldsymbol{\theta}) \geq 0$ for all $a$.
3. **Expressive**: can represent a wide range of behaviors.

### 2.2 Softmax Policy (Discrete Actions)

For discrete action spaces, the most common parameterization computes a **preference score** $h(s, a;\, \boldsymbol{\theta})$ for each action (e.g., the output of a neural network), then converts scores to probabilities via **softmax**:

$$\pi(a \mid s;\, \boldsymbol{\theta}) = \frac{e^{h(s, a;\, \boldsymbol{\theta})}}{\sum_{a'} e^{h(s, a';\, \boldsymbol{\theta})}}$$

**Properties of softmax:**
- All probabilities are positive (since $e^x > 0$ always)
- All probabilities sum to 1 (by construction of the denominator)
- The action with the highest score gets the most probability — but all actions get *some* probability (never exactly 0)
- As training progresses, one action's score grows much larger than others → its probability approaches 1 (near-deterministic)

**Intuitive Example — Choosing a Study Method:**

The network computes scores for three study strategies:

| Strategy | Score $h(s, a)$ | $e^{h}$ | Probability |
|----------|-----------------|---------|-------------|
| Read notes | 1.2 | $e^{1.2} = 3.32$ | $3.32 / 10.28 = 32.3\%$ |
| Practice problems | 2.1 | $e^{2.1} = 8.17$ | $8.17 / 10.28 = 79.5\%$ |
| Watch videos | 0.0 | $e^{0.0} = 1.00$ | $1.00 / 10.28 = \;\;9.7\%$ |
| **Total** | | **10.28** | **~100%** (rounding) |

Wait — these sum to 121.5%, let me redo with the correct total: $3.32 + 8.17 + 1.00 = 12.49$.

| Strategy | Score $h(s, a)$ | $e^{h}$ | Probability |
|----------|-----------------|---------|-------------|
| Read notes | 1.2 | 3.32 | $3.32/12.49 = 26.6\%$ |
| Practice problems | 2.1 | 8.17 | $8.17/12.49 = 65.4\%$ |
| Watch videos | 0.0 | 1.00 | $1.00/12.49 = \;\;8.0\%$ |
| **Total** | | **12.49** | **100%** ✓ |

The agent would most often choose "Practice problems" — but occasionally tries the others. As learning progresses and practice problems keep paying off, the score gap widens and the probability approaches 100%.

### 2.3 Gaussian Policy (Continuous Actions)

For continuous action spaces, the standard parameterization outputs a **Gaussian distribution**:

$$\pi(a \mid s;\, \boldsymbol{\theta}) = \mathcal{N}(\mu(s;\, \boldsymbol{\theta}),\; \sigma^2(s;\, \boldsymbol{\theta}))$$

The network outputs two values per action dimension:
- **Mean** $\mu(s;\, \boldsymbol{\theta})$: the "best guess" action
- **Standard deviation** $\sigma(s;\, \boldsymbol{\theta})$: how confident the agent is (large $\sigma$ = lots of exploration, small $\sigma$ = committed to $\mu$)

The agent samples the actual action: $a \sim \mathcal{N}(\mu, \sigma^2)$

**Intuition:** A robot arm learning to grasp an object might output $\mu = 45°$ (best estimated joint angle) and $\sigma = 10°$ (uncertainty). It samples an actual angle like $43.2°$ or $48.7°$ from this distribution. As training progresses, $\sigma$ shrinks — the agent becomes more confident and precise.

**The log-probability** of a Gaussian action $a$ is needed for gradient computation:

$$\log \pi(a \mid s;\, \boldsymbol{\theta}) = -\frac{(a - \mu)^2}{2\sigma^2} - \log\sigma - \frac{1}{2}\log(2\pi)$$

---

## 3. The Objective Function

> *Before we can improve the policy, we need a precise mathematical definition of "better." That's the objective function.*

### 3.1 Defining What We Want to Maximize

The **policy gradient objective** is the expected total discounted return when the agent follows policy $\pi_{\boldsymbol{\theta}}$:

$$J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ G(\tau) \right] = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=0}^{T} \gamma^t r_{t+1} \right]$$

Where:
- $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_T)$ is a **trajectory** — one complete episode
- $\tau \sim \pi_{\boldsymbol{\theta}}$ means the trajectory is generated by following policy $\pi_{\boldsymbol{\theta}}$
- $G(\tau)$ is the total discounted return of that trajectory

**In plain language:** $J(\boldsymbol{\theta})$ is the average score the agent would get if we ran many episodes following policy $\pi_{\boldsymbol{\theta}}$.

We want: $\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$

### 3.2 The Challenge: Differentiating Through a Distribution

Here's the core difficulty: $J(\boldsymbol{\theta})$ is an expectation *over trajectories generated by* $\pi_{\boldsymbol{\theta}}$. When we change $\boldsymbol{\theta}$, we change both:
1. Which actions the agent takes at each step (the distribution we sample from)
2. The resulting states and rewards

We need $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ to do gradient ascent. But how do you differentiate through a stochastic sampling process? The sampled trajectory $\tau$ is a random variable — you can't backpropagate through randomness directly.

**The log-derivative trick** (also called the REINFORCE trick or score function estimator) solves this elegantly.

---

## 4. The Policy Gradient Theorem

> *The central theorem of this entire week. It converts an impossible differentiation problem into a tractable expectation that we can estimate from experience — without knowing the environment's dynamics.*

### 4.1 Derivation via the Log-Derivative Trick

Start with the objective:

$$J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} [G(\tau)] = \int_\tau G(\tau)\, p(\tau;\, \boldsymbol{\theta})\, d\tau$$

Where $p(\tau;\, \boldsymbol{\theta})$ is the probability of trajectory $\tau$ under policy $\boldsymbol{\theta}$.

Take the gradient with respect to $\boldsymbol{\theta}$:

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \int_\tau G(\tau)\, \nabla_{\boldsymbol{\theta}} p(\tau;\, \boldsymbol{\theta})\, d\tau$$

Now apply the **log-derivative identity**: for any function $f$,

$$\nabla_{\boldsymbol{\theta}} f = f \cdot \nabla_{\boldsymbol{\theta}} \log f \quad \Longleftrightarrow \quad \nabla_{\boldsymbol{\theta}} \log f = \frac{\nabla_{\boldsymbol{\theta}} f}{f}$$

Apply this to $p(\tau;\, \boldsymbol{\theta})$:

$$\nabla_{\boldsymbol{\theta}} p(\tau;\, \boldsymbol{\theta}) = p(\tau;\, \boldsymbol{\theta}) \cdot \nabla_{\boldsymbol{\theta}} \log p(\tau;\, \boldsymbol{\theta})$$

Substituting back:

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \int_\tau G(\tau)\, p(\tau;\, \boldsymbol{\theta}) \cdot \nabla_{\boldsymbol{\theta}} \log p(\tau;\, \boldsymbol{\theta})\, d\tau = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \bigl[ G(\tau) \cdot \nabla_{\boldsymbol{\theta}} \log p(\tau;\, \boldsymbol{\theta}) \bigr]$$

Now expand $\log p(\tau;\, \boldsymbol{\theta})$. A trajectory's probability factorizes as:

$$p(\tau;\, \boldsymbol{\theta}) = \underbrace{p(s_0)}_{\text{initial state}} \prod_{t=0}^{T-1} \underbrace{\pi(a_t \mid s_t;\, \boldsymbol{\theta})}_{\text{policy}} \underbrace{\mathcal{T}(s_{t+1} \mid s_t, a_t)}_{\text{environment dynamics}}$$

Taking the log:

$$\log p(\tau;\, \boldsymbol{\theta}) = \log p(s_0) + \sum_{t=0}^{T-1} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta}) + \sum_{t=0}^{T-1} \log \mathcal{T}(s_{t+1} \mid s_t, a_t)$$

When we take $\nabla_{\boldsymbol{\theta}}$, the terms that don't involve $\boldsymbol{\theta}$ — $\log p(s_0)$ and $\log \mathcal{T}$ — **vanish**. This is the magic: the environment dynamics $\mathcal{T}$ disappear completely!

$$\nabla_{\boldsymbol{\theta}} \log p(\tau;\, \boldsymbol{\theta}) = \sum_{t=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta})$$

Substituting:

$$\boxed{\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ G(\tau) \cdot \sum_{t=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta}) \right]}$$

Or, step-wise (the form used in practice):

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\pi_{\boldsymbol{\theta}}} \left[ G_t \cdot \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta}) \right]$$

### 4.2 What This Theorem Is Saying

Break the update rule into parts:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot G_t \cdot \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta})$$

| Term | Meaning |
|------|---------|
| $\nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta})$ | **Score function** — the direction in parameter space that increases the log-probability of action $a_t$ in state $s_t$ |
| $G_t$ | **Return** — how good the episode turned out after taking action $a_t$ |
| $G_t \cdot \nabla_{\boldsymbol{\theta}} \log \pi(\cdots)$ | **Weighted direction** — push the policy in the direction that increases $a_t$'s probability, scaled by how good the outcome was |

**In plain language, three beautiful rules:**
1. If $G_t > 0$ (the episode went well), **increase** the probability of all actions taken. We want to do what worked.
2. If $G_t < 0$ (the episode went badly), **decrease** the probability of all actions taken. Avoid what failed.
3. Actions that led to *higher* $G_t$ get *bigger* updates — the magnitude of the push is proportional to the outcome.

**Analogy — Film Director's Feedback:**

A director watches an actor's improvised scene. If the audience loved it (+high $G_t$), the director says: "Do exactly what you just did, but even more of it." If the audience hated it (−low $G_t$), the director says: "Whatever you just did — do less of that." The Policy Gradient update is precisely this feedback loop, encoded in mathematics.

### 4.3 The Critical Insight: No Model Needed

Look at what disappeared: the environment transition $\mathcal{T}(s_{t+1} \mid s_t, a_t)$ dropped out completely. The Policy Gradient Theorem gives us a gradient formula that depends **only on**:
- The policy $\pi(a \mid s;\, \boldsymbol{\theta})$ — which we control
- The observed returns $G_t$ — which we measure from experience

We never need to know *how* the environment works. This makes Policy Gradient methods fully model-free, just like Q-Learning.

---

## 5. The REINFORCE Algorithm

> *REINFORCE is the simplest, most direct implementation of the Policy Gradient Theorem. Named by Williams (1992), it's the "Hello World" of policy gradient methods — and a masterpiece of clarity.*

### 5.1 The Algorithm

REINFORCE is a **Monte Carlo** policy gradient algorithm — it runs complete episodes, computes the actual return $G_t$ at each step, then updates the policy using the Policy Gradient Theorem.

```
╔══════════════════════════════════════════════════════════════════════╗
║                    REINFORCE Algorithm                               ║
╠══════════════════════════════════════════════════════════════════════╣
║ Initialize policy network π(a|s; θ) with random weights θ           ║
║ Set learning rate α                                                  ║
║                                                                      ║
║ For each episode i = 1, 2, 3, …:                                    ║
║                                                                      ║
║   ① GENERATE EPISODE by following π(·|·; θ):                        ║
║       s₀ → a₀ → r₁ → s₁ → a₁ → r₂ → s₂ → ⋯ → sT                ║
║                                                                      ║
║   ② COMPUTE RETURNS for each time step t:                            ║
║       Gₜ = rₜ₊₁ + γ rₜ₊₂ + γ² rₜ₊₃ + ⋯ + γᵀ⁻ᵗ⁻¹ rT             ║
║       (work backwards from the terminal step)                        ║
║                                                                      ║
║   ③ FOR EACH time step t = 0, 1, …, T−1:                            ║
║       Compute gradient: ∇_θ log π(aₜ | sₜ; θ)                      ║
║       Update policy:                                                 ║
║       θ ← θ + α · Gₜ · ∇_θ log π(aₜ | sₜ; θ)                     ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Key properties:**
- **Monte Carlo**: must complete the full episode before any updates
- **On-policy**: uses experience generated by the current policy $\pi_{\boldsymbol{\theta}}$
- **No replay buffer**: generated experience is used once and discarded (stale experience from an old $\boldsymbol{\theta}$ would bias the gradient)
- **Gradient ascent** (not descent): we add $\alpha \cdot G_t \cdot \nabla \log \pi$ to $\boldsymbol{\theta}$, because we're maximizing $J$

### 5.2 Computing the Score Function $\nabla_{\boldsymbol{\theta}} \log \pi$

For a **softmax policy** with linear preferences $h(s, a;\, \boldsymbol{\theta}) = \boldsymbol{\theta}^T \phi(s, a)$ (where $\phi(s, a)$ is a feature vector for state-action pair), the gradient has a particularly clean form:

$$\nabla_{\boldsymbol{\theta}} \log \pi(a \mid s;\, \boldsymbol{\theta}) = \phi(s, a) - \sum_{a'} \pi(a' \mid s;\, \boldsymbol{\theta})\, \phi(s, a')$$

This is the **feature of the chosen action** minus the **expected feature under the current policy** — a contrast between what happened and what was expected.

**Intuition:** If you took action `Jump` and the features of `Jump` are $[1, 0, 0]$, and the average feature under the policy is $[0.3, 0.4, 0.3]$, then the score function is $[0.7, -0.4, -0.3]$ — a direction that increases the distinctiveness of the `Jump` action.

---

## 6. Worked Calculation: REINFORCE Step by Step

**Environment:** A simple 3-step game. The agent has two actions at each step: `Left (L)` and `Right (R)`. The episode always has exactly 3 steps. Terminal reward only: $+10$ for reaching the Goal, $-5$ for falling.

**Policy representation:** Linear softmax. For simplicity, the policy has one parameter $\theta$ per action-state pair. We use a single-state example (same state at each step for clarity):

$$h(s, L;\, \theta) = \theta_L, \quad h(s, R;\, \theta) = \theta_R$$

$$\pi(L \mid s;\, \boldsymbol{\theta}) = \frac{e^{\theta_L}}{e^{\theta_L} + e^{\theta_R}}, \qquad \pi(R \mid s;\, \boldsymbol{\theta}) = \frac{e^{\theta_R}}{e^{\theta_L} + e^{\theta_R}}$$

**Initial parameters:** $\theta_L = 0.5$, $\theta_R = 1.0$

**Discount factor:** $\gamma = 1.0$ (no discounting for this example), **Learning rate:** $\alpha = 0.1$

---

### Step 1 — Compute Current Policy Probabilities

$$e^{\theta_L} = e^{0.5} \approx 1.649, \quad e^{\theta_R} = e^{1.0} \approx 2.718$$

$$\text{Total} = 1.649 + 2.718 = 4.367$$

$$\pi(L \mid s) = \frac{1.649}{4.367} \approx \mathbf{0.378} \quad (37.8\%)$$

$$\pi(R \mid s) = \frac{2.718}{4.367} \approx \mathbf{0.622} \quad (62.2\%)$$

The agent currently favors `Right` but hasn't fully committed.

---

### Step 2 — Generate an Episode

The agent plays one episode. By sampling from the policy, the episode turns out to be:

$$s_0 \xrightarrow{L} s_1 \xrightarrow{L} s_2 \xrightarrow{R} \text{Fall! } r = -5$$

Actions taken: $(a_0 = L,\; a_1 = L,\; a_2 = R)$

Rewards: $r_1 = 0,\; r_2 = 0,\; r_3 = -5$ (only terminal reward)

---

### Step 3 — Compute Returns $G_t$ (Working Backwards)

$$G_2 = r_3 = -5$$
$$G_1 = r_2 + \gamma G_2 = 0 + 1.0 \times (-5) = -5$$
$$G_0 = r_1 + \gamma G_1 = 0 + 1.0 \times (-5) = -5$$

All steps share the same return $G_t = -5$ because $\gamma = 1$ and rewards only come at the end.

---

### Step 4 — Compute Score Functions $\nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s;\, \boldsymbol{\theta})$

For a softmax policy, the gradient of $\log \pi(a \mid s)$ with respect to $\theta_a$ is:

$$\frac{\partial \log \pi(a \mid s)}{\partial \theta_a} = 1 - \pi(a \mid s) \quad \text{(gradient for the chosen action)}$$

$$\frac{\partial \log \pi(a \mid s)}{\partial \theta_{a'}} = -\pi(a' \mid s) \quad \text{(gradient for all other actions } a' \neq a\text{)}$$

**Intuition:** If action $L$ was chosen, we want to increase $\theta_L$ (by $1 - \pi(L)$) and simultaneously decrease $\theta_R$ (by $\pi(R)$), so the policy mass shifts toward $L$.

For step $t=0$: $a_0 = L$

$$\frac{\partial \log \pi(L)}{\partial \theta_L} = 1 - \pi(L) = 1 - 0.378 = +0.622$$
$$\frac{\partial \log \pi(L)}{\partial \theta_R} = -\pi(R) = -0.622$$

For steps $t=1$ (also $a = L$) and $t=2$ ($a = R$):

| Step | Action | $\partial \log\pi / \partial \theta_L$ | $\partial \log\pi / \partial \theta_R$ |
|------|--------|---------------------------------------|---------------------------------------|
| 0 | L | $+0.622$ | $-0.622$ |
| 1 | L | $+0.622$ | $-0.622$ |
| 2 | R | $-0.378$ | $+0.378$ |

---

### Step 5 — Compute Policy Gradient Updates

The REINFORCE update for each parameter:

$$\Delta\theta_L = \alpha \sum_{t=0}^{2} G_t \cdot \frac{\partial \log \pi(a_t)}{\partial \theta_L}$$

$$= 0.1 \times \Bigl[(-5)(+0.622) + (-5)(+0.622) + (-5)(-0.378)\Bigr]$$

$$= 0.1 \times \Bigl[-3.11 - 3.11 + 1.89\Bigr]$$

$$= 0.1 \times (-4.33) = \mathbf{-0.433}$$

$$\Delta\theta_R = \alpha \sum_{t=0}^{2} G_t \cdot \frac{\partial \log \pi(a_t)}{\partial \theta_R}$$

$$= 0.1 \times \Bigl[(-5)(-0.622) + (-5)(-0.622) + (-5)(+0.378)\Bigr]$$

$$= 0.1 \times \Bigl[+3.11 + 3.11 - 1.89\Bigr]$$

$$= 0.1 \times (+4.33) = \mathbf{+0.433}$$

---

### Step 6 — Update Parameters

$$\theta_L \leftarrow 0.5 + (-0.433) = \mathbf{0.067}$$

$$\theta_R \leftarrow 1.0 + (+0.433) = \mathbf{1.433}$$

---

### Step 7 — Recompute Policy Probabilities

$$e^{0.067} \approx 1.069, \quad e^{1.433} \approx 4.191 \quad \text{Total} = 5.260$$

$$\pi(L \mid s) = \frac{1.069}{5.260} \approx \mathbf{0.203} \quad (20.3\%)$$

$$\pi(R \mid s) = \frac{4.191}{5.260} \approx \mathbf{0.797} \quad (79.7\%)$$

**Before:** $\pi(L) = 37.8\%$, $\pi(R) = 62.2\%$

**After:** $\pi(L) = 20.3\%$, $\pi(R) = 79.7\%$

**Interpretation:** This episode was bad (fell and got −5). The actions taken were mainly $L, L, R$. The gradient penalized $L$ (took from its probability) and rewarded $R$. But wait — $R$ was also in the final step when the agent fell! So why did $\pi(R)$ increase?

The reason is that the gradient can't attribute blame to individual actions — it treats the entire trajectory as a unit. Step $t=2$ used action $R$, which got a negative return and should *decrease* $\pi(R)$. But steps $t=0$ and $t=1$ used action $L$, and since $L$ failing penalizes $\pi(L)$ strongly, this forces probability mass toward $R$.

This **credit assignment problem** (inability to distinguish which specific action caused the bad outcome) is the core weakness of REINFORCE — and it leads directly to the need for **baselines** and eventually **Actor-Critic** methods.

---

## 7. The Variance Problem

> *REINFORCE is correct in expectation but agonizingly noisy in practice. It's like trying to learn to cook by eating one meal, throwing out your recipe, writing a new one, and repeating — you'll eventually get there, but it takes forever.*

### 7.1 Why REINFORCE Has High Variance

The return $G_t$ can vary enormously between episodes, even when the agent follows exactly the same policy:

- One episode of a game: the agent makes it to level 3 by luck → $G_t = +100$
- Next episode: same policy, bad random outcomes → $G_t = -20$

The policy gradient update is multiplied by $G_t$ directly. This means the gradient estimate fluctuates wildly from episode to episode — some updates push the policy strongly one direction, the next strongly the opposite. Convergence requires many, many episodes to average out this noise.

**A concrete demonstration of why this is bad:**

Consider an agent choosing between two actions. Both have true expected return of $+5$ (equally good). But in practice:
- Action A's returns in 10 episodes: $+1, +3, +8, +5, +7, +2, +9, +4, +6, +5$ → mean = 5.0
- Action B's returns in 10 episodes: $-10, +20, -5, +15, +5, -8, +18, -2, +12, +5$ → mean = 5.0

Action B has much higher variance. REINFORCE would erratically push toward B (when it got +20) then away from B (when it got −10), oscillating wildly, even though both are equally good. This causes slow, unstable learning.

### 7.2 The Baseline: Subtracting the Mean

The core fix is to subtract a **baseline** $b(s_t)$ from the return before computing the gradient update:

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \approx \sum_t \bigl(G_t - b(s_t)\bigr) \cdot \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta})$$

**Crucial property: the baseline does not introduce bias.**

Proof sketch — the additional term $b(s_t) \cdot \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta})$ has zero expectation:

$$\mathbb{E}_{\pi} \bigl[ b(s_t) \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t) \bigr] = b(s_t)\, \mathbb{E}_{\pi} \bigl[ \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t) \bigr] = b(s_t) \times 0 = 0$$

*(The expected score function is always zero: $\mathbb{E}_a[\nabla_{\boldsymbol{\theta}} \log \pi(a \mid s)] = \nabla_{\boldsymbol{\theta}} \mathbb{E}_a[1] = \nabla_{\boldsymbol{\theta}} 1 = 0$)*

So subtracting a baseline **reduces variance without changing the expected gradient direction** — a free lunch!

**Intuitive explanation:** Without a baseline, "good" is defined by $G_t > 0$. If all episodes give positive rewards (imagine a game where you always score between 50 and 100), every action looks "good" and gets reinforced equally — the gradient carries no useful information. Subtracting the average return ($b \approx 75$) refocuses the signal: now episodes scoring 80 look good (+5) and episodes scoring 60 look bad (−15).

### 7.3 The Advantage Function

The optimal baseline is $b(s_t) = V^\pi(s_t)$ — the **expected return from state $s_t$** under the current policy. With this baseline:

$$A_t = G_t - V^\pi(s_t)$$

This is called the **Advantage function** — it answers the question:

> *"How much better (or worse) was the actual return $G_t$ compared to what we expected from this state?"*

| $A_t > 0$ | The action led to a better-than-average outcome → **increase its probability** |
|-----------|--------------------------------------------------------------------------------|
| $A_t = 0$ | Exactly average → no change |
| $A_t < 0$ | Worse than average → **decrease its probability** |

The policy gradient update becomes:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot A_t \cdot \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\, \boldsymbol{\theta})$$

**Intuition:** If your study group's average exam score is 75 and you scored 85, you know your study method (action) worked above average. If you scored 65, it was below average. The advantage function makes this comparison precise, centering the feedback signal around the expected baseline.

**The problem:** To use this baseline, we need $V^\pi(s_t)$ — which brings us right back to learning a value function! This is the key insight that motivates **Actor-Critic** methods (Week 7): learn **both** a policy network (the actor) and a value network (the critic) simultaneously, using the critic's $V(s)$ estimates as the baseline for the actor's gradient updates.

### 7.4 Worked Example: Baseline Reduces Variance

Suppose over 4 episodes, an agent always takes action $R$ in state $s$ and gets returns:

$G^{(1)} = +12,\quad G^{(2)} = +14,\quad G^{(3)} = +11,\quad G^{(4)} = +13$

**REINFORCE gradients** (multiplied by score, proportional to $G_t$):

Variance of $\{12, 14, 11, 13\}$ around mean: $\bar{G} = 12.5$

$$\text{Var} = \frac{(12-12.5)^2 + (14-12.5)^2 + (11-12.5)^2 + (13-12.5)^2}{4} = \frac{0.25+2.25+2.25+0.25}{4} = \mathbf{1.25}$$

**REINFORCE with baseline** $b = \bar{G} = 12.5$: advantages are $\{-0.5, +1.5, -1.5, +0.5\}$

$$\text{Var} = \frac{0.25+2.25+2.25+0.25}{4} = \mathbf{1.25}$$

Wait — the variance is the same here because the baseline is the mean. The *squared values of the gradient updates* scale with the magnitude:

Without baseline: gradients scale with $\{12, 14, 11, 13\}$ → large gradient steps even for average outcomes

With baseline: gradients scale with $\{-0.5, 1.5, -1.5, 0.5\}$ → **small, precise corrections**

The key is not just variance of values but the **magnitude of gradient updates** and their alignment with the true gradient. With a good baseline, most updates are small and informative; without it, updates are large and noisy even when the policy is nearly optimal.

---

## 8. REINFORCE with Baseline: Full Algorithm

The improved algorithm incorporating the advantage estimate:

```
╔══════════════════════════════════════════════════════════════════════╗
║              REINFORCE with Baseline Algorithm                       ║
╠══════════════════════════════════════════════════════════════════════╣
║ Initialize:                                                          ║
║   Policy network π(a|s; θ) with random weights θ                    ║
║   Baseline b (a constant, a running average, or a value network)    ║
║   Learning rate α                                                    ║
║                                                                      ║
║ For each episode:                                                    ║
║   ① Generate episode: s₀,a₀,r₁,s₁,a₁,r₂,…,sT                     ║
║                                                                      ║
║   ② Compute returns Gₜ for each t (backward from T)                 ║
║                                                                      ║
║   ③ Compute baseline estimate b̂ₜ (e.g., average return so far)     ║
║                                                                      ║
║   ④ For each t = 0, 1, …, T−1:                                      ║
║       Advantage: Âₜ = Gₜ − b̂ₜ                                     ║
║       Update: θ ← θ + α · Âₜ · ∇_θ log π(aₜ|sₜ; θ)              ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 9. Continuous Action Spaces: Gaussian Policy in Detail

> *The real payoff of policy gradient methods: the ability to control a robot arm, a car, or a drone with continuous, precise actions — something Q-Learning can never do.*

### 9.1 Parameterizing the Gaussian Policy

For a robot with one joint angle (one-dimensional continuous action $a \in \mathbb{R}$), the network outputs:
- $\mu_{\boldsymbol{\theta}}(s)$: the center of the action distribution
- $\sigma_{\boldsymbol{\theta}}(s)$: the spread (often $\sigma > 0$ enforced via softplus or exp)

The agent samples: $a \sim \mathcal{N}(\mu_{\boldsymbol{\theta}}(s),\; \sigma_{\boldsymbol{\theta}}^2(s))$

The log-probability (needed for the gradient):

$$\log \pi(a \mid s;\, \boldsymbol{\theta}) = -\frac{(a - \mu)^2}{2\sigma^2} - \log\sigma - \frac{1}{2}\log(2\pi)$$

The gradient with respect to $\mu$ and $\log\sigma$ (reparameterized for numerical stability):

$$\frac{\partial \log \pi}{\partial \mu} = \frac{a - \mu}{\sigma^2}, \qquad \frac{\partial \log \pi}{\partial \log\sigma} = \frac{(a - \mu)^2}{\sigma^2} - 1$$

**Intuition for the gradient:**
- $\partial \log\pi / \partial \mu$: proportional to $(a - \mu)$ — if we took an action *above* the mean and it worked well, push $\mu$ upward.
- $\partial \log\pi / \partial \log\sigma$: reduces $\sigma$ when $(a-\mu)^2 < \sigma^2$ (the action was close to the mean — tighten the distribution) and increases $\sigma$ when $(a-\mu)^2 > \sigma^2$ (the action was far from the mean — explore more).

### 9.2 Worked Calculation: Gaussian Policy Update

**Robot arm scenario:** A robot arm must position its joint at angle $45°$ to pick up an object. The network currently outputs $\mu = 40°$, $\sigma = 8°$.

The agent samples $a = 47°$ (a random draw from $\mathcal{N}(40, 64)$) and receives reward $G = +5$ (closer to the goal than average).

**Step 1 — Log-probability:**

$$\log \pi(47 \mid s) = -\frac{(47-40)^2}{2 \times 64} - \log(8) - \frac{1}{2}\log(2\pi)$$

$$= -\frac{49}{128} - 2.079 - 0.919 = -0.383 - 2.079 - 0.919 = -3.381$$

*(We don't need the full log-prob for the update — we only need the gradients.)*

**Step 2 — Gradients of log-policy:**

$$\frac{\partial \log \pi}{\partial \mu} = \frac{a - \mu}{\sigma^2} = \frac{47 - 40}{64} = \frac{7}{64} = \mathbf{+0.109}$$

$$\frac{\partial \log \pi}{\partial \log\sigma} = \frac{(a-\mu)^2}{\sigma^2} - 1 = \frac{49}{64} - 1 = 0.766 - 1 = \mathbf{-0.234}$$

**Step 3 — REINFORCE update** ($\alpha = 0.1$, $G = +5$, no baseline for simplicity):

$$\mu \leftarrow 40 + 0.1 \times 5 \times 0.109 = 40 + 0.0547 = \mathbf{40.05°}$$

$$\log\sigma \leftarrow \log(8) + 0.1 \times 5 \times (-0.234) = 2.079 - 0.117 = 1.962$$

$$\sigma \leftarrow e^{1.962} = \mathbf{7.12°}$$

**Before → After:**
- Mean: $40° \to 40.05°$ (shifted slightly toward $47°$ — the successful action)
- Std: $8° \to 7.12°$ (narrowed slightly — action was within one standard deviation, so the agent becomes more confident)

Over hundreds of episodes, $\mu$ will converge toward the optimal $45°$ and $\sigma$ will shrink, making the agent increasingly precise.

---

## 10. Comparing Policy Gradient Methods to Value-Based Methods

| Dimension | Value-Based (DQN) | Policy Gradient (REINFORCE) |
|-----------|-------------------|----------------------------|
| **What is learned** | $Q(s,a;\boldsymbol{\theta})$ — action values | $\pi(a \mid s;\boldsymbol{\theta})$ — action probabilities |
| **Optimal policy** | Implicit: $\arg\max_a Q$ | Explicit: the network itself |
| **Action space** | Discrete only | Discrete **and** continuous |
| **Policy type** | Deterministic | Stochastic (natural) |
| **Sample efficiency** | Higher (experience replay) | Lower (on-policy, no replay) |
| **Variance** | Low (bootstrapping) | High (MC returns) |
| **Bias** | High (from bootstrapping) | Low (MC is unbiased) |
| **Convergence** | Can diverge with function approx | Guaranteed to local optimum |
| **Best for** | Games with discrete moves | Robotics, continuous control |
| **Key weakness** | Can't handle continuous actions | High variance, slow learning |

**The synthesis:** Neither approach is universally better. Actor-Critic methods (Week 7) marry both — using a policy network (actor) optimized by policy gradients, and a value network (critic) that provides a low-variance baseline. This combination dominates modern deep RL.

---

## 11. Real-World Applications

| Application | Why Policy Gradient | Details |
|-------------|---------------------|---------|
| **Robot locomotion** (OpenAI, DeepMind) | Continuous joint torques | Agents learn to walk, run, and recover from falls — impossible with Q-Learning |
| **Robot manipulation** | Continuous 6-DOF arm control | Grasping, sorting, assembly tasks trained via REINFORCE + baselines |
| **RLHF for LLMs** (ChatGPT, Claude) | Stochastic text generation | Language models output token distributions; policy gradients update the model based on human preference rewards |
| **AlphaGo / MuZero** | Policy network for move selection | Policy gradients refine the probability distribution over moves, guided by MCTS |
| **Drone aerobatics** (ETH Zürich) | Continuous thrust + attitude control | Policy gradient agents learn to fly through racing gates faster than human champions |
| **NLP: Text summarization** | Discrete but non-differentiable metrics | ROUGE/BLEU metrics can't be backpropagated through; REINFORCE allows optimizing them directly |
| **Drug molecule design** | Continuous molecular geometry | Agents propose 3D molecular configurations; reward = predicted binding affinity |
| **Video game character control** | Smooth, human-like motion | Continuous body movement in physics simulations — joints, balance, interaction |

**Special focus — RLHF (Reinforcement Learning from Human Feedback):**

This is why Policy Gradient methods are crucial for modern AI. When training LLMs like GPT-4 or Claude:
1. A language model outputs a token probability distribution — a stochastic policy over words
2. Human raters evaluate the output (reward signal)
3. REINFORCE-style updates adjust the model weights to make preferred outputs more probable

The language model IS a policy: $\pi(\text{token} \mid \text{context};\, \boldsymbol{\theta})$. RLHF uses Policy Gradient methods to align it with human values. This will be studied in depth in Week 10.

---

## 12. Summary

| Concept | Key Idea |
|---------|----------|
| **DQN limitations** | Requires discrete actions, produces deterministic policy, policy is implicit |
| **Policy gradient** | Directly optimize $\pi(a \mid s;\, \boldsymbol{\theta})$ via gradient ascent on $J(\boldsymbol{\theta})$ |
| **Softmax policy** | Converts network scores to action probabilities: $\pi(a) \propto e^{h(s,a;\boldsymbol{\theta})}$ |
| **Gaussian policy** | For continuous actions: network outputs $\mu$ and $\sigma$; sample $a \sim \mathcal{N}(\mu, \sigma^2)$ |
| **Objective $J(\boldsymbol{\theta})$** | Expected total return under the current policy |
| **Log-derivative trick** | $\nabla_{\boldsymbol{\theta}} \mathbb{E}[f(x)] = \mathbb{E}[f(x) \nabla_{\boldsymbol{\theta}} \log p(x;\boldsymbol{\theta})]$ |
| **Policy Gradient Theorem** | $\nabla_{\boldsymbol{\theta}} J = \mathbb{E}[G_t \cdot \nabla_{\boldsymbol{\theta}} \log \pi(a_t \mid s_t;\boldsymbol{\theta})]$ |
| **Model-free** | Transition dynamics $\mathcal{T}$ cancel out — no environment model needed |
| **REINFORCE** | Monte Carlo policy gradient: run episode, compute $G_t$, update policy |
| **High variance** | MC returns fluctuate wildly → slow, unstable learning |
| **Baseline** | Subtract $b(s_t)$ from $G_t$ to reduce variance without introducing bias |
| **Advantage $A_t$** | $A_t = G_t - V(s_t)$ — how much better than average was this outcome? |
| **Score function** | $\nabla_{\boldsymbol{\theta}} \log \pi(a \mid s;\boldsymbol{\theta})$ — direction to increase action $a$'s probability |
| **On-policy** | REINFORCE must use fresh experience from the current policy (no replay buffer) |
| **Actor-Critic preview** | Combining policy gradient (actor) with a learned value function (critic) → Week 7 |

---

## 13. Exercises

### Conceptual Questions

1. **Q-Learning cannot solve the continuous robot arm problem, but REINFORCE can.** Explain precisely which part of the Q-Learning algorithm breaks down when actions are continuous, and how the Gaussian policy parameterization solves this.

2. **Prove informally that a stochastic policy is necessary** in Rock-Paper-Scissors. Suppose you play a fixed deterministic policy (always Scissors). Show that no deterministic policy is a Nash equilibrium. What is the optimal stochastic policy?

3. **The Policy Gradient Theorem shows that $\mathcal{T}$ disappears** from the gradient formula. Why is this remarkable? What does it mean practically for applying policy gradient methods to real-world problems?

4. **REINFORCE is described as "correct in expectation but noisy in practice."** What does "correct in expectation" mean mathematically? Why does high variance in the gradient estimates slow down learning?

5. **Explain why subtracting a baseline does not introduce bias** in the policy gradient estimate. (Hint: consider what $\mathbb{E}_a[\nabla_{\boldsymbol{\theta}} \log \pi(a \mid s;\, \boldsymbol{\theta})]$ equals for any fixed state $s$, and use this to show the baseline term vanishes in expectation.)

---

### Calculation Problems

**Problem 1 — Softmax Policy**

A policy has three actions with scores: $h(s, a_0) = 2.0$, $h(s, a_1) = 0.5$, $h(s, a_2) = -1.0$.

(a) Compute $e^{h(s,a_i)}$ for each action.

(b) Compute the softmax probabilities $\pi(a_i \mid s)$.

(c) The agent takes action $a_0$. Compute the score function $\nabla_{\theta_{a_i}} \log \pi(a_0 \mid s)$ for all $i$.

---

**Problem 2 — REINFORCE Return Calculation**

An episode has the following trajectory:

$$s_0 \xrightarrow{a_0} r_1 = 0 \quad \to \quad s_1 \xrightarrow{a_1} r_2 = +3 \quad \to \quad s_2 \xrightarrow{a_2} r_3 = -2 \quad \to \quad s_3 \; (\text{terminal})$$

Discount factor $\gamma = 0.9$.

(a) Compute $G_2$, $G_1$, and $G_0$.

(b) The policy probabilities at each step were: $\pi(a_0 \mid s_0) = 0.6$, $\pi(a_1 \mid s_1) = 0.3$, $\pi(a_2 \mid s_2) = 0.7$. Compute the log-probability of each chosen action.

(c) Compute the REINFORCE gradient contribution $G_t \cdot \nabla \log \pi(a_t \mid s_t)$ for step $t=0$ and $t=1$.

(Treat $\nabla \log \pi$ as a scalar $= \log \pi(a_t \mid s_t)$ for this simplified version.)

---

**Problem 3 — Softmax Score Function**

For a 2-action softmax policy with parameters $\theta_L = 1.0$, $\theta_R = 2.0$:

(a) Compute $\pi(L \mid s)$ and $\pi(R \mid s)$.

(b) The agent takes action $L$. Compute $\partial \log \pi(L) / \partial \theta_L$ and $\partial \log \pi(L) / \partial \theta_R$.

(c) The episode return is $G = +8$, learning rate $\alpha = 0.05$. Compute the parameter updates $\Delta\theta_L$ and $\Delta\theta_R$.

(d) What are the new parameter values? What happened to $\pi(L)$ — did it increase or decrease, and why does the direction of change make sense?

---

**Problem 4 — Gaussian Policy**

A robot's policy outputs $\mu = 2.0$ (m/s target speed) and $\sigma = 0.5$ (m/s). The agent samples $a = 2.8$ m/s and receives return $G = +4$ (good outcome). Learning rate $\alpha = 0.1$.

(a) Compute $\partial \log \pi(a \mid s) / \partial \mu$.

(b) Compute $\partial \log \pi(a \mid s) / \partial \log\sigma$ (reparameterized).

(c) Compute the REINFORCE updates $\Delta\mu$ and $\Delta\log\sigma$.

(d) Compute the new $\mu$ and new $\sigma$ (remember: $\sigma = e^{\log\sigma}$, starting from $\log\sigma = \log(0.5) \approx -0.693$). Did $\sigma$ increase or decrease? Why does this make sense?

---

### Answer Key

**Problem 1:**

(a) $e^{2.0} = 7.389$, $e^{0.5} = 1.649$, $e^{-1.0} = 0.368$. Sum $= 9.406$

(b) $\pi(a_0) = 7.389/9.406 = \mathbf{0.786}$, $\pi(a_1) = 1.649/9.406 = \mathbf{0.175}$, $\pi(a_2) = 0.368/9.406 = \mathbf{0.039}$

(c) For chosen action $a_0$:
$\partial \log\pi(a_0)/\partial \theta_{a_0} = 1 - \pi(a_0) = 1 - 0.786 = \mathbf{+0.214}$
$\partial \log\pi(a_0)/\partial \theta_{a_1} = -\pi(a_1) = \mathbf{-0.175}$
$\partial \log\pi(a_0)/\partial \theta_{a_2} = -\pi(a_2) = \mathbf{-0.039}$

---

**Problem 2:**

(a) Working backwards:
$G_2 = r_3 = \mathbf{-2}$
$G_1 = r_2 + \gamma G_2 = 3 + 0.9(-2) = 3 - 1.8 = \mathbf{+1.2}$
$G_0 = r_1 + \gamma G_1 = 0 + 0.9(1.2) = \mathbf{+1.08}$

(b) $\log(0.6) = -0.511$, $\log(0.3) = -1.204$, $\log(0.7) = -0.357$

(c) Simplified gradient contribution (using $\log\pi$ as proxy):
$t=0$: $G_0 \times \log\pi(a_0) = 1.08 \times (-0.511) = \mathbf{-0.552}$
$t=1$: $G_1 \times \log\pi(a_1) = 1.2 \times (-1.204) = \mathbf{-1.445}$

Note: both are negative because the chosen actions had probability $< 1$ (log-prob $< 0$). In full REINFORCE with the true score function (not $\log\pi$ itself), the gradient direction is determined correctly by the score function $\nabla\log\pi$, not by the log-prob value.

---

**Problem 3:**

(a) $e^{1.0} = 2.718$, $e^{2.0} = 7.389$. Sum $= 10.107$.
$\pi(L) = 2.718/10.107 = \mathbf{0.269}$, $\pi(R) = 7.389/10.107 = \mathbf{0.731}$

(b) $\partial\log\pi(L)/\partial\theta_L = 1 - 0.269 = \mathbf{+0.731}$
$\partial\log\pi(L)/\partial\theta_R = -\pi(R) = \mathbf{-0.731}$

(c) $\Delta\theta_L = 0.05 \times 8 \times 0.731 = \mathbf{+0.292}$
$\Delta\theta_R = 0.05 \times 8 \times (-0.731) = \mathbf{-0.292}$

(d) New values: $\theta_L = 1.0 + 0.292 = \mathbf{1.292}$, $\theta_R = 2.0 - 0.292 = \mathbf{1.708}$

New probabilities: $e^{1.292} = 3.641$, $e^{1.708} = 5.519$. Sum $= 9.16$.
$\pi(L)_{\text{new}} = 3.641/9.16 = \mathbf{0.397}$

$\pi(L)$ increased from 26.9% to 39.7% ✓ — the episode was good ($G=+8$), action $L$ was taken, so REINFORCE increases its probability. This is exactly the intended behavior: reward good actions by making them more likely.

---

**Problem 4:**

(a) $\partial\log\pi/\partial\mu = (a-\mu)/\sigma^2 = (2.8 - 2.0)/0.25 = 0.8/0.25 = \mathbf{+3.2}$

(b) $\partial\log\pi/\partial\log\sigma = (a-\mu)^2/\sigma^2 - 1 = 0.64/0.25 - 1 = 2.56 - 1 = \mathbf{+1.56}$

(c) $\Delta\mu = 0.1 \times 4 \times 3.2 = \mathbf{+1.28}$
$\Delta\log\sigma = 0.1 \times 4 \times 1.56 = \mathbf{+0.624}$

(d) New $\mu = 2.0 + 1.28 = \mathbf{3.28}$ m/s
New $\log\sigma = -0.693 + 0.624 = -0.069$
New $\sigma = e^{-0.069} = \mathbf{0.933}$ m/s

$\sigma$ **increased** (from 0.5 to 0.933 m/s). This makes sense: the sampled action $a=2.8$ was far from the mean $\mu=2.0$ (1.6 standard deviations away), and it worked well. The agent concludes it should **explore more** — widen its distribution to discover potentially even better actions in that region. If the action had been very close to the mean, $\sigma$ would have decreased (become more confident).

---

*Coming up in Week 7 — Actor-Critic Methods: we combine the best of both worlds — a policy gradient actor with a value-function critic that provides low-variance advantage estimates. This framework underpins PPO, the algorithm used to train ChatGPT and Claude.*
