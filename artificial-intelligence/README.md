# Artificial Intelligence — Course Overview

> *How do we build machines that don't just "predict," but "act"?*

This course takes you on a journey from the mathematical foundations of autonomous decision-making to the cutting-edge world of Agentic AI. You will learn how agents learn from their own mistakes, how to fine-tune Large Language Models to use tools, and how to build teams of AI agents that work together to solve complex problems.

---

## Course Structure

| Part | Topic | Weeks |
|------|-------|-------|
| 1 | Reinforcement Learning | 1 – 7 |
| 2 | LLMs & Agentic AI | 8 – 15 |

---

## Part 1: Reinforcement Learning (Weeks 1–7)

### [Week 1 — Foundations of Markov Decision Processes (MDPs)](week_1_rl_and_mdp.md)
**The "Grammar" of decision-making.**
We define the RL universe: agents, environments, and rewards. You'll learn the **Bellman Equation** — the mathematical secret behind calculating the "value" of being in a specific situation.

### [Week 2 — Tabular RL & Dynamic Programming](week_2_tabular_rl_and_dynamic_programming.md)
**Solving the world when you know the rules.**
We explore "perfect world" scenarios. You will learn how to compute the best possible path for an agent when the rules of the environment are fully known using algorithms like **Value Iteration**.

### Week 3 — Model-Free Prediction & Control
**Learning from trial and error.**
Most real-world AI doesn't know the rules beforehand. We cover **Monte Carlo** and **Temporal-Difference (TD) learning** — methods that allow an agent to learn simply by "experiencing" the environment and updating its beliefs.

### Week 4 — Temporal-Difference Control (Q-Learning)
**The "Greedy" approach to success.**
We dive into the most famous algorithm in RL: **Q-Learning**. You'll learn the "Exploration vs. Exploitation" dilemma — how an agent decides between trying something new and sticking to what it knows works.

### Week 5 — Value-Based Deep RL (DQN)
**Giving the agent a "Brain."**
We combine RL with Deep Learning. You'll learn how **Deep Q-Networks (DQN)** allowed AI to beat human champions at Atari games by using neural networks to approximate complex values.

### Week 6 — Policy Gradient Methods
**Learning the "Vibe," not the "Value."**
Instead of calculating how good a state is, we teach the agent to directly learn the best action. This is the foundation for controlling robots and systems with complex, continuous movements.

### Week 7 — Advanced Deep RL & Actor-Critic
**The Gold Standard of modern RL.**
We combine the best of both worlds (Value + Policy) into the **Actor-Critic** framework. We will introduce **PPO** — the engine used to align modern models like ChatGPT to be more helpful and safe.

---

## Part 2: LLMs & Agentic AI (Weeks 8–15)

### Week 8 — LLM Foundations for Agents
**Understanding the "Agent Brain."**
We look under the hood of **Transformers**. You'll learn how modern LLMs transition from simple text-generators to reasoning engines capable of following complex instructions.

### Week 9 — Fine-Tuning for Agency (PEFT & LoRA)
**Specialized training for specific tasks.**
You can't always use a massive model. You'll learn how to "up-skill" smaller models using **Parameter-Efficient Fine-Tuning (LoRA)** so they can handle specialized domains like medicine or law.

### Week 10 — Alignment & Preference Optimization
**Teaching the agent "Human Values."**
How do we make sure an agent is helpful and not harmful? We explore **RLHF** and **DPO** — the techniques used to fine-tune AI based on human preferences and safety.

### Week 11 — Agent Architectures & Reasoning
**Teaching the AI "Chain of Thought."**
An agent shouldn't just blurt out an answer. You'll learn reasoning frameworks like **ReAct**, which teach agents to "Think" before they "Act" and reflect on their own observations.

### Week 12 — Tool Use & Function Calling
**Giving the AI "Hands."**
A brain is useless if it can't touch the world. You'll learn how to teach agents to use external tools like calculators, web browsers, and databases by generating structured code and API calls.

### Week 13 — Agentic Memory & RAG
**Long-term memory and "Self-Retrieval."**
Agents need to remember past interactions. We explore how to build **Retrieval-Augmented Generation (RAG)** systems where the agent decides what it needs to look up in its memory bank to solve a problem.

### Week 14 — Multi-Agent Orchestration
**Building an "AI Company."**
One agent isn't always enough. You'll learn how to build "Crews" or "Swarms" where specialized agents (e.g., a "Coder," a "Researcher," and a "Manager") work together to finish a project.

### Week 15 — Evaluation, Safety, & The Frontier
**Testing and the future of Agency.**
How do we know if our agent is actually good? We look at modern benchmarks, security risks like **prompt injection**, and the future of "Embodied AI" where agents control physical or software-based OS environments.

---

## Bi-Weekly Laboratory Schedule

| Lab | Week | Title | Description |
|-----|------|-------|-------------|
| Lab 1 | Week 2 | Dynamic Programming — The GridWorld Solver | Implement Value Iteration and Policy Iteration from scratch to help an agent navigate a 2D grid with obstacles and rewards. |
| Lab 2 | Week 4 | Q-Learning — The Smart Taxi | Train a tabular Q-Learning agent to pick up and drop off passengers in a city environment, mastering "Exploration vs. Exploitation." |
| Lab 3 | Week 6 | Deep RL (DQN/PPO) — Lunar Lander | Use Deep RL to land a spacecraft on the moon, handling continuous state spaces and physics. |
| Lab 4 | Week 9 | Fine-Tuning (LoRA) — Model Adaptation | Perform Parameter-Efficient Fine-Tuning (LoRA) to teach a small-scale LLM a specific domain language or persona. |
| Lab 5 | Week 11 | The ReAct Agent — Reasoning & Acting | Build an agent that uses a "thought-action-observation" loop to solve multi-step math and logic puzzles by calling external tools. |
| Lab 6 | Week 13 | Agentic RAG — The Knowledgeable Assistant | Build a system where the agent manages its own memory, deciding when to query a vector database to answer complex user questions. |
| Lab 7 | Week 15 | Multi-Agent Coding — The Automated Dev Team | Design a "Crew" of two agents: one that writes Python code and another that tests it, iterating until the code is bug-free. |

---

## Suggested Reading

### Part 1 — Reinforcement Learning

**Core Textbook**
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction.* MIT Press.
  - Focus Chapters: 3 (MDPs), 4 (Dynamic Programming), 6 (TD Learning)

**Key Research Papers**
- **Week 4 (Q-Learning):** Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning.*
- **Week 5 (DQN):** Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature.*
- **Week 6 (Policy Gradients):** Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS.*
- **Week 7 (PPO):** Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv.*

### Part 2 — LLMs & Agentic AI

**The Brain: LLM Architecture & Scaling**
- Vaswani et al. (2017). *Attention is All You Need* — The foundational Transformer paper.
- Brown et al. (2020). *Language Models are Few-Shot Learners* — In-Context Learning for agents.

**Adaptation: Fine-Tuning & Alignment**
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models* — Industry standard for fine-tuning on consumer hardware.
- Ouyang et al. (2022). *Training Language Models to Follow Instructions (InstructGPT)* — Bridges Part 1 & 2 via PPO-based alignment.
- Rafailov et al. (2023). *Direct Preference Optimization* — A stable alternative to RLHF.

**Reasoning: Agentic Loops**
- Wei et al. (2022). *Chain-of-Thought Prompting* — "Thinking step-by-step" improves agent logic.
- Yao et al. (2023). *ReAct: Synergizing Reasoning and Acting* — Core framework for modern agents.
- Schick et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools.*

**Systems: Memory & Multi-Agent Collaboration**
- Lewis et al. (2020). *Retrieval-Augmented Generation* — Foundation of agent long-term memory.
- Park et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior* — Memory architectures in multi-agent environments.
- Wu et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications* — Multi-agent conversation and task decomposition.

**Evaluation & Frontiers**
- Mialon et al. (2023). *GAIA: A Benchmark for General AI Assistants* — Measuring real-world task completion.
- Ng, A. (2024). *Practitioner's Guide to Agentic Workflow* — DeepLearning.AI Series.

---

## Technical Documentation

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Hugging Face PEFT Library](https://huggingface.co/docs/peft/)
- [LangChain / LangGraph Conceptual Guide](https://python.langchain.com/docs/concepts/)
