# Reinforcement Learning


### Topics Covered:
- **Markov Decision Processes (MDPs)**
- **Q-learning**
- **Deep Q-Networks (DQNs)**

### Objectives:
1. Understand the fundamental concepts of reinforcement learning (RL) and how it differs from other types of machine learning.
2. Learn about Markov Decision Processes (MDPs) as a formal framework for RL.
3. Explore Q-learning and Deep Q-Networks (DQNs) as key techniques in RL.
4. Recognize practical applications of reinforcement learning in various domains.

---

## 1. Introduction to Reinforcement Learning

### Definition:
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties and learns to maximize the cumulative reward over time. Unlike supervised learning, RL does not require labeled input/output pairs and instead relies on exploration and exploitation to learn optimal actions.

### Key Concepts:
- **Agent:** The entity that makes decisions.
- **Environment:** The world in which the agent operates and interacts.
- **State:** A representation of the current situation of the environment.
- **Action:** A decision made by the agent that affects the environment.
- **Reward:** Feedback received by the agent from the environment after taking an action.
- **Policy:** A strategy used by the agent to determine the next action based on the current state.

### Applications:
- **Game Playing:** RL is used in AI systems that play games, such as AlphaGo, where the agent learns to play the game through trial and error.
- **Robotics:** RL helps robots learn to perform tasks like walking, grasping objects, or navigating environments.
- **Finance:** RL is used for portfolio management, where an agent learns to make investment decisions to maximize returns.

---

## 2. Markov Decision Processes (MDPs)

### Definition:
A Markov Decision Process (MDP) is a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker (agent). MDPs provide a formalization of the environment in RL.

### Key Components:
- **States (S):** The set of all possible states in the environment.
- **Actions (A):** The set of all possible actions the agent can take.
- **Transition Function (T):** Describes the probability of transitioning from one state to another given a particular action, denoted as $P(s' | s, a)$.
- **Reward Function (R):** Gives the expected immediate reward received after transitioning from state $s$ to state $s'$ due to action $a$, denoted as $R(s, a, s')$.
- **Discount Factor (γ):** A factor between 0 and 1 that represents the importance of future rewards.

### Mathematical Formulation:
An MDP is defined by the tuple $(S, A, T, R, \gamma)$.

The goal in an MDP is to find a policy $\pi(s)$ that maximizes the expected cumulative reward (return):

$$ \text{Return} = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right] $$

Where:
- $s_t$ is the state at time $t$,
- $a_t$ is the action taken at time $t$,
- $\gamma$ is the discount factor.

### Example:
- **Grid World:** An environment where an agent moves on a grid to reach a goal state while avoiding obstacles. The MDP defines the states (grid positions), actions (movements), rewards (reaching the goal), and transitions (movement outcomes).

---

## 3. Q-learning

### Definition:
Q-learning is a model-free reinforcement learning algorithm used to learn the value of actions in a given state. It does not require a model of the environment (i.e., the transition probabilities and rewards) and learns the optimal policy by updating the Q-values based on the agent's experiences.

### Key Concepts:
- **Q-Value (Q-function):** Represents the expected future reward of taking action $a$ in state $s$ and following the optimal policy thereafter. The Q-function is denoted as $Q(s, a)$.
- **Bellman Equation:** The core equation in Q-learning that relates the Q-value of a state-action pair to the rewards and the Q-values of subsequent states.

### Mathematical Formulation:
The Q-value is updated using the Bellman equation:

$$ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] $$

Where:
- $s$ is the current state,
- $a$ is the action taken,
- $s'$ is the next state,
- $a'$ is the next action,
- $α$ is the learning rate,
- $γ$ is the discount factor.

### Example:
- **Game Playing:** In a simple game like Tic-Tac-Toe, Q-learning can be used to learn the optimal strategy by updating the Q-values based on the rewards (winning or losing) and the states visited during the game.

---

## 4. Deep Q-Networks (DQNs)

### Definition:
Deep Q-Networks (DQNs) are an extension of Q-learning that use deep neural networks to approximate the Q-function. This approach enables Q-learning to be applied to environments with high-dimensional state spaces, such as video games or complex simulations.

### Key Concepts:
- **Experience Replay:** A technique where the agent's experiences (state, action, reward, next state) are stored in a replay buffer. The agent learns by sampling random batches of experiences from the buffer, which helps break correlations between consecutive updates and improves stability.
- **Target Network:** A separate neural network used to compute the target Q-values during training. The target network is periodically updated to the weights of the current Q-network, providing more stable training.

### Mathematical Formulation:
The Q-network is a neural network that takes a state $s$ as input and outputs Q-values for all possible actions. The Q-values are updated using the following loss function:

$$ L(\theta) = \mathbb{E}\left[\left( R(s, a) + \gamma \max_{a'} Q_{\text{target}}(s', a'; \theta^-) - Q(s, a; \theta) \right)^2\right] $$

Where:
- $θ$ are the parameters of the Q-network,
- $θ^-$ are the parameters of the target network,
- $Q_{\text{target}}$ is the target Q-value.

### Applications:
- **Atari Game Playing:** DQNs have been used to achieve superhuman performance in playing Atari games by learning directly from raw pixel input.
- **Autonomous Driving:** DQNs can be applied to train agents in driving simulators to navigate roads and avoid obstacles.

---

### Recommended Reading:
- **["Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto](https://ieeexplore.ieee.org/document/712192)**
- **"Deep Reinforcement Learning Hands-On" by Maxim Lapan [Book Repository](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On)**

### Further Exploration:
- **OpenAI Gym:** Explore reinforcement learning environments and practice implementing RL algorithms [here](https://gym.openai.com/).