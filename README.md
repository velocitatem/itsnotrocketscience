# Actor-Critic Implementation Documentation

## Overview
This repository contains an implementation of the Actor-Critic algorithm with entropy regularization for reinforcement learning tasks, specifically designed for rocket landing control. The implementation uses PyTorch and includes several key components for effective policy learning.

## Environment Configuration

### Rocket Environment (Gymnasium-Compatible)

The environment has been wrapped to be fully compatible with the Gymnasium API, providing a standardized interface for reinforcement learning algorithms.

#### Environment Modes
1. **Hover Mode**
   - Task: Maintain position near a target point
   - Success Criteria: Stay within target radius while maintaining upright position
   - Reward Structure:
     - Distance-based reward (closer to target = higher reward)
     - Pose reward (more upright = higher reward)
     - Bonus rewards for staying within target zones

2. **Landing Mode**
   - Task: Land safely at target location
   - Success Criteria: 
     - Land within target radius
     - Maintain safe landing velocity (< 15 m/s)
     - Keep rocket nearly upright (< 10 degrees)
   - Reward Structure:
     - Distance and pose rewards during flight
     - Large bonus for successful landing
     - Penalties for crashes

#### Observation Space
```python
spaces.Box(
    low=np.array([-np.inf] * 8),  # 8-dimensional continuous space
    high=np.array([np.inf] * 8),
    dtype=np.float32
)
```
Components:
- Position (x, y)
- Velocity (vx, vy)
- Angle (theta)
- Angular velocity (vtheta)
- Nozzle angle (phi)
- Thrust force (f)

#### Action Space
```python
spaces.Discrete(9)  # 9 possible actions
```
Each action represents a combination of:
- Thrust levels (3 options: 0.2g, 1.0g, 2.0g)
- Nozzle angles (3 options: 0°, +30°, -30°)

#### Environment Interface
```python
# Initialize environment
env = RocketEnv(task='hover', max_steps=800, viewport_h=768)

# Reset environment
obs, info = env.reset()

# Step environment
obs, reward, terminated, truncated, info = env.step(action)

# Render environment
env.render(mode="human")
```

#### Additional Features
- Configurable viewport size
- Optional rendering
- Detailed info dictionary with:
  - Distance to target
  - Crash status
  - Landing success status
- Proper resource cleanup
- Seed support for reproducibility

## Architecture

### Core Components

1. **Actor Network**
   - A neural network that outputs action probabilities (policy)
   - Uses a multilayer perceptron (MLP) with positional mapping
   - Outputs a probability distribution over possible actions
   - Implements exploration through both epsilon-greedy and entropy regularization

2. **Critic Network**
   - Estimates the value function for states
   - Uses the same MLP architecture as the actor
   - Outputs a scalar value representing the expected return from a state

3. **Positional Mapping Layer**
   - Transforms input coordinates into a higher dimensional space
   - Helps in approximating high-frequency functions
   - Based on the NeRF paper's positional encoding technique
   - Configurable through the `L` parameter (default: 7)

### Network Architecture Details

```python
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=128, L=7, learning_rate=5e-5):
        # Actor network for policy
        self.actor = MLP(input_dim=input_dim, output_dim=output_dim,
                        hidden_layers=hidden_layers, hidden_size=hidden_size, L=L)
        # Critic network for value estimation
        self.critic = MLP(input_dim=input_dim, output_dim=1,
                         hidden_layers=hidden_layers, hidden_size=hidden_size, L=L)
```

## Algorithm Components

### 1. Policy Learning (Actor)
The actor learns to select actions by maximizing the expected return while considering entropy:

```python
# Policy loss with entropy regularization
actor_loss = (-log_probs * advantage.detach()).mean()
entropy = -(policy_dist * torch.log(policy_dist + 1e-9)).sum(dim=-1).mean()
```

### 2. Value Estimation (Critic)
The critic learns to estimate state values to reduce variance in policy updates:

```python
# Value function loss
critic_loss = 0.5 * advantage.pow(2).mean()
```

### 3. Advantage Calculation
The advantage function helps determine how much better an action is compared to the average:

```python
advantage = Qvals - values
```

## Key Features

### 1. Entropy Regularization
- Encourages exploration by maximizing policy entropy
- Prevents premature convergence to deterministic policies
- Implemented through an entropy term in the loss function
- Controlled by `entropy_coef` (default: 0.01)

### 2. Positional Mapping
- Enhances the network's ability to approximate complex functions
- Uses sinusoidal encoding for input features
- Configurable frequency bands through the `L` parameter

### 3. Exploration Strategies
- Epsilon-greedy exploration (configurable through `exploration` parameter)
- Entropy-based exploration through regularization
- Deterministic action selection option for evaluation

## Training Process

1. **State Processing**
   - Input states are transformed through positional mapping
   - Processed by both actor and critic networks

2. **Action Selection**
   - Actor outputs action probabilities
   - Actions are selected based on exploration strategy
   - Log probabilities are stored for policy updates

3. **Value Estimation**
   - Critic estimates state values
   - Used for advantage calculation
   - Helps in reducing variance of policy updates

4. **Policy Update**
   - Combines policy gradient with entropy regularization
   - Updates both actor and critic networks
   - Uses RMSprop optimizer (configurable learning rate)

## Usage

The implementation can be configured through several parameters:

```python
# Default configuration
hidden_layers = 2
hidden_size = 128
positional_mapping_L = 7
learning_rate = 5e-5
entropy_coef = 0.01

# Alternative configuration for simpler problems
hidden_layers = 0
hidden_size = 256
positional_mapping_L = 0
learning_rate = 3e-4
```

## Technical Considerations

1. **Numerical Stability**
   - Small epsilon (1e-9) added to prevent log(0)
   - Proper tensor shape handling
   - Gradient flow management with `detach()`

2. **Memory Efficiency**
   - Uses `torch.no_grad()`