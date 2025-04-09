import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
from collections import deque
import random
from rocket import Rocket

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

class DQN(nn.Module):
    """
    Deep Q-Network for the rocket environment
    """
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class HERBuffer:
    """
    Hindsight Experience Replay Buffer
    """
    def __init__(self, max_size=100000, n_sampled_goal=4, goal_selection_strategy='future'):
        self.max_size = max_size
        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.buffer = deque(maxlen=max_size)
        
    def add(self, obs, action, reward, next_obs, done, goal):
        """
        Add a transition to the buffer
        """
        self.buffer.append((obs, action, reward, next_obs, done, goal))
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer
        """
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        # Unpack the batch
        obs, actions, rewards, next_obs, dones, goals = zip(*batch)
        
        # Convert to tensors
        obs = torch.FloatTensor(np.array(obs)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        goals = torch.FloatTensor(np.array(goals)).to(device)
        
        return obs, actions, rewards, next_obs, dones, goals
    
    def sample_her(self, batch_size):
        """
        Sample a batch of transitions with HER
        """
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        # Unpack the batch
        obs, actions, rewards, next_obs, dones, goals = zip(*batch)
        
        # Create virtual transitions with HER
        her_obs = []
        her_actions = []
        her_rewards = []
        her_next_obs = []
        her_dones = []
        her_goals = []
        
        for i in range(batch_size):
            # Original transition
            her_obs.append(obs[i])
            her_actions.append(actions[i])
            her_rewards.append(rewards[i])
            her_next_obs.append(next_obs[i])
            her_dones.append(dones[i])
            her_goals.append(goals[i])
            
            # Virtual transitions with HER
            for _ in range(self.n_sampled_goal):
                # Select a new goal based on the strategy
                if self.goal_selection_strategy == 'future':
                    # Select a future state as the goal
                    future_idx = np.random.randint(i, len(batch))
                    new_goal = next_obs[future_idx]
                elif self.goal_selection_strategy == 'episode':
                    # Select a random state from the episode as the goal
                    episode_idx = np.random.randint(0, len(batch))
                    new_goal = next_obs[episode_idx]
                elif self.goal_selection_strategy == 'final':
                    # Use the final state of the episode as the goal
                    new_goal = next_obs[-1]
                else:
                    raise ValueError(f"Unknown goal selection strategy: {self.goal_selection_strategy}")
                
                # Compute the reward for the new goal
                new_reward = self.compute_reward(next_obs[i], new_goal)
                
                # Add the virtual transition
                her_obs.append(obs[i])
                her_actions.append(actions[i])
                her_rewards.append(new_reward)
                her_next_obs.append(next_obs[i])
                her_dones.append(dones[i])
                her_goals.append(new_goal)
        
        # Convert to tensors - handle potential shape issues
        # First, ensure all goals have the same shape
        goal_shape = np.array(goals[0]).shape
        for i in range(len(her_goals)):
            if np.array(her_goals[i]).shape != goal_shape:
                # If goal shape doesn't match, reshape or pad as needed
                if isinstance(her_goals[i], np.ndarray):
                    if len(her_goals[i]) < len(goals[0]):
                        # Pad with zeros if needed
                        padded_goal = np.zeros_like(goals[0])
                        padded_goal[:len(her_goals[i])] = her_goals[i]
                        her_goals[i] = padded_goal
                    else:
                        # Truncate if needed
                        her_goals[i] = her_goals[i][:len(goals[0])]
                else:
                    # Convert to numpy array if not already
                    her_goals[i] = np.array(her_goals[i])
        
        # Now convert to tensors
        her_obs = torch.FloatTensor(np.array(her_obs)).to(device)
        her_actions = torch.LongTensor(np.array(her_actions)).to(device)
        her_rewards = torch.FloatTensor(np.array(her_rewards)).to(device)
        her_next_obs = torch.FloatTensor(np.array(her_next_obs)).to(device)
        her_dones = torch.FloatTensor(np.array(her_dones)).to(device)
        her_goals = torch.FloatTensor(np.array(her_goals)).to(device)
        
        return her_obs, her_actions, her_rewards, her_next_obs, her_dones, her_goals
    
    def compute_reward(self, achieved_goal, desired_goal):
        """
        Compute the reward for a given achieved goal and desired goal
        """
        # For the rocket environment, we use the negative distance as the reward
        return -np.linalg.norm(achieved_goal - desired_goal)
    
    def __len__(self):
        return len(self.buffer)

class HERAgent:
    """
    DQN agent with Hindsight Experience Replay
    """
    def __init__(self, state_dim, action_dim, hidden_size=128, learning_rate=1e-4, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=100000, n_sampled_goal=4, goal_selection_strategy='future',
                 batch_size=64, target_update=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Calculate the input dimension for the DQN (state + goal)
        # For the rocket environment, the goal is the target position (x, y)
        self.goal_dim = 2  # target_x, target_y
        self.input_dim = state_dim + self.goal_dim
        
        # Create the networks
        self.policy_net = DQN(self.input_dim, action_dim, hidden_size).to(device)
        self.target_net = DQN(self.input_dim, action_dim, hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Create the optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Create the replay buffer
        self.replay_buffer = HERBuffer(buffer_size, n_sampled_goal, goal_selection_strategy)
        
        # Initialize the step counter
        self.steps = 0
        
    def select_action(self, state, goal, deterministic=False):
        """
        Select an action using epsilon-greedy policy
        """
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            # Concatenate state and goal
            state_goal = np.concatenate([state, goal])
            state_goal = torch.FloatTensor(state_goal).unsqueeze(0).to(device)
            
            # Get Q-values
            q_values = self.policy_net(state_goal)
            
            # Select the action with the highest Q-value
            return q_values.argmax().item()
    
    def update(self):
        """
        Update the policy network using the replay buffer
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # Return 0 loss if buffer is not full
        
        # Sample a batch of transitions with HER
        obs, actions, rewards, next_obs, dones, goals = self.replay_buffer.sample_her(self.batch_size)
        
        # Print shapes for debugging
        print(f"obs shape: {obs.shape}, goals shape: {goals.shape}")
        
        # Concatenate state and goal
        state_goals = torch.cat([obs, goals], dim=1)
        next_state_goals = torch.cat([next_obs, goals], dim=1)
        
        # Print shapes for debugging
        print(f"state_goals shape: {state_goals.shape}")
        
        # Compute the current Q-values
        current_q_values = self.policy_net(state_goals).gather(1, actions.unsqueeze(1))
        
        # Compute the next Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_goals).max(1)[0]
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute the loss
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update the target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path):
        """
        Save the model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path):
        """
        Load the model
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

def train_her(env, agent, max_episodes=1000, max_steps=800, save_interval=100, 
              checkpoint_dir='./her_ckpt', render_interval=100):
    """
    Train the HER agent
    """
    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Initialize the episode counter
    episode_id = 0
    
    # Initialize the rewards list
    rewards = []
    
    # Training loop
    while episode_id < max_episodes:
        # Reset the environment
        state = env.reset()
        
        # Initialize the episode variables
        episode_reward = 0
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_dones = []
        
        # Set the goal (for the rocket environment, we use the target position)
        goal = np.array([env.target_x, env.target_y])
        
        # Episode loop
        for step in range(max_steps):
            # Select an action
            action = agent.select_action(state, goal)
            
            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            
            # Store the transition
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_next_states.append(next_state)
            episode_dones.append(done)
            
            # Update the state
            state = next_state
            
            # Update the episode reward
            episode_reward += reward
            
            # Render the environment
            if episode_id % render_interval == 0:
                env.render()
            
            # End the episode if done
            if done:
                break
        
        # Add the transitions to the replay buffer
        for t in range(len(episode_states)):
            # Add the original transition
            agent.replay_buffer.add(
                episode_states[t],
                episode_actions[t],
                episode_rewards[t],
                episode_next_states[t],
                episode_dones[t],
                goal
            )
        
        # Update the agent
        loss = agent.update()
        
        # Store the episode reward
        rewards.append(episode_reward)
        
        # Print the episode information
        print(f'Episode {episode_id}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Loss: {loss:.4f}')
        
        # Save the model
        if episode_id % save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f'her_model_{episode_id}.pt'))
            print(f'Model saved at episode {episode_id}')
        
        # Increment the episode counter
        episode_id += 1
    
    return rewards

if __name__ == '__main__':
    # Create the environment
    task = 'hover'  # 'hover' or 'landing'
    max_steps = 800
    env = Rocket(task=task, max_steps=max_steps)
    
    # Create the agent
    state_dim = env.state_dims
    action_dim = env.action_dims
    
    # Print dimensions for debugging
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Calculate the input dimension for the DQN (state + goal)
    # For the rocket environment, the goal is the target position (x, y)
    goal_dim = 2  # target_x, target_y
    input_dim = state_dim + goal_dim
    
    print(f"Input dimension for DQN: {input_dim}")
    
    agent = HERAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=128,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        n_sampled_goal=4,
        goal_selection_strategy='future',
        batch_size=64,
        target_update=1000
    )
    
    # Train the agent
    rewards = train_her(
        env=env,
        agent=agent,
        max_episodes=10_000,
        max_steps=max_steps,
        save_interval=100,
        checkpoint_dir=f'./{task}_her_ckpt',
        render_interval=100
    )
    
    # Plot the rewards
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('HER Training Rewards')
    plt.savefig(f'./{task}_her_rewards.jpg')
    plt.close() 