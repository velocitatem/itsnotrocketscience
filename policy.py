import random
import numpy as np
import torch
import utils
import torch.optim as optim

import torch.nn as nn

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (Aug/2020)
    """

    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):

        x = x * self.scale

        if self.L == 0:
            return x

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1) / self.scale


class MLP(nn.Module):
    """
    Multilayer perception with an embedded positional mapping (if L=0, then no positional mapping)
    """

    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=128, L=7):
        super().__init__()

        self.mapping = PositionalMapping(input_dim=input_dim, L=L)

        k = 1; self.add_module('linear'+str(k),nn.Linear(in_features=self.mapping.output_dim, out_features=hidden_size, bias=True)) # input layer
        for layer in range(hidden_layers):
            k += 1; self.add_module('linear'+str(k),nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True))
        k += 1; self.add_module('linear'+str(k),nn.Linear(in_features=hidden_size, out_features=output_dim, bias=True)) # output layer
        self.layers = [module for module in self.modules() if isinstance(module,nn.Linear)]

        negative_slope = 0.2; self.relu = nn.LeakyReLU(negative_slope)
        
        for child in self.named_children(): print(child)
        print(self.layers)
        

    def forward(self, x): # x: state
        # Convert NumPy array to PyTorch tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
            
        # shape x: 1 x m_token x m_state
        # Reshape input to match expected dimensions
        if len(x.shape) == 3:
            # If input is 3D (batch x token x state), reshape to 2D (batch*token x state)
            batch_size, num_tokens, state_dim = x.shape
            x = x.reshape(batch_size * num_tokens, state_dim)
        elif len(x.shape) == 2:
            # If input is 2D (batch x state), keep as is
            pass
        else:
            # If input is 1D (state), add batch dimension
            x = x.unsqueeze(0) if len(x.shape) == 1 else x
            
        # Apply positional mapping
        x = self.mapping(x)
        
        # Process through layers
        for k in range(len(self.layers)-1):
            x = self.relu(self.layers[k](x))
            
        # Final output layer
        x = self.layers[-1](x)
        
        # Reshape back to original dimensions if needed
        if len(x.shape) == 2 and hasattr(self, 'output_dim') and self.output_dim > 1:
            # For actor network (output_dim > 1), reshape to match input batch/token structure
            if hasattr(self, 'original_shape') and len(self.original_shape) == 3:
                x = x.reshape(self.original_shape[0], self.original_shape[1], -1)
                
        return x

GAMMA = 0.99

class ActorCritic(nn.Module):
    """
    RL policy and update rules
    input_dim = num_inputs
    output_dim = num_actions

    Default configuration:
        hidden_layers=2
        hidden_size=128 
        positional mapping L=7
        learning_rate = 5e-5

    Other configurations to be tried (for simpler problems):
        hidden_layers=0
        hidden_size=256 
        No positional mapping L=0
        learning_rate = 3e-4
    """

    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=128, L=7, learning_rate=5e-5):
        super().__init__()

        self.output_dim = output_dim
        self.actor  = MLP(input_dim=input_dim, output_dim=output_dim, # output = num_actions
                          hidden_layers=hidden_layers, hidden_size=hidden_size, L=L)
        self.critic = MLP(input_dim=input_dim, output_dim=1,          # output = scalar value
                          hidden_layers=hidden_layers, hidden_size=hidden_size, L=L)
        self.softmax = nn.Softmax(dim=-1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        #              = optim.Adam(self.parameters(),    lr=learning_rate)

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        y = self.actor(x)
        probs = self.softmax(y)
        value = self.critic(x)

        return probs, value

    def get_action(self, state, deterministic=False, exploration=0.01):

        # Convert state to tensor and ensure proper shape
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            
        # Store original shape for reshaping later
        self.actor.original_shape = state.shape
        
        # Add batch dimension if needed
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Move to device
        state = state.to(device)
        
        # Get policy distribution and value
        probs, value = self.forward(state)
        
        # Extract probabilities for the first (and only) batch item
        if len(probs.shape) == 3:
            probs = probs[0, :, :]
        probs = probs[0, :]  # Get first batch item
        value = value[0]     # Get first batch item

        if deterministic:
            action_id = np.argmax(np.squeeze(probs.detach().cpu().numpy()))
        else:
            if random.random() < exploration:  # exploration
                action_id = random.randint(0, self.output_dim - 1)
            else:
                action_id = np.random.choice(self.output_dim, p=np.squeeze(probs.detach().cpu().numpy()))

        log_prob = torch.log(probs[action_id] + 1e-9)

        return action_id, log_prob, value

    @staticmethod
    def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=GAMMA, current_state=None):

        # compute Q values
        Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
        Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        # Calculate policy entropy
        # Get the policy distribution from the current state
        with torch.no_grad():
            if current_state is not None:
                # Use the provided current state
                # Convert NumPy array to PyTorch tensor if needed
                if isinstance(current_state, np.ndarray):
                    state = torch.tensor(current_state, dtype=torch.float32).to(device)
                else:
                    state = current_state
            else:
                # Fallback to a dummy state if current_state is not provided
                state = torch.zeros(network.actor.input_dim, dtype=torch.float32).to(device)
            
            # Store original shape for reshaping later
            if hasattr(network.actor, 'original_shape'):
                network.actor.original_shape = state.shape
                
            # Forward pass to get policy distribution
            policy_dist, _ = network.forward(state)
            
            # Ensure policy_dist is properly shaped for entropy calculation
            if len(policy_dist.shape) == 3:
                policy_dist = policy_dist.reshape(-1, policy_dist.shape[-1])
                
            # Calculate entropy: -∑ π(a|s) log π(a|s)
            entropy = -(policy_dist * torch.log(policy_dist + 1e-9)).sum(dim=-1).mean()

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        
        # Add entropy regularization term
        entropy_coef = 0.01  # This is a hyperparameter that controls the strength of entropy regularization
        ac_loss = actor_loss + critic_loss - entropy_coef * entropy

        network.optimizer.zero_grad()
        ac_loss.backward()
        network.optimizer.step()

