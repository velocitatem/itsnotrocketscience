"""
+--------------------------------------------------------------------------------+
|  WARNING!!!                                                                    |
|  THIS IS JUST AN STUB FILE (TEMPLATE)                                          |
|  PROBABLY ALL LINES SHOULD BE CHANGED OR TOTALLY REPLACED IN ORDER TO GET A    |
|  WORKING FUNCTIONAL VERSION FOR YOUR ASSIGNMENT                                |
+--------------------------------------------------------------------------------+
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rocket import Rocket
import cv2

class RocketEnv(gym.Env):
    """
    A Gymnasium-compatible wrapper for the Rocket environment.
    This environment simulates a rocket's landing or hovering task.
    
    The environment supports two modes:
    - hover: The rocket must maintain position near a target point
    - landing: The rocket must land safely at a target location
    
    Observation Space:
    - 8-dimensional continuous space containing:
        [x, y, vx, vy, theta, vtheta, phi, f]
        where:
        - x, y: position coordinates
        - vx, vy: velocity components
        - theta: angle of the rocket
        - vtheta: angular velocity
        - phi: nozzle angle
        - f: thrust force
    
    Action Space:
    - Discrete space with 9 possible actions
    - Each action represents a combination of thrust and nozzle angle
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, task='hover', max_steps=800, viewport_h=768):
        super(RocketEnv, self).__init__()
        
        # Initialize the underlying Rocket environment
        self.rocket = Rocket(max_steps=max_steps, task=task, viewport_h=viewport_h)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.rocket.action_table))
        
        # Define observation space based on state dimensions
        # Using the flattened state from Rocket class
        obs_low = np.array([-np.inf] * self.rocket.state_dims)
        obs_high = np.array([np.inf] * self.rocket.state_dims)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        
        # Store task type
        self.task = task
        
        # Initialize rendering flag
        self.is_view = True
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset the rocket environment
        obs = self.rocket.reset()
        
        # Convert observation to numpy array if it isn't already
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
            
        return obs, {}  # Return observation and empty info dict as per Gymnasium API
        
    def step(self, action):
        """Execute one time step within the environment."""
        # Execute action in the rocket environment
        self.rocket.step(action)
        
        # Get the current state
        obs = self.rocket.flatten(self.rocket.state)
        
        # Calculate reward
        reward = self.rocket.calculate_reward(self.rocket.state)
        
        # Check if episode is done
        done = self.rocket.check_crash(self.rocket.state)
        
        # Check for successful landing (only in landing task)
        if self.task == 'landing' and self.rocket.check_landing_success(self.rocket.state):
            done = True
            reward += 10.0  # Bonus for successful landing
            
        # Check if max steps reached
        if self.rocket.step_id >= self.rocket.max_steps:
            done = True
            
        # Prepare info dictionary
        info = {
            'distance': self.rocket.distance if hasattr(self.rocket, 'distance') else None,
            'crash': done and not self.rocket.check_landing_success(self.rocket.state),
            'landing_success': self.rocket.check_landing_success(self.rocket.state) if self.task == 'landing' else False
        }
        
        # Convert observation to numpy array if it isn't already
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
            
        return obs, reward, done, False, info  # Return truncated=False as per Gymnasium API
        
    def render(self, mode="human"):
        """Render the environment."""
        if self.is_view:
            self.rocket.render()
            
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'rocket'):
            cv2.destroyAllWindows()
            
    def set_view(self, flag):
        """Enable or disable rendering."""
        self.is_view = flag
        
    def get_state(self):
        """Get the current state of the environment."""
        return self.rocket.state
        
    def get_action_space(self):
        """Get the action space description."""
        return self.rocket.action_table

if __name__ == "__main__":
    print("File __name__ is set to: {}" .format(__name__))
    # Using the gym library to create the environment
    # env = gym.make('your_environment_name-v0')
    from stable_baselines3.common.env_checker import check_env
    env = RocketEnv()
    print('CHECK_ENV','OK' if check_env(env) is None else 'ERROR')
    print(env.observation_space) 
    print(env.action_space) 
    print(type(env).__name__)
