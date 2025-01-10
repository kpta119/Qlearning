import gymnasium as gym

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

class FrozenAgent:
    def __init__(self, env :gym.Env, learning_rate : float)
        self.env = env
        self.lr = learning_rate
        
