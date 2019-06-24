import gym
from park.envs.query_optimizer.query_optimizer import QueryOptEnv


class ParkQueryOptimizer(gym.Env):
    def __init__(self):
        self.hidden_env = QueryOptEnv()
        self.observation_space = self.hidden_env.observation_space
        self.action_space = self.hidden_env.action_space

    def reset(self):
        self.hidden_env.reset()

    def step(self, action):
        return self.hidden_env.step(action)
