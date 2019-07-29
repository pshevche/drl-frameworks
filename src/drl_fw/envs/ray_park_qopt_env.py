import gym
from gym.spaces import Dict, Box, Discrete

from drl_fw.envs.park_qopt_env import ParkQOptEnv


class RayParkQOptEnv(gym.Env):
    """
    Wrapper around custom ParkQOptEnv. Since there is no way for a Ray's model to access the environment, 
    it is required to pass action masks in form of a dictionary.
    """

    def __init__(self):
        self.wrapped = ParkQOptEnv()
        self.action_space = self.wrapped.action_space
        self.observation_space = Dict({
            "action_mask": Box(low=0.0, high=1.0, shape=(self.total_nodes*self.total_nodes,)),
            "graph": Box(low=0.0, high=1.0, shape=(self.total_nodes*self.total_nodes,))
        })

    def reset(self):
        orig_obs = self.wrapped.reset()
        return {
            "action_mask": self.wrapped.action_mask,
            "graph": orig_obs
        }

    def step(self, action):
        orig_obs, reward, done, info = self.wrapped.step(action)
        obs = {
            "action_mask": self.wrapped.action_mask,
            "graph": orig_obs
        }
        return obs, reward, done, info

    @property
    def total_nodes(self):
        return self.wrapped.total_nodes

    @property
    def park_env(self):
        return self.wrapped.park_env
