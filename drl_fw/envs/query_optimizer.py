import gym
from park.envs.query_optimizer.query_optimizer import QueryOptEnv
from gym import spaces
import numpy as np

IMBD_TABLES_COUNT = 21


class ParkQueryOptimizer(gym.Env):
    def __init__(self):
        self.park_env = QueryOptEnv()
        # observation space is the adjacency matrix
        self.total_nodes = 2 * IMBD_TABLES_COUNT - 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.total_nodes*self.total_nodes,))
        # action space will map possible edges to numbers
        self.action_space = spaces.Discrete(1)
        # Hack for dopamine: additional wrapper around this env to avoid errors in gym_lib:69
        self.env = self

    def reset(self):
        graph = self.park_env.reset()

        obs = np.zeros(self.observation_space.shape)
        for e in graph.graph.edges:
            obs[e[0] * self.total_nodes + e[1]] = 1

        # map edges to numbers
        self.action_space.n = graph.number_of_edges()

        return obs

    def step(self, action):
        # map number to edge
        edges = [e for e in self.park_env.graph.graph.edges]
        edge_act = edges[action]

        # perform action as usually
        graph, reward, done, info = self.park_env.step(edge_act)

        # update action space
        self.action_space.n = graph.number_of_edges()

        # get current graph state
        obs = np.zeros(self.observation_space.shape)
        for e in edges:
            obs[e[0] * self.total_nodes + e[1]] = 1

        # Ray does type-checking on info-dict
        if info is None:
            info = {}

        return obs, reward, done, info
