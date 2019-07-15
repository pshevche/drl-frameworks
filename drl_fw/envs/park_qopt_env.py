import gym
from park.envs.query_optimizer import QueryOptEnv
from gym import spaces
import numpy as np

IMBD_TABLES_COUNT = 21


class ParkQOptEnv(gym.Env):
    """
    Wrapper around Park's Query Optimizer environment. 
    Key differences: 
    1. Graph observation space is mapped to a Box-shape which represents a flattened adjacency matrix.
    2. Edge action space is mapped to a Discrete space which will be filtered in framework's agents to select only valid actions.
    """

    def __init__(self):
        self.park_env = QueryOptEnv()

        # observation space is the adjacency matrix
        self.total_nodes = 2 * IMBD_TABLES_COUNT - 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.total_nodes*self.total_nodes,))

        # action space will map possible edges to numbers
        self.action_space = spaces.Discrete(
            self.total_nodes * self.total_nodes)

        # action mask corresponds to observation space
        self.action_mask = np.zeros(self.observation_space.shape)

        # Hack for dopamine: additional wrapper around this env to avoid errors in gym_lib:69
        self.env = self

    def reset(self):
        graph = self.park_env.reset()

        obs = np.zeros(self.observation_space.shape)
        for e in graph.graph.edges:
            obs[e[0] * self.total_nodes + e[1]] = 1
            obs[e[1] * self.total_nodes + e[0]] = 1

        # action mask corresponds to observation space
        self.action_mask = obs

        return obs

    def step(self, action):
        # map number to edge
        edge_act = (action // self.total_nodes, action % self.total_nodes)

        # perform action as usually
        graph, reward, done, info = self.park_env.step(edge_act)

        # get current graph state
        obs = np.zeros(self.observation_space.shape)
        for e in graph.graph.edges:
            obs[e[0] * self.total_nodes + e[1]] = 1
            obs[e[1] * self.total_nodes + e[0]] = 1

        # action mask corresponds to observation space
        self.action_mask = obs

        # Ray does type-checking on info-dict
        if info is None:
            info = {}

        return obs, reward, done, info
