import gym
from envs.query_optimizer import ParkQueryOptimizer


class Agent(object):
    def __init__(self, state_space, action_space, *args, **kwargs):
        self.state_space = state_space
        self.action_space = action_space

    def get_action(self, obs, prev_reward, prev_done, prev_info):
        act = self.action_space.sample()
        # implement real action logic here
        return act


env = gym.make('ParkQueryOptimizer-v0')

# the run script will start the real system
# and periodically invoke agent.get_action
agent = Agent(env.observation_space, env.action_space)
obs = env.reset()
done = False
iteration = 0

while not done:
    # act = agent.get_action(obs)
    act = env.action_space.sample()
    obs, reward, done, info = env.step(act)
    print("Reward = " + str(reward))
