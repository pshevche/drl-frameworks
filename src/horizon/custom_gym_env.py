import numpy as np
from typing import Union


from Horizon.ml.rl.test.gym.open_ai_gym_environment import (
    OpenAIGymEnvironment,
    EnvType
)
from Horizon.ml.rl.training.rl_predictor_pytorch import RLPredictor
from Horizon.ml.rl.test.gym.gym_predictor import GymPredictor
from Horizon.ml.rl.thrift.core.ttypes import RLParameters


class CustomGymEnvironment(OpenAIGymEnvironment):
    def __init__(
        self,
        gymenv,
        epsilon=0,
        softmax_policy=False,
        gamma=0.99,
        epsilon_decay=1,
        minimum_epsilon=None,
    ):
        """
        Creates a CustomGymEnvironment object.

        :param gymenv: String identifier for desired environment.
        :param epsilon: Fraction of the time the agent should select a random
            action during training.
        :param softmax_policy: 1 to use softmax selection policy or 0 to use
            max q selection.
        :param gamma: Discount rate
        :param epsilon_decay: How much to decay epsilon over each iteration in training.
        :param minimum_epsilon: Lower bound of epsilon.
        """
        super(CustomGymEnvironment, self).__init__(gymenv, epsilon,
                                                   softmax_policy, gamma, epsilon_decay, minimum_epsilon)

    def run_episode(
        self,
        predictor: Union[RLPredictor, GymPredictor, None],
        max_steps=None,
        test=False,
        render=False,
        state_preprocessor=None,
    ):
        """
        Runs an episode of the environment and returns the sum of rewards
        experienced in the episode. For evaluation purposes.

        :param predictor: RLPredictor/GymPredictor object whose policy to
            follow. If set to None, use a random policy.
        :param max_steps: Max number of timesteps before ending episode.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        :param state_preprocessor: State preprocessor to use to preprocess states
        """
        terminal = False
        next_state = self.transform_state(self.env.reset())
        next_action, _ = self.policy(
            predictor, next_state, test, state_preprocessor)
        reward_sum = 0
        discounted_reward_sum = 0
        num_steps_taken = 0

        while not terminal:
            action = next_action
            if render:
                self.env.render()

            if self.action_type == EnvType.DISCRETE_ACTION:
                action_index = np.argmax(action)
                next_state, reward, terminal, _ = self.env.step(action_index)
            else:
                next_state, reward, terminal, _ = self.env.step(action)

            next_state = self.transform_state(next_state)
            num_steps_taken += 1
            next_action, _ = self.policy(
                predictor, next_state, test, state_preprocessor
            )
            reward_sum += reward
            discounted_reward_sum += reward * \
                self.gamma ** (num_steps_taken - 1)

            if max_steps and num_steps_taken >= max_steps:
                break

        self.env.reset()
        return reward_sum, discounted_reward_sum, num_steps_taken

    def run_n_steps(self,
                    steps_count,
                    predictor: Union[RLPredictor, GymPredictor, None],
                    max_steps=None,
                    test=False,
                    render=False,
                    state_preprocessor=None,):
        steps = 0
        reward_sum = 0
        discounted_reward_sum = 0
        while steps < steps_count:
            ep_reward_sum, ep_discounted_reward_sum, ep_steps = self.run_episode(
                predictor, max_steps, test, render, state_preprocessor)
            steps += ep_steps
            reward_sum += ep_reward_sum
        return reward_sum, discounted_reward_sum, steps


def create_custom_env(params):
    """ Returns an instance of CustomGymEnvironment that allows to better control the number of executed steps.
    """
    rl_parameters = RLParameters(**params["rl"])

    env_type = params["env"]
    epsilon = rl_parameters.epsilon

    epsilon_decay, minimum_epsilon = 1.0, None
    if "epsilon_decay" in params["run_details"]:
        epsilon_decay = params["run_details"]["epsilon_decay"]
        del params["run_details"]["epsilon_decay"]
    if "minimum_epsilon" in params["run_details"]:
        minimum_epsilon = params["run_details"]["minimum_epsilon"]
        del params["run_details"]["minimum_epsilon"]

    return CustomGymEnvironment(
        env_type,
        epsilon,
        rl_parameters.softmax_policy,
        rl_parameters.gamma,
        epsilon_decay,
        minimum_epsilon,
    )
