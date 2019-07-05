import os
import pytest
import ray
import yaml

from drl_fw.ray.custom_trainer import get_agent_class
from drl_fw.tensorboard.custom_tensorboard import Tensorboard

CONFIG_FILE = os.path.abspath(
    './tests/drl_fw/ray/test_data/cartpole_small.yml')


class TestRay(object):
    def test_smoke(self):
        """
        Smoke test that runs a small CartPole experiment and fails if any exception during its execution was raised.
        """
        try:
            # Load configuration file
            with open(CONFIG_FILE) as f:
                experiments = yaml.load(f)

            # extract info about experiment
            experiment_name = list(experiments.keys())[0]
            experiment_info = list(experiments.values())[0]

            agent_name = experiment_info["run"]
            env_name = experiment_info["env"]
            results_dir = experiment_info['local_dir']
            checkpoint_freq = experiment_info["checkpoint_freq"]
            checkpoint_at_end = experiment_info["checkpoint_at_end"]
            checkpoint_dir = os.path.join(results_dir, experiment_name)
            num_iterations = experiment_info["stop"]["training_iteration"]
            config = experiment_info["config"]
            training_steps = experiment_info["agent_training_steps"]
            evaluation_steps = experiment_info["agent_evaluation_steps"]

            # init training agent
            ray.init()
            agent_class = get_agent_class(agent_name)
            agent = agent_class(env=env_name, config=config,
                                training_steps=training_steps, evaluation_steps=evaluation_steps)

            average_reward_train, train_episodes = [], []
            average_reward_eval, eval_episodes = [], []
            timesteps_history = []

            for iteration in range(num_iterations):
                    # train agent
                train_result = agent.train()
                timesteps_history.append(train_result["timesteps_total"])
                average_reward_train.append(
                    train_result["episode_reward_mean"])
                train_episodes.append(train_result["episodes_this_iter"])

                # evaluate agent
                eval_result = agent._evaluate()
                average_reward_eval.append(
                    eval_result["evaluation"]["episode_reward_mean"])
                eval_episodes.append(
                    eval_result["evaluation"]["episodes_this_iter"])

                # checkpoint agent's state
                if checkpoint_freq != 0 and iteration % checkpoint_freq == 0:
                    agent.save(checkpoint_dir)

            # checkpoint agent's last state
            if checkpoint_at_end:
                agent.save(checkpoint_dir)

            # log results to tensorboard
            tensorboard = Tensorboard(
                os.path.join(results_dir, experiment_name))
            for i in range(len(average_reward_eval)):
                tensorboard.log_summary(
                    average_reward_train[i], train_episodes[i], average_reward_eval[i], eval_episodes[i], i)
            tensorboard.close()
        except Exception:
            pytest.fail('Ray Smoke Test Failed')
