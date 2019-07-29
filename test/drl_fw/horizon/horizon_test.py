import json
import os
import tensorflow as tf
import pytest

from ml.rl.test.gym import run_gym as horizon_runner
from ml.rl.training.rl_dataset import RLDataset
from drl_fw.horizon.components.custom_workflow import (
    custom_train,
    create_park_trainer,
    create_park_predictor)
from drl_fw.tensorboard.custom_tensorboard import Tensorboard

horizon_runner.train = custom_train
horizon_runner.create_trainer = create_park_trainer
horizon_runner.create_predictor = create_park_predictor

PARAMETERS = os.path.abspath(
    './test/drl_fw/horizon/test_data/cartpole_small.json')
FILE_PATH = os.path.abspath(
    './test/drl_fw/dopamine/test_data/checkpoints.json')
EVALUATION_PATH = os.path.abspath(
    './test/drl_fw/dopamine/test_data')


class TestHorizon(object):
    def test_smoke(self):
        """
        Smoke test that runs a small CartPole experiment and fails if any exception during its execution was raised.
        """
        try:
            with open(PARAMETERS) as f:
                params = json.load(f)

            checkpoint_freq = params["run_details"]["checkpoint_after_ts"]
            # train agent
            dataset = RLDataset(FILE_PATH)
            # log experiment info to Tensorboard
            evaluation_file = EVALUATION_PATH
            config_file = PARAMETERS
            experiment_name = config_file[config_file.rfind(
                '/') + 1: config_file.rfind('.json')]
            os.environ["TENSORBOARD_DIR"] = os.path.join(
                evaluation_file, experiment_name)
            average_reward_train, num_episodes_train, average_reward_eval, num_episodes_eval, timesteps_history, trainer, predictor, env = horizon_runner.run_gym(
                params,
                False,
                None,
                -1,
                dataset
            )

            if dataset:
                dataset.save()

        except Exception:
            pytest.fail('Horizon Smoke Test Failed')
