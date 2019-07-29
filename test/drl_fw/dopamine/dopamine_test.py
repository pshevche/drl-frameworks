import os
import tensorflow as tf
import pytest

from dopamine.discrete_domains import run_experiment
from drl_fw.dopamine.components import checkpoint_runner

BASE_DIR = os.path.abspath('./test/drl_fw/test_results')
PARAMS = [os.path.abspath('./test/drl_fw/dopamine/test_data/cartpole_small.gin'),
          os.path.abspath('./test/drl_fw/dopamine/test_data/qopt_small.gin')
          ]


def test_complete_experiment():
    """
    Smoke test that runs small experiments for CartPole and ParkQOpt environemtns and fails if any exception during its execution was raised.
    """
    try:
        # init logging
        tf.logging.set_verbosity(tf.logging.ERROR)

        # configure experiment
        run_experiment.load_gin_configs(PARAMS, [])
        # create the agent and run experiment
        runner = checkpoint_runner.create_runner(BASE_DIR)
        runner.run_experiment()
    except Exception:
        pytest.fail(
            'Running experiments in Dopamine failed!')
