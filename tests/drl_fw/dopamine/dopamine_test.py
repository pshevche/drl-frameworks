import os
import tensorflow as tf
import pytest

from dopamine.discrete_domains import run_experiment
from drl_fw.dopamine.components import checkpoint_runner

BASE_DIR = os.path.abspath('./tests/drl_fw/dopamine/test_data')
GIN_FILES = [os.path.abspath(
    './tests/drl_fw/dopamine/test_data/cartpole_small.gin')]


class TestDopamine(object):
    def test_smoke(self):
        """
        Smoke test that runs a small CartPole experiment and fails if any exception during its execution was raised.
        """
        try:
            # init logging
            tf.logging.set_verbosity(tf.logging.ERROR)

            # configure experiment
            run_experiment.load_gin_configs(GIN_FILES, [])
            ginfile = str(GIN_FILES[0])
            experiment_name = ginfile[ginfile.rfind(
                '/') + 1: ginfile.rfind('.gin')]
            log_dir = os.path.join(BASE_DIR, experiment_name)

            # create the agent and run experiment
            runner = checkpoint_runner.create_runner(log_dir)
            runner.run_experiment()
        except Exception:
            pytest.fail('Dopamine Smoke Test Failed')
