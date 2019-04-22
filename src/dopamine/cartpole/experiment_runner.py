import gin
import tensorflow as tf

from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import atari_lib


@gin.configurable
class CheckpointRunner(run_experiment.Runner):
    """
    Object that handles running Dopamine experiments.
    Extends Dopamine runner and allows to control the frequency of checkpointing.
    """

    def __init__(self,
                 base_dir,
                 create_agent_fn,
                 create_environment_fn=atari_lib.create_atari_environment,
                 checkpoint_file_prefix='ckpt',
                 checkpoint_freq=1,
                 logging_file_prefix='log',
                 log_every_n=1,
                 num_iterations=200,
                 training_steps=250000,
                 evaluation_steps=125000,
                 max_steps_per_episode=27000):
        super(CheckpointRunner, self).__init__(base_dir,
                                               create_agent_fn,
                                               create_environment_fn,
                                               checkpoint_file_prefix,
                                               logging_file_prefix,
                                               log_every_n,
                                               num_iterations,
                                               training_steps,
                                               evaluation_steps,
                                               max_steps_per_episode)
        self.checkpoint_freq = checkpoint_freq

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        tf.logging.info('Beginning training...')
        if self._num_iterations <= self._start_iteration:
            tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                               self._num_iterations, self._start_iteration)
            return

        for iteration in range(self._start_iteration, self._num_iterations):
            statistics = self._run_one_iteration(iteration)
            self._log_experiment(iteration, statistics)
            self._checkpoint_experiment(iteration)


def create_runner(base_dir):
    """Creates an experiment CheckpointRunner.

    Args:
        base_dir: str, base directory for hosting all subdirectories.

    Returns:
        runner: A `CheckpointRunner` like object.
    """
    return CheckpointRunner(base_dir, run_experiment.create_agent)
