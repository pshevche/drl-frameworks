import gin
import tensorflow as tf

from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import iteration_statistics
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from drl_fw.dopamine.components import parametric_agents


@gin.configurable
def create_parametric_agent(sess, environment, agent_name=None, summary_writer=None,
                            debug_mode=False):
    """Creates an agent.

    Args:
        sess: A `tf.Session` object for running associated ops.
        environment: A gym environment (e.g. Atari 2600).
        agent_name: str, name of the agent to create.
        summary_writer: A Tensorflow summary writer to pass to the agent
        for in-agent training statistics in Tensorboard.
        debug_mode: bool, whether to output Tensorboard summaries. If set to true,
        the agent will output in-episode statistics to Tensorboard. Disabled by
        default as this results in slower training.

    Returns:
        agent: An RL agent.

    Raises:
        ValueError: If `agent_name` is not in supported list.
    """
    assert agent_name is not None
    if not debug_mode:
        summary_writer = None
    if agent_name == 'dqn':
        return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                                  summary_writer=summary_writer)
    elif agent_name == 'rainbow':
        return rainbow_agent.RainbowAgent(
            sess, num_actions=environment.action_space.n,
            summary_writer=summary_writer)
    elif agent_name == 'implicit_quantile':
        return implicit_quantile_agent.ImplicitQuantileAgent(
            sess, num_actions=environment.action_space.n,
            summary_writer=summary_writer)
    elif agent_name == 'parametric_dqn':
        return parametric_agents.ParametricDQNAgent(sess, num_actions=environment.action_space.n, environment=environment.environment, summary_writer=summary_writer)
    elif agent_name == 'parametric_rainbow':
        return parametric_agents.ParametricRainbowAgent(sess, num_actions=environment.action_space.n, environment=environment.environment, summary_writer=summary_writer)
    elif agent_name == 'parametric_implicit_quantile':
        return parametric_agents.ParametricImplicitQuantileAgent(sess, num_actions=environment.action_space.n, environment=environment.environment, summary_writer=summary_writer)
    else:
        raise ValueError('Unknown agent: {}'.format(agent_name))


def create_runner(base_dir):
    """Creates an experiment CheckpointRunner.

    Args:
        base_dir: str, base directory for hosting all subdirectories.

    Returns:
        runner: A `CheckpointRunner` like object.
    """
    return CheckpointRunner(base_dir, create_parametric_agent)


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
                 max_steps_per_episode=27000,
                 inference_steps=None):
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
        self.current_checkpoint = 0
        self.inference_steps = inference_steps

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        tf.logging.info('Beginning training...')
        # init checkpoint number
        self.current_checkpoint = 0
        if self._num_iterations <= self._start_iteration:
            tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                               self._num_iterations, self._start_iteration)
            return

        for iteration in range(self._start_iteration, self._num_iterations):
            statistics = self._run_one_iteration(iteration)
            self._log_experiment(iteration, statistics)
            # checkpoint with given frequency and after last iteration
            if self.checkpoint_freq != 0 and ((iteration + 1) % self.checkpoint_freq == 0 or (iteration + 1) == self._num_iterations):
                self._checkpoint_experiment(iteration)

    def _checkpoint_experiment(self, iteration):
        """Checkpoint experiment data. Overwrite parent method to better handle checkpointing frequency.

        Args:
        iteration: int, iteration number for checkpointing.
        """
        experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                            iteration)
        if experiment_data:
            experiment_data['current_iteration'] = iteration
            experiment_data['logs'] = self._logger.data
            self._checkpointer.save_checkpoint(
                self.current_checkpoint, experiment_data)
            self.current_checkpoint = self.current_checkpoint + 1

    def run_inference_test(self):
        statistics = iteration_statistics.IterationStatistics()
        _ = self._run_one_phase(
            self.inference_steps, statistics, 'eval')
