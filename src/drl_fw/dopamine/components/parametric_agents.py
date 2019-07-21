import gin
import numpy as np
import random
import tensorflow as tf

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.discrete_domains import atari_lib


@gin.configurable
class ParametricDQNAgent(dqn_agent.DQNAgent):
    """
    Parametric version of DQN agent that handles variable-length action spaces. Currently supports only vector-like spaces.
    """

    def __init__(self,
                 sess,
                 num_actions,
                 environment=None,
                 observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=atari_lib.NATURE_DQN_DTYPE,
                 stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
                 network=atari_lib.nature_dqn_network,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=20000,
                 update_period=4,
                 target_update_period=8000,
                 epsilon_fn=dqn_agent.linearly_decaying_epsilon,
                 epsilon_train=0.01,
                 epsilon_eval=0.001,
                 epsilon_decay_period=250000,
                 tf_device='/cpu:*',
                 eval_mode=False,
                 use_staging=True,
                 max_tf_checkpoints_to_keep=4,
                 optimizer=tf.train.RMSPropOptimizer(
                     learning_rate=0.00025,
                     decay=0.95,
                     momentum=0.0,
                     epsilon=0.00001,
                     centered=True),
                 summary_writer=None,
                 summary_writing_frequency=500,
                 allow_partial_reload=False):
        dqn_agent.DQNAgent.__init__(
            self,
            sess=sess,
            num_actions=num_actions,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            stack_size=stack_size,
            network=network,
            gamma=gamma,
            update_horizon=update_horizon,
            min_replay_history=min_replay_history,
            update_period=update_period,
            target_update_period=target_update_period,
            epsilon_fn=epsilon_fn,
            epsilon_train=epsilon_train,
            epsilon_eval=epsilon_eval,
            epsilon_decay_period=epsilon_decay_period,
            tf_device=tf_device,
            use_staging=use_staging,
            optimizer=optimizer,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency)
        self.environment = environment

    def _select_action(self):
        """Select an action from the set of available actions.

        Chooses an action randomly with probability self._calculate_epsilon(), and
        otherwise acts greedily according to the current Q-value estimates.

        Returns:
        int, the selected action.
        """
        if self.eval_mode:
            epsilon = self.epsilon_eval
        else:
            epsilon = self.epsilon_fn(
                self.epsilon_decay_period,
                self.training_steps,
                self.min_replay_history,
                self.epsilon_train)
        if random.random() <= epsilon:
            # Choose a random action with probability epsilon.
            if self.environment:
                possible_actions = np.where(self.environment.action_mask > 0)
                return random.choice(possible_actions[0])
            else:
                return random.randint(0, self.num_actions - 1)
        else:
            if self.environment:
                # Choose the action with highest Q-value at the current state.
                # Implicit quantile agent stores raw Q-values differently than other DQN-based agents.
                if isinstance(self, ParametricImplicitQuantileAgent):
                    unmasked_actions = self._sess.run(
                        self._q_values, {self.state_ph: self.state})
                else:
                    unmasked_actions = self._sess.run(
                        self._net_outputs.q_values, {self.state_ph: self.state})[0]
                # map Q-values of invalid actions to -infinity instead of zero (supports negative Q-values)
                masked_actions = [v if self.environment.action_mask[i]
                                  else -np.inf for i, v in enumerate(unmasked_actions)]
                return np.argmax(masked_actions)
            else:
                return tf.argmax(self._net_outputs.q_values, axis=1)[0]


@gin.configurable
class ParametricRainbowAgent(rainbow_agent.RainbowAgent, ParametricDQNAgent):
    """
    Parametric version of Rainbow DQN agent that handles variable-length action spaces.
    """

    def __init__(self,
                 sess,
                 num_actions,
                 environment=None,
                 observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
                 stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
                 network=atari_lib.rainbow_network,
                 num_atoms=51,
                 vmax=10.,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=20000,
                 update_period=4,
                 target_update_period=8000,
                 epsilon_fn=dqn_agent.linearly_decaying_epsilon,
                 epsilon_train=0.01,
                 epsilon_eval=0.001,
                 epsilon_decay_period=250000,
                 replay_scheme='prioritized',
                 tf_device='/cpu:*',
                 use_staging=True,
                 optimizer=tf.train.AdamOptimizer(
                     learning_rate=0.00025, epsilon=0.0003125),
                 summary_writer=None,
                 summary_writing_frequency=500):
        rainbow_agent.RainbowAgent.__init__(
            self,
            sess=sess,
            num_actions=num_actions,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            stack_size=stack_size,
            network=network,
            num_atoms=num_atoms,
            vmax=vmax,
            gamma=gamma,
            update_horizon=update_horizon,
            min_replay_history=min_replay_history,
            update_period=update_period,
            target_update_period=target_update_period,
            epsilon_fn=epsilon_fn,
            epsilon_train=epsilon_train,
            epsilon_eval=epsilon_eval,
            epsilon_decay_period=epsilon_decay_period,
            replay_scheme=replay_scheme,
            tf_device=tf_device,
            use_staging=use_staging,
            optimizer=optimizer,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency)
        ParametricDQNAgent.__init__(
            self,
            sess=sess,
            num_actions=num_actions,
            environment=environment,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            stack_size=stack_size,
            network=network,
            gamma=gamma,
            update_horizon=update_horizon,
            min_replay_history=min_replay_history,
            update_period=update_period,
            target_update_period=target_update_period,
            epsilon_fn=epsilon_fn,
            epsilon_train=epsilon_train,
            epsilon_eval=epsilon_eval,
            epsilon_decay_period=epsilon_decay_period,
            tf_device=tf_device,
            use_staging=use_staging,
            optimizer=self.optimizer,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency)


@gin.configurable
class ParametricImplicitQuantileAgent(implicit_quantile_agent.ImplicitQuantileAgent, ParametricRainbowAgent):
    """
    Parametric version of Implicit Quantile agent that handles variable-length action spaces.
    """

    def __init__(self,
                 sess,
                 num_actions,
                 environment=None,
                 network=atari_lib.implicit_quantile_network,
                 kappa=1.0,
                 num_tau_samples=32,
                 num_tau_prime_samples=32,
                 num_quantile_samples=32,
                 quantile_embedding_dim=64,
                 double_dqn=False,
                 summary_writer=None,
                 summary_writing_frequency=500):
        implicit_quantile_agent.ImplicitQuantileAgent.__init__(
            self,
            sess=sess,
            num_actions=num_actions,
            network=network,
            kappa=kappa,
            num_tau_samples=num_tau_samples,
            num_tau_prime_samples=num_tau_prime_samples,
            num_quantile_samples=num_quantile_samples,
            quantile_embedding_dim=quantile_embedding_dim,
            double_dqn=double_dqn,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency)
        ParametricRainbowAgent.__init__(
            self,
            sess=sess,
            num_actions=num_actions,
            environment=environment,
            network=network,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency
        )
