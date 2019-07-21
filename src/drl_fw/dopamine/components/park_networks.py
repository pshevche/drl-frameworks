import gin
import math
import numpy as np
import tensorflow as tf

from dopamine.discrete_domains.gym_lib import _basic_discrete_domain_network

slim = tf.contrib.slim

# shape is an adjacency matrix (0 - no edge, 1 - edge)
QOPT_MIN_VALS = np.array([0.0])
QOPT_MAX_VALS = np.array([1.0])


@gin.configurable
def qopt_dqn_network(num_actions, network_type, state):
    """Wrapper around Dopamine's _basic_discrete_domain_network.

    Args:
      num_actions: int, number of actions.
      network_type: namedtuple, collection of expected values to return.
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    q_values = _basic_discrete_domain_network(
        QOPT_MIN_VALS, QOPT_MAX_VALS, num_actions, state)
    return network_type(q_values)


@gin.configurable
def qopt_rainbow_network(num_actions, num_atoms, support, network_type,
                         state):
    """Wrapper around Dopamine's _basic_discrete_domain_network, adjusted for Rainbow DQN agent.

    Args:
        num_actions: int, number of actions.
        num_atoms: int, the number of buckets of the value function distribution.
        support: tf.linspace, the support of the Q-value distribution.
        network_type: `namedtuple`, collection of expected values to return.
        state: `tf.Tensor`, contains the agent's current state.

    Returns:
        net: _network_type object containing the tensors output by the network.
    """
    net = _basic_discrete_domain_network(
        QOPT_MIN_VALS, QOPT_MAX_VALS, num_actions, state,
        num_atoms=num_atoms)
    logits = tf.reshape(net, [-1, num_actions, num_atoms])
    probabilities = tf.contrib.layers.softmax(logits)
    q_values = tf.reduce_sum(support * probabilities, axis=2)
    return network_type(q_values, logits, probabilities)


@gin.configurable
def qopt_iq_network(num_actions, quantile_embedding_dim,
                    network_type, state, num_quantiles):
    """Wrapper around Dopamine's _basic_discrete_domain_network, adjusted for Rainbow DQN agent.

    Args:
        num_actions: int, number of actions.
        num_atoms: int, the number of buckets of the value function distribution.
        support: tf.linspace, the support of the Q-value distribution.
        network_type: `namedtuple`, collection of expected values to return.
        state: `tf.Tensor`, contains the agent's current state.

    Returns:
        net: _network_type object containing the tensors output by the network.
    """
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    state_net = _basic_discrete_domain_network(
        QOPT_MIN_VALS, QOPT_MAX_VALS, num_actions, state)
    state_net_size = state_net.get_shape().as_list()[-1]
    state_net_tiled = tf.tile(state_net, [num_quantiles, 1])

    batch_size = state_net.get_shape().as_list()[0]
    quantiles_shape = [num_quantiles * batch_size, 1]
    quantiles = tf.random_uniform(
        quantiles_shape, minval=0, maxval=1, dtype=tf.float32)

    quantile_net = tf.tile(quantiles, [1, quantile_embedding_dim])
    pi = tf.constant(math.pi)
    quantile_net = tf.cast(tf.range(
        1, quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
    quantile_net = tf.cos(quantile_net)
    quantile_net = slim.fully_connected(quantile_net, state_net_size,
                                        weights_initializer=weights_initializer)
    # Hadamard product.
    net = tf.multiply(state_net_tiled, quantile_net)

    net = slim.fully_connected(
        net, 512, weights_initializer=weights_initializer)
    quantile_values = slim.fully_connected(
        net,
        num_actions,
        activation_fn=None,
        weights_initializer=weights_initializer)

    return network_type(quantile_values=quantile_values, quantiles=quantiles)
