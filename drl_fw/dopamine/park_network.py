from dopamine.discrete_domains.gym_lib import _basic_discrete_domain_network
import numpy as np
import gin

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
