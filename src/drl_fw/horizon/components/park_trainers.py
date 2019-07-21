import numpy as np

from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer
from ml.rl.thrift.core.ttypes import (
    ContinuousActionModelParameters,
    DiscreteActionModelParameters)


class ParkDQNTrainer(DQNTrainer):
    def __init__(
        self,
        q_network,
        q_network_target,
        reward_network,
        parameters: DiscreteActionModelParameters,
        use_gpu=False,
        q_network_cpe=None,
        q_network_cpe_target=None,
        metrics_to_score=None,
        imitator=None,
        env=None,
    ) -> None:
        self.env = env
        DQNTrainer.__init__(
            self,
            q_network,
            q_network_target,
            reward_network,
            parameters,
            use_gpu,
            q_network_cpe,
            q_network_cpe_target,
            metrics_to_score,
            imitator,
        )

    def internal_prediction(self, input):
        """
        Only used by Gym
        """
        unmasked_q_values = DQNTrainer.internal_prediction(self, input)
        masked_q_values = np.zeros(unmasked_q_values.shape)
        # map Q-values of invalid actions to -infinity instead of zero (supports negative Q-values)
        if self.action_mask is not None:
            for i in range(len(self.action_mask)):
                if self.action_mask[i]:
                    masked_q_values[0][i] = unmasked_q_values[0][i]
                else:
                    masked_q_values[0][i] = -np.inf
        else:
            masked_q_values = unmasked_q_values
        return masked_q_values

    @property
    def action_mask(self):
        return self.env.action_mask if hasattr(self.env, 'action_mask') else None
