from ray.rllib.agents import (ppo, dqn)
from ray.rllib.agents.trainer import Trainer
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.agents import registry as reg


class CustomDQNTrainer(dqn.DQNTrainer):
    """
    Overrides DQNTrainer from RLLib. Main difference: evaluation is performed step-wise, not episode-wise. 
    This is done to bring all evaluated frameworks to the common ground (Dopamine implements step-wise evaluation).
    Episode-wise evaluation makes it hard to decide on iteration's number of steps (consequently, its runtime).
    """
    @override(dqn.DQNTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)

    @override(dqn.DQNTrainer)
    def _evaluate(self):
        ep_count = 0
        rew_sum = 0
        steps = 0
        self.evaluation_ev.restore(self.local_evaluator.save())
        self.evaluation_ev.foreach_policy(lambda p, _: p.set_epsilon(0))
        while steps < self.config["timesteps_per_iteration"]:
            self.evaluation_ev.sample()
            eval_result = collect_metrics(self.evaluation_ev)
            ep_count += 1
            rew_sum += eval_result["episode_reward_mean"]
            steps += eval_result["episode_len_mean"]

        metrics = {
            "episode_reward_mean": rew_sum / ep_count,
            "episodes_this_iter": ep_count
        }

        return {"evaluation": metrics}


class CustomAPPOTrainer(ppo.APPOTrainer):
    """
    Overrides APPOTrainer from RLLib. Main difference: evaluation is performed step-wise, not episode-wise. 
    This is done to bring all evaluated frameworks to the common ground (Dopamine implements step-wise evaluation).
    Episode-wise evaluation makes it hard to decide on iteration's number of steps (consequently, its runtime).
    """
    @override(ppo.APPOTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)

    def set_timesteps_per_iteration(self, timesteps_per_iteration):
        self.timesteps_per_iteration = timesteps_per_iteration

    @override(Trainer)
    def _train(self):
        prev_steps = self.optimizer.num_steps_sampled
        while self.optimizer.num_steps_sampled - prev_steps < self.timesteps_per_iteration:
            self.optimizer.step()
        result = self.collect_metrics()
        result.update(timesteps_this_iter=self.optimizer.num_steps_sampled -
                      prev_steps)
        return result

    def _evaluate(self):
        # TODO: can't figure out how to evaluate this agent

        metrics = {
            "episode_reward_mean": 0.0,
            "episodes_this_iter": 0.0
        }

        return {"evaluation": metrics}


CUSTOM_ALGORITHMS = {
    "DQN": CustomDQNTrainer,
    "APPO": CustomAPPOTrainer
}


def get_agent_class(agent_name):
    """
    Returns the class that corresponds to the agent_name.
    """
    if agent_name in CUSTOM_ALGORITHMS:
        return CUSTOM_ALGORITHMS[agent_name]
    else:
        return reg.get_agent_class(agent_name)
