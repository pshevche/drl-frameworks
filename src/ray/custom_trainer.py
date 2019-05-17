import logging

from ray.rllib.agents import (ppo, dqn)
from ray.rllib.agents.trainer import Trainer
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.agents import registry as reg

logger = logging.getLogger(__name__)


class CustomDQNTrainer(dqn.DQNTrainer):
    """
    Overrides DQNTrainer from RLLib. Main difference: evaluation is performed step-wise, not episode-wise. 
    This is done to bring all evaluated frameworks to the common ground (Dopamine implements step-wise evaluation).
    Episode-wise evaluation makes it hard to decide on iteration's number of steps (consequently, its runtime).
    """
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


class CustomPPOTrainer(ppo.PPOTrainer):
    """
    Overrides PPOTrainer from RLLib. Main difference: evaluation is performed step-wise, not episode-wise. 
    This is done to bring all evaluated frameworks to the common ground (Dopamine implements step-wise evaluation).
    Episode-wise evaluation makes it hard to decide on iteration's number of steps (consequently, its runtime).
    """

    def __init__(self, config=None, env=None, logger_creator=None, ts_per_iter=1000):
        self.timesteps_per_iteration = ts_per_iter
        super().__init__(config, env, logger_creator)

    @override(ppo.PPOTrainer)
    def _train(self):
        prev_steps = self.optimizer.num_steps_sampled
        while self.optimizer.num_steps_sampled - prev_steps < self.timesteps_per_iteration:
            fetches = self.optimizer.step()
            if "kl" in fetches:
                # single-agent
                self.local_evaluator.for_policy(
                    lambda pi: pi.update_kl(fetches["kl"]))
            else:

                def update(pi, pi_id):
                    if pi_id in fetches:
                        pi.update_kl(fetches[pi_id]["kl"])
                    else:
                        logger.debug(
                            "No data for {}, not updating kl".format(pi_id))

                # multi-agent
                self.local_evaluator.foreach_trainable_policy(update)
        res = self.collect_metrics()
        res.update(
            timesteps_this_iter=self.optimizer.num_steps_sampled - prev_steps,
            info=res.get("info", {}))

        # Warn about bad clipping configs
        if self.config["vf_clip_param"] <= 0:
            rew_scale = float("inf")
        elif res["policy_reward_mean"]:
            rew_scale = 0  # punt on handling multiagent case
        else:
            rew_scale = round(
                abs(res["episode_reward_mean"]) / self.config["vf_clip_param"],
                0)
        if rew_scale > 200:
            logger.warning(
                "The magnitude of your environment rewards are more than "
                "{}x the scale of `vf_clip_param`. ".format(rew_scale) +
                "This means that it will take more than "
                "{} iterations for your value ".format(rew_scale) +
                "function to converge. If this is not intended, consider "
                "increasing `vf_clip_param`.")
        return res

    @override(Trainer)
    def _evaluate(self):
        ep_count = 0
        rew_sum = 0
        steps = 0
        self.evaluation_ev.restore(self.local_evaluator.save())
        while steps < self.timesteps_per_iteration:
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


CUSTOM_ALGORITHMS = {
    "DQN": CustomDQNTrainer,
    "PPO": CustomPPOTrainer
}


def get_agent_class(agent_name):
    """
    Returns the class that corresponds to the agent_name.
    """
    if agent_name in CUSTOM_ALGORITHMS:
        return CUSTOM_ALGORITHMS[agent_name]
    else:
        return reg.get_agent_class(agent_name)
