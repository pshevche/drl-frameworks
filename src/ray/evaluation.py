import ray
from ray.tune.logger import pretty_print
import ray.rllib.agents.dqn as dqn

import logging
from ray.rllib.evaluation.metrics import collect_metrics


class MyAgent(dqn.DQNAgent):

    def __init__(self, config=None, env=None):
        super(MyAgent, self).__init__(config, env)

    def _evaluate(self):
        logger = logging.getLogger(__name__)
        logger.info("Evaluating current policy for {} episodes".format(
            self.config["evaluation_num_episodes"]))
        self.evaluation_ev.restore(self.local_evaluator.save())
        self.evaluation_ev.foreach_policy(lambda p, _: p.set_epsilon(0))
        for _ in range(self.config["evaluation_num_episodes"]):
            batch = self.evaluation_ev.sample()
            t = batch.count
        metrics = collect_metrics(self.evaluation_ev)
        return {"evaluation": metrics}


def main():
    ray.init()

    config = dqn.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_cpus_per_worker"] = 1
    config["num_workers"] = 1
    config["evaluation_interval"] = 1
    config["evaluation_num_episodes"] = 1
    config["batch_mode"] = "truncate_episodes"
    config["sample_batch_size"] = 1
    trainer = MyAgent(config=config, env="CartPole-v0")

    # Can optionally call trainer.restore(path) to load a checkpoint.

    for _ in range(1):
        # Perform one iteration of training the policy with PPO
        train_result = trainer.train()
        print(pretty_print(train_result))
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)


if __name__ == '__main__':
    main()
