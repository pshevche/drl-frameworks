#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import time
import yaml

import ray
from ray.tune.config_parser import make_parser

from custom_trainer import get_agent_class

EXAMPLE_USAGE = """
Training example via RLlib CLI:
    ./run_evaluation -f src/ray/experiments/cartpole/ray_dqn_cpu_cp1.yml
"""


class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_summary(self, average_reward_train, num_episodes_train, average_reward_eval, num_episodes_eval, iteration):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Train/NumEpisodes',
                             simple_value=num_episodes_train),
            tf.Summary.Value(tag='Train/AverageReturns',
                             simple_value=average_reward_train),
            tf.Summary.Value(tag='Eval/NumEpisodes',
                             simple_value=num_episodes_eval),
            tf.Summary.Value(tag='Eval/AverageReturns',
                             simple_value=average_reward_eval)
        ])
        self.writer.add_summary(summary, iteration)
        self.writer.flush()


def create_parser(parser_creator=None):
    """ Creates argument parser."""
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    return parser


def run(args, parser):
    # Load configuration file
    with open(args.config_file) as f:
        experiments = yaml.load(f)

    # extract info about experiment
    experiment_name = list(experiments.keys())[0]
    experiment_info = list(experiments.values())[0]

    agent_name = experiment_info["run"]
    env_name = experiment_info["env"]
    results_dir = experiment_info['local_dir']
    checkpoint_freq = experiment_info["checkpoint_freq"]
    checkpoint_at_end = experiment_info["checkpoint_at_end"]
    checkpoint_dir = os.path.join(results_dir, experiment_name)
    num_iterations = experiment_info["stop"]["training_iteration"]
    config = experiment_info["config"]

    # init training agent
    ray.init()
    agent_class = get_agent_class(agent_name)
    if agent_name == "DQN":
        agent = agent_class(env=env_name, config=config)
    elif agent_name == "PPO":
        ts_per_iter = experiment_info["agent_timesteps_per_iteration"]
        agent = agent_class(env=env_name, config=config,
                            ts_per_iter=ts_per_iter)

    average_reward_train, train_episodes = [], []
    average_reward_eval, eval_episodes = [], []
    timesteps_history = []

    start_time = time.time()
    for iteration in range(num_iterations):
            # train agent
        train_result = agent.train()
        timesteps_history.append(train_result["timesteps_total"])
        average_reward_train.append(train_result["episode_reward_mean"])
        train_episodes.append(train_result["episodes_this_iter"])

        # evaluate agent
        eval_result = agent._evaluate()
        average_reward_eval.append(
            eval_result["evaluation"]["episode_reward_mean"])
        eval_episodes.append(eval_result["evaluation"]["episodes_this_iter"])

        # checkpoint agent's state
        if checkpoint_freq != 0 and iteration % checkpoint_freq == 0:
            agent.save(checkpoint_dir)

    # checkpoint agent's last state
    if checkpoint_at_end:
        agent.save(checkpoint_dir)
    end_time = time.time()

    # log results to tensorboard
    tensorboard = Tensorboard(os.path.join(results_dir, experiment_name))
    for i in range(len(average_reward_eval)):
        tensorboard.log_summary(
            average_reward_train[i], train_episodes[i], average_reward_eval[i], eval_episodes[i], i)
    tensorboard.close()

    # save runtime
    runtime_file = os.path.join(results_dir, 'runtime', 'runtime.csv')
    f = open(runtime_file, 'a+')
    f.write(experiment_name + ', ' +
            str(end_time - start_time) + '\n')
    f.close()

    # inference testing
    try:
        inference_steps = experiment_info["inference_steps"]
        print("--- STARTING RAY CARTPOLE INFERENCE EXPERIMENT ---")
        orig_ts_per_iter = agent.config["timesteps_per_iteration"]
        agent.config["timesteps_per_iteration"] = inference_steps
        start_time = time.time()
        agent._evaluate()
        end_time = time.time()
        inference_file = os.path.join(results_dir, 'runtime', 'inference.csv')
        f = open(inference_file, 'a+')
        f.write(experiment_name + ', ' +
                str(end_time - start_time) + '\n')
        f.close()
        agent.config["timesteps_per_iteration"] = orig_ts_per_iter
        print("--- RAY CARTPOLE INFERENCE EXPERIMENT COMPLETED ---")
    except KeyError:
        pass


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
