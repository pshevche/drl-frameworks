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
from ray.rllib.agents.registry import get_agent_class
from ray.tune.logger import pretty_print

EXAMPLE_USAGE = """
Training example via RLlib CLI:
    ./run_evaluation -f src/ray/experiments/cartpole/ray_dqn_cpu_cp1.yml
"""


class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_train_summary(self, average_reward_train, num_episodes_train, iteration):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Train/NumEpisodes',
                             simple_value=num_episodes_train),
            tf.Summary.Value(tag='Train/AverageReturns',
                             simple_value=average_reward_train),
        ])
        self.writer.add_summary(summary, iteration)
        self.writer.flush()

    def log_eval_summary(self, average_reward_eval, num_episodes_eval, iteration):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Eval/NumEpisodes',
                             simple_value=num_episodes_eval),
            tf.Summary.Value(tag='Eval/AverageReturns',
                             simple_value=average_reward_eval)
        ])
        self.writer.add_summary(summary, iteration)
        self.writer.flush()


def create_parser(parser_creator=None):
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
    agent = agent_class(env=env_name, config=config)
    average_reward_train, train_episodes = [], []
    average_reward_eval, eval_episodes = [], []
    timesteps_history = []

    # train agent
    start_time = time.time()
    for iteration in range(num_iterations):
        result = agent.train()
        timesteps_history.append(result["timesteps_total"])
        average_reward_train.append(result["episode_reward_mean"])
        train_episodes.append(result["episodes_this_iter"])
        try:
            average_reward_eval.append(
                result["evaluation"]["episode_reward_mean"])
            eval_episodes.append(result["evaluation"]["episodes_this_iter"])
        except KeyError:
            pass

        if iteration % checkpoint_freq == 0:
            last_checkpoint = agent.save(checkpoint_dir)

    if checkpoint_at_end:
        last_checkpoint = agent.save(checkpoint_dir)
    end_time = time.time()

    # log results to tensorboard
    tensorboard = Tensorboard(os.path.join(results_dir, experiment_name))
    for i in range(len(average_reward_train)):
        tensorboard.log_train_summary(
            average_reward_train[i], train_episodes[i], i)
    for i in range(len(average_reward_eval)):
        tensorboard.log_train_summary(
            average_reward_eval[i], eval_episodes[i], i)
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
        start_time = time.time()
        steps = 0
        while steps < inference_steps:
            result = agent._evaluate()
            steps += result["evaluation"]["episodes_this_iter"] * \
                result["evaluation"]["episode_len_mean"]
        end_time = time.time()
        inference_file = os.path.join(results_dir, 'runtime', 'inference.csv')
        f = open(inference_file, 'a+')
        f.write(experiment_name + ', ' +
                str(end_time - start_time) + '\n')
        f.close()
        print("--- RAY CARTPOLE INFERENCE EXPERIMENT COMPLETED ---")
    except KeyError:
        pass


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
