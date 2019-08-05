#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import tensorflow as tf
import time
import yaml

import ray
from ray.tune.config_parser import make_parser

from drl_fw.ray.components.custom_trainer import get_agent_class
from drl_fw.tensorboard.custom_tensorboard import Tensorboard

EXAMPLE_USAGE = """
Training example via RLlib CLI:
    ./run_evaluation -f src/ray/experiments/cartpole/ray_dqn_cpu_cp1.yml
"""

logger = logging.getLogger(__name__)


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
    training_steps = experiment_info["agent_training_steps"]
    evaluation_steps = experiment_info["agent_evaluation_steps"]

    # init training agent
    ray.init()
    agent_class = get_agent_class(agent_name)
    agent = agent_class(env=env_name, config=config,
                        training_steps=training_steps, evaluation_steps=evaluation_steps)

    # log results to tensorboard
    tensorboard = Tensorboard(os.path.join(results_dir, experiment_name))

    start_time = time.time()
    for iteration in range(num_iterations):
        logger.info('Starting iteration ' + str(iteration))
        # train agent
        train_result = agent.train()
        average_reward_train = train_result["episode_reward_mean"]
        train_episodes = train_result["episodes_this_iter"]

        # evaluate agent
        eval_result = agent._evaluate()
        average_reward_eval = eval_result["evaluation"]["episode_reward_mean"]
        eval_episodes = eval_result["evaluation"]["episodes_this_iter"]

        # checkpoint agent's state
        if checkpoint_freq != 0 and iteration % checkpoint_freq == 0:
            agent.save(checkpoint_dir)

        # publish tensorboard summary
        tensorboard.log_summary(
            average_reward_train, train_episodes, average_reward_eval, eval_episodes, iteration)

    # checkpoint agent's last state
    if checkpoint_at_end:
        agent.save(checkpoint_dir)
    end_time = time.time()

    # save runtime
    runtime_file = os.path.join(results_dir, 'runtime', 'runtime.csv')
    f = open(runtime_file, 'a+')
    f.write(experiment_name + ', ' +
            str(end_time - start_time) + '\n')
    f.close()

    # inference testing
    try:
        inference_steps = experiment_info["inference_steps"]
        print("--- STARTING RAY INFERENCE EXPERIMENT ---")
        orig_ts_per_iter = agent.evaluation_steps
        agent.evaluation_steps = inference_steps
        start_time = time.time()
        agent._evaluate()
        end_time = time.time()
        inference_file = os.path.join(results_dir, 'runtime', 'inference.csv')

        f = open(inference_file, 'a+')
        f.write(experiment_name + ', ' +
                str(end_time - start_time) + '\n')
        f.close()
        agent.evaluation_steps = orig_ts_per_iter
        print("--- RAY INFERENCE EXPERIMENT COMPLETED ---")
    except KeyError:
        pass


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
