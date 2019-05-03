#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import time
import yaml

import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.rollout import rollout


EXAMPLE_USAGE = """
Example Usage:
    python src/ray/run_propagation_test.py --checkpoint=src/ray/results/dqn_cartpole_cpu_1/dqn_cartpole_cpu_1/DQN_CartPole-v0_0_2019-05-03_13-40-44mqad943p/src/ray/results/dqn_cartpole_cpu_1/dqn_cartpole_cpu_1/DQN_CartPole-v0_0_2019-05-03_13-40-44mqad943p/checkpoint_1/checkpoint-1 -f=src/ray/experiments/cartpole/dqn_cartpole_cpu_1.yml --steps=500000
"""


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "--checkpoint", type=str, help="Checkpoint from which to roll out.")
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    return parser


def run(args, parser):
    print("--- STARTING RAY CARTPOLE PROPAGATION EXPERIMENT ---\n")
    # Load configuration from file
    with open(args.config_file) as f:
        experiments = yaml.load(f)

    agent_name = list(experiments.values())[0]["run"]
    env_name = list(experiments.values())[0]["env"]
    config = list(experiments.values())[0]["config"]
    results_dir = list(experiments.values())[0]['local_dir']
    experiment_name = list(experiments.keys())[0]

    ray.init()

    cls = get_agent_class(agent_name)
    agent = cls(env=env_name, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    start_time = time.time()
    rollout(agent, env_name, num_steps)
    end_time = time.time()

    filename = 'propagation_runtime_' + experiment_name + '.txt'
    runtime_path = os.path.join(results_dir, filename)

    f = open(runtime_path, 'w+')
    f.write(experiment_name + ' took ' +
            str(end_time - start_time) + ' seconds for propagation.')
    f.close()
    print("--- RAY CARTPOLE PROPAGATION EXPERIMENT COMPLETED ---\n")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
