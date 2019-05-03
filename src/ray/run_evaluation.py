#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import yaml

import ray
from ray.tests.cluster_utils import Cluster
from ray.tune.config_parser import make_parser
from ray.tune.trial import resources_to_json
from ray.tune.tune import _make_scheduler, run_experiments

from ray.rllib.train import create_parser

EXAMPLE_USAGE = """
Training example via RLlib CLI:
    rllib train --run DQN --env CartPole-v0

Grid search example via RLlib CLI:
    rllib train -f tuned_examples/cartpole-grid-search-example.yaml

Grid search example via executable:
    ./train.py -f tuned_examples/cartpole-grid-search-example.yaml

Note that -f overrides all other trial-specific command-line options.
"""


def run(args, parser):
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.load(f)
    else:
        # Note: keep this in sync with tune/config_parser.py
        experiments = {
            args.experiment_name: {  # i.e. log to ~/ray_results/default
                "run": args.run,
                "checkpoint_freq": args.checkpoint_freq,
                "local_dir": args.local_dir,
                "resources_per_trial": (
                    args.resources_per_trial and
                    resources_to_json(args.resources_per_trial)),
                "stop": args.stop,
                "config": dict(args.config, env=args.env),
                "restore": args.restore,
                "num_samples": args.num_samples,
                "upload_dir": args.upload_dir,
            }
        }

    for exp in experiments.values():
        if not exp.get("run"):
            parser.error("the following arguments are required: --run")
        if not exp.get("env") and not exp.get("config", {}).get("env"):
            parser.error("the following arguments are required: --env")

    if args.ray_num_nodes:
        cluster = Cluster()
        for _ in range(args.ray_num_nodes):
            cluster.add_node(
                num_cpus=args.ray_num_cpus or 1,
                num_gpus=args.ray_num_gpus or 0,
                object_store_memory=args.ray_object_store_memory,
                redis_max_memory=args.ray_redis_max_memory)
        ray.init(redis_address=cluster.redis_address)
    else:
        ray.init(
            redis_address=args.redis_address,
            object_store_memory=args.ray_object_store_memory,
            redis_max_memory=args.ray_redis_max_memory,
            num_cpus=args.ray_num_cpus,
            num_gpus=args.ray_num_gpus)

    # store runtime
    start_time = time.time()
    run_experiments(
        experiments,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials,
        resume=args.resume)
    end_time = time.time()

    experiment_name = list(experiments.keys())[0]
    results_dir = list(experiments.values())[0]['local_dir']
    filename = 'runtime_' + experiment_name + '.txt'
    runtime_path = os.path.join(results_dir, filename)

    f = open(runtime_path, 'w+')
    f.write(experiment_name + ' took ' +
            str(end_time - start_time) + ' seconds.')
    f.close()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
