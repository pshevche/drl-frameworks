#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import time
import tensorflow as tf
import io

from drl_fw.horizon.components.custom_workflow import custom_train
from drl_fw.horizon.components.custom_workflow import create_park_trainer
from drl_fw.horizon.components.custom_workflow import create_park_predictor
from ml.rl.training.rl_dataset import RLDataset
from ml.rl.test.gym import run_gym as horizon_runner
from ml.rl.test.base.utils import write_lists_to_csv
from ml.rl.test.gym.open_ai_gym_environment import (
    OpenAIGymEnvironment,
)
from ml.rl.thrift.core.ttypes import RLParameters

USE_CPU = -1
logger = logging.getLogger(__name__)
horizon_runner.train = custom_train
horizon_runner.create_trainer = create_park_trainer
horizon_runner.create_predictor = create_park_predictor


def create_parser():
    parser = argparse.ArgumentParser(
        description="Train a RL net to play in an OpenAI Gym environment."
    )
    parser.add_argument("-p", "--parameters",
                        help="Path to JSON parameters file.")
    parser.add_argument(
        "-s",
        "--score-bar",
        help="Bar for averaged tests scores.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="If set, will use GPU with specified ID. Otherwise will use CPU.",
        default=USE_CPU,
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help="If set, use logging level specified (debug, info, warning, error, "
        "critical). Else defaults to info.",
        default="info",
    )
    parser.add_argument(
        "-f",
        "--file_path",
        help="If set, save all collected samples as an RLDataset to this file.",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--start_saving_from_score",
        type=int,
        help="If file_path is set, start saving episodes after this score is hit.",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--results_file_path",
        help="If set, save evaluation results to file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--offline_train",
        action="store_true",
        help="If set, collect data using a random policy then train RL offline.",
    )
    parser.add_argument(
        "--path_to_pickled_transitions",
        help="Path to saved transitions to load into replay buffer.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-v",
        "--evaluation_file_path",
        help="If set, save evaluation results to file.",
        type=str,
        default=None,
    )
    return parser


def main(args):
    parser = create_parser()
    args = parser.parse_args(args)

    # load experiment configuration
    with open(args.parameters.strip(), "r") as f:
        params = json.load(f)

    checkpoint_freq = params["run_details"]["checkpoint_after_ts"]
    # train agent
    dataset = RLDataset(
        args.file_path) if checkpoint_freq != 0 and args.file_path else None

    # log experiment info to Tensorboard
    evaluation_file = args.evaluation_file_path
    config_file = args.parameters.strip()
    experiment_name = config_file[config_file.rfind(
        '/') + 1: config_file.rfind('.json')]
    os.environ["TENSORBOARD_DIR"] = os.path.join(
        evaluation_file, experiment_name)

    start_time = time.time()
    average_reward_train, num_episodes_train, average_reward_eval, num_episodes_eval, timesteps_history, trainer, predictor, env = horizon_runner.run_gym(
        params,
        args.offline_train,
        args.score_bar,
        args.gpu_id,
        dataset,
        args.start_saving_from_score,
        args.path_to_pickled_transitions,
    )

    if dataset:
        dataset.save()
    end_time = time.time()

    # save runtime
    runtime_file = os.path.join(evaluation_file, 'runtime', 'runtime.csv')
    f = open(runtime_file, 'a+')
    f.write(experiment_name + ', ' +
            str(end_time - start_time) + '\n')
    f.close()

    # inference testing
    try:
        num_inference_steps = params["run_details"]["num_inference_steps"]
        if num_inference_steps:
            print("--- STARTING HORIZON CARTPOLE INFERENCE EXPERIMENT ---")
            start_time = time.time()
            _ = env.run_n_steps(
                num_inference_steps, predictor, test=True
            )
            end_time = time.time()
            print("--- HORIZON CARTPOLE INFERENCE EXPERIMENT COMPLETED ---")
            inference_file = os.path.join(
                evaluation_file, 'runtime', 'inference.csv')
            f = open(inference_file, 'a+')
            f.write(experiment_name + ', ' +
                    str(end_time - start_time) + '\n')
            f.close()
    except KeyError:
        pass

    return average_reward_eval


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = sys.argv
    main(args[1:])
