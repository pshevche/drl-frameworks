#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import time
import tensorflow as tf
import io

from custom_trainer import custom_train
from custom_gym_env import create_custom_env

from Horizon.ml.rl.training.rl_dataset import RLDataset
from Horizon.ml.rl.test.gym import run_gym as horizon_runner
from Horizon.ml.rl.test.utils import write_lists_to_csv
from Horizon.ml.rl.test.gym.open_ai_gym_environment import (
    OpenAIGymEnvironment,
)


USE_CPU = -1
logger = logging.getLogger(__name__)
horizon_runner.train = custom_train


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


def main(args):
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
        help="If set, save test results to file.",
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
    args = parser.parse_args(args)

    if args.log_level not in ("debug", "info", "warning", "error", "critical"):
        raise Exception(
            "Logging level {} not valid level.".format(args.log_level))
    else:
        logger.setLevel(getattr(logging, args.log_level.upper()))

    with open(args.parameters.strip(), "r") as f:
        params = json.load(f)

    dataset = RLDataset(args.file_path) if args.file_path else None
    start_time = time.time()
    avg_train_history, train_episodes, avg_eval_history, eval_episodes, timesteps_history, trainer, predictor = horizon_runner.run_gym(
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
    if args.results_file_path:
        write_lists_to_csv(args.results_file_path,
                           avg_train_history, timesteps_history)
    end_time = time.time()

    # save runtime to file
    evaluation_file = args.evaluation_file_path
    config_file = args.parameters.strip()
    experiment_name = config_file[config_file.rfind(
        '/') + 1: config_file.rfind('.json')]

    runtime_file = os.path.join(evaluation_file, 'runtime', 'runtime.csv')

    f = open(runtime_file, 'a+')
    f.write(experiment_name + ', ' +
            str(end_time - start_time) + '\n')
    f.close()

    tensorboard = Tensorboard(os.path.join(evaluation_file, experiment_name))
    for i in range(0, len(avg_eval_history)):
        tensorboard.log_summary(
            avg_train_history[i], train_episodes[i], avg_eval_history[i], eval_episodes[i], i)
    tensorboard.close()

    # inference testing
    try:
        num_inference_steps = params["run_details"]["num_inference_steps"]
        if num_inference_steps:
            print("--- STARTING HORIZON CARTPOLE inference EXPERIMENT ---")
            env = create_custom_env(params)
            start_time = time.time()
            _ = env.run_n_steps(
                num_inference_steps, predictor, test=True
            )
            end_time = time.time()
            print("--- HORIZON CARTPOLE inference EXPERIMENT COMPLETED ---")
            inference_file = os.path.join(
                evaluation_file, 'runtime', 'inference.csv')
            f = open(inference_file, 'a+')
            f.write(experiment_name + ', ' +
                    str(end_time - start_time) + '\n')
            f.close()
    except KeyError:
        pass

    return avg_eval_history


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = sys.argv
    main(args[1:])
