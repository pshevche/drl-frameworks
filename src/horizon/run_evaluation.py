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

    def log_summary(self, num_episodes_train, average_reward_train, num_episodes_eval, average_reward_eval, iteration):
        summary = tf.Summary(value=[
        tf.Summary.Value(tag='HorizonTrain/NumEpisodes',
                         simple_value=num_episodes_train),
        tf.Summary.Value(tag='HorizonTrain/AverageReturns',
                         simple_value=average_reward_train),
        tf.Summary.Value(tag='HorizonEval/NumEpisodes',
                         simple_value=num_episodes_eval),
        tf.Summary.Value(tag='HorizonEval/AverageReturns',
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
    reward_history, test_history, timestep_history, test_episodes, eval_episodes, trainer, predictor = horizon_runner.run_gym(
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
    if args.evaluation_file_path:
        write_lists_to_csv(args.evaluation_file_path,
                           reward_history, timestep_history)
    if args.results_file_path:
        write_lists_to_csv(args.results_file_path,
                           test_history, timestep_history)

    end_time = time.time()


    # save runtime to file
    #result_file = args.results_file_path
    evaluation_file = args.evaluation_file_path
    base_dir = evaluation_file[:evaluation_file.rfind('/')]
    config_file = args.parameters.strip()
    experiment_name = config_file[config_file.rfind(
        '/') + 1: config_file.rfind('.json')]
    filename = 'runtime_' + experiment_name + '.txt'
    runtime_path = os.path.join(base_dir, filename)

    f = open(runtime_path, 'w+')
    f.write(experiment_name + ' took ' +
            str(end_time - start_time) + ' seconds.')
    f.close()
    
    tensorboard = Tensorboard(base_dir+"/"+experiment_name)
    for i in range(0,len(reward_history)):
      tensorboard.log_summary(test_episodes[i], test_history[i],eval_episodes[i], reward_history[i], timestep_history[i])
    tensorboard.close()


    # propagation testing
    try:
        num_propagation_steps = params["run_details"]["num_propagation_steps"]
        if num_propagation_steps:
            print("--- STARTING HORIZON CARTPOLE PROPAGATION EXPERIMENT ---")
            env = create_custom_env(params)
            start_time = time.time()
            _ = env.run_n_steps(
                num_propagation_steps, predictor, test=True
            )
            end_time = time.time()
            print("--- HORIZON CARTPOLE PROPAGATION EXPERIMENT COMPLETED ---")
            filename = 'propagation_runtime_' + experiment_name + '.txt'
            runtime_path = os.path.join(base_dir, filename)

            f = open(runtime_path, 'w+')
            f.write(experiment_name + ' took ' +
                    str(end_time - start_time) + ' seconds.')
            f.close()
    except KeyError:
        pass

    return reward_history


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = sys.argv
    main(args[1:])
