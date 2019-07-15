import argparse
import gym
import numpy as np
import sys

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from drl_fw.envs.query_optimizer import ParkQueryOptimizer


def create_parser():
    parser = argparse.ArgumentParser(
        description='Generate experiences for Park QOpt environment to be used in Horizon and Ray.'
    )
    parser.add_argument('-n', help='Number of experiences to log.', type=int)
    parser.add_argument(
        '-f', '--file-path', help='Path to the folder where experiences should be saved.')
    return parser


def edge_to_discrete(edge, nodes_count):
    return edge[0] * nodes_count + edge[1]


def main(args):
    parser = create_parser()
    args = parser.parse_args(args)

    path = args.file_path
    exp_count = args.n

    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(path)

    # works only for local version of Park's qopt env
    env = gym.make('ParkQueryOptimizer-v0')
    nodes_count = env.total_nodes
    park_env = env.park_env

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. As for now, we'll stick to a no-op preprocessor
    prep = get_preprocessor(env.observation_space)(env.observation_space)

    t = 0
    eps_id = 0
    while t < exp_count:
        obs = env.reset()
        prev_action = np.zeros_like(edge_to_discrete(
            park_env.action_space.sample(), nodes_count))
        prev_reward = 0
        done = False
        while not done:
            action = edge_to_discrete(
                park_env.action_space.sample(), nodes_count)
            new_obs, rew, done, info = env.step(action)
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0,  # put the true action probability here
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos=info,
                new_obs=prep.transform(new_obs))
            obs = new_obs
            prev_action = action
            prev_reward = rew
            t += 1
        eps_id += 1
        writer.write(batch_builder.build_and_reset())


if __name__ == '__main__':
    args = sys.argv
    main(args[1:])
