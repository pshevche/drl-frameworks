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


EXAMPLE_USAGE = """
Example Usage:
    python src/ray/run_propagation_test.py --checkpoint=src/ray/results/dqn_cartpole_cpu_1/dqn_cartpole_cpu_1/DQN_CartPole-v0_0_2019-05-03_13-40-44mqad943p/src/ray/results/dqn_cartpole_cpu_1/dqn_cartpole_cpu_1/DQN_CartPole-v0_0_2019-05-03_13-40-44mqad943p/checkpoint_1/checkpoint-1 -f=src/ray/experiments/cartpole/dqn_cartpole_cpu_1.yml --steps=500000
"""

# Note: if you use any custom models or envs, register them here first, e.g.:
#
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionCartpole(10))


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


def rollout(agent, env_name, num_steps, out=None, no_render=True):
    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env
        multiagent = agent.local_evaluator.multiagent
        if multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]
            mapping_cache = {}
        policy_map = agent.local_evaluator.policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    if out is not None:
        rollouts = []
    steps = 0
    while steps < (num_steps or steps + 1):
        if out is not None:
            rollout = []
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            if multiagent:
                action_dict = {}
                for agent_id in state.keys():
                    a_state = state[agent_id]
                    if a_state is not None:
                        policy_id = mapping_cache.setdefault(
                            agent_id, policy_agent_mapping(agent_id))
                        p_use_lstm = use_lstm[policy_id]
                        if p_use_lstm:
                            a_action, p_state_init, _ = agent.compute_action(
                                a_state,
                                state=state_init[policy_id],
                                policy_id=policy_id)
                            state_init[policy_id] = p_state_init
                        else:
                            a_action = agent.compute_action(
                                a_state, policy_id=policy_id)
                        action_dict[agent_id] = a_action
                action = action_dict
            else:
                if use_lstm[DEFAULT_POLICY_ID]:
                    action, state_init, _ = agent.compute_action(
                        state, state=state_init)
                else:
                    action = agent.compute_action(state)

            next_state, reward, done, _ = env.step(action)

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if not no_render:
                env.render()
            if out is not None:
                rollout.append([state, action, next_state, reward, done])
            steps += 1
            state = next_state
        if out is not None:
            rollouts.append(rollout)
        print("Episode reward", reward_total)

    if out is not None:
        pickle.dump(rollouts, open(out, "wb"))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
