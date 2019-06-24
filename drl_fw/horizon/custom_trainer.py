import logging
import numpy as np
import io

from ml.rl.test.gym import run_gym as horizon_runner
from ml.rl.test.gym.open_ai_gym_environment import (
    EnvType,
    ModelType,
    OpenAIGymEnvironment,
)

logger = logging.getLogger(__name__)


def custom_train(
    c2_device,
    gym_env,
    offline_train,
    replay_buffer,
    model_type,
    trainer,
    predictor,
    test_run_name,
    score_bar,
    num_episodes=301,
    max_steps=None,
    train_every_ts=100,
    train_after_ts=10,
    test_every_ts=100,
    test_after_ts=10,
    num_train_batches=1,
    avg_over_num_episodes=100,
    render=False,
    save_timesteps_to_dataset=None,
    start_saving_from_score=None,
    solved_reward_threshold=None,
    max_episodes_to_run_after_solved=None,
    stop_training_after_solved=False,
    offline_train_epochs=3,
    path_to_pickled_transitions=None,
    bcq_imitator_hyperparams=None,
    timesteps_total=1000,
    checkpoint_after_ts=1,
    num_inference_steps=None,
    avg_over_num_steps=1000,
):
    if offline_train:
        return horizon_runner.train_gym_offline_rl(
            c2_device,
            gym_env,
            replay_buffer,
            model_type,
            trainer,
            predictor,
            test_run_name,
            score_bar,
            max_steps,
            avg_over_num_episodes,
            offline_train_epochs,
            path_to_pickled_transitions,
            bcq_imitator_hyperparams,
        )
    else:
        return custom_train_gym_online_rl(
            c2_device,
            gym_env,
            replay_buffer,
            model_type,
            trainer,
            predictor,
            test_run_name,
            score_bar,
            num_episodes,
            max_steps,
            train_every_ts,
            train_after_ts,
            test_every_ts,
            test_after_ts,
            num_train_batches,
            avg_over_num_episodes,
            render,
            save_timesteps_to_dataset,
            start_saving_from_score,
            solved_reward_threshold,
            max_episodes_to_run_after_solved,
            stop_training_after_solved,
            timesteps_total,
            checkpoint_after_ts,
            avg_over_num_steps
        )


def custom_train_gym_online_rl(
    c2_device,
    gym_env,
    replay_buffer,
    model_type,
    trainer,
    predictor,
    test_run_name,
    score_bar,
    num_episodes,
    max_steps,
    train_every_ts,
    train_after_ts,
    test_every_ts,
    test_after_ts,
    num_train_batches,
    avg_over_num_episodes,
    render,
    save_timesteps_to_dataset,
    start_saving_from_score,
    solved_reward_threshold,
    max_episodes_to_run_after_solved,
    stop_training_after_solved,
    timesteps_total,
    checkpoint_after_ts,
    avg_over_num_steps
):
    """Train off of dynamic set of transitions generated on-policy."""
    ep_i = 0
    ts = 0
    policy_id = 0
    # logging
    average_reward_train, num_episodes_train = [], []
    average_reward_eval, num_episodes_eval = [], []
    timesteps_history = []
    reward_hist = list()
    while ep_i < num_episodes and ts < timesteps_total:
        terminal = False
        next_state = gym_env.transform_state(gym_env.env.reset())
        next_action, next_action_probability = gym_env.policy(
            predictor, next_state, False
        )
        reward_sum = 0
        ep_timesteps = 0

        if model_type == ModelType.CONTINUOUS_ACTION.value:
            trainer.noise.clear()

        while not terminal:
            state = next_state
            action = next_action
            action_probability = next_action_probability

            # Get possible actions
            possible_actions, _ = horizon_runner.get_possible_actions(
                gym_env, model_type, terminal)

            if render:
                gym_env.env.render()

            timeline_format_action, gym_action = horizon_runner._format_action_for_log_and_gym(
                action, gym_env.action_type, model_type
            )
            next_state, reward, terminal, _ = gym_env.env.step(gym_action)
            next_state = gym_env.transform_state(next_state)

            ep_timesteps += 1
            ts += 1
            next_action, next_action_probability = gym_env.policy(
                predictor, next_state, False
            )
            reward_sum += reward

            (possible_actions, possible_actions_mask) = horizon_runner.get_possible_actions(
                gym_env, model_type, False
            )

            # Get possible next actions
            (possible_next_actions, possible_next_actions_mask) = horizon_runner.get_possible_actions(
                gym_env, model_type, terminal
            )

            replay_buffer.insert_into_memory(
                np.float32(state),
                action,
                np.float32(reward),
                np.float32(next_state),
                next_action,
                terminal,
                possible_next_actions,
                possible_next_actions_mask,
                1,
                possible_actions,
                possible_actions_mask,
                policy_id,
            )

            if save_timesteps_to_dataset and (
                    ts % checkpoint_after_ts == 0 or ts == timesteps_total):
                save_timesteps_to_dataset.insert(
                    mdp_id=ep_i,
                    sequence_number=ep_timesteps - 1,
                    state=state,
                    action=action,
                    timeline_format_action=timeline_format_action,
                    action_probability=action_probability,
                    reward=reward,
                    next_state=next_state,
                    next_action=next_action,
                    terminal=terminal,
                    possible_next_actions=possible_next_actions,
                    possible_next_actions_mask=possible_next_actions_mask,
                    time_diff=1,
                    possible_actions=possible_actions,
                    possible_actions_mask=possible_actions_mask,
                    policy_id=policy_id,
                )

            # Training loop
            if (
                ts % train_every_ts == 0
                and ts > train_after_ts
                and len(replay_buffer.replay_memory) >= trainer.minibatch_size
            ):
                for _ in range(num_train_batches):
                    samples = replay_buffer.sample_memories(
                        trainer.minibatch_size, model_type
                    )
                    samples.set_type(trainer.dtype)
                    trainer.train(samples)
                    # Every time we train, the policy changes
                    policy_id += 1

            # Evaluation loop
            if ts % test_every_ts == 0 and ts > test_after_ts:
                avg_ep_count, avg_rewards = gym_env.run_n_steps(
                    avg_over_num_steps, predictor, test=True
                )

                # save Tensorboard statistics
                timesteps_history.append(ts)
                avg_train_reward = sum(reward_hist) / len(reward_hist)
                average_reward_train.append(avg_train_reward)
                num_episodes_train.append(len(reward_hist))
                average_reward_eval.append(avg_rewards)
                num_episodes_eval.append(avg_ep_count)

                logger.info(
                    "Achieved an average reward score of {} over {} evaluations."
                    " Total episodes: {}, total timesteps: {}.".format(
                        avg_rewards, avg_ep_count, ep_i + 1, ts
                    )
                )
                logger.info(
                    "Achieved an average reward score of {} during {} training episodes."
                    " Total episodes: {}, total timesteps: {}.".format(
                        avg_train_reward, len(
                            reward_hist), ep_i + 1, ts
                    )
                )
                reward_hist.clear()
                if score_bar is not None and avg_rewards > score_bar:
                    logger.info(
                        "Avg. reward history during evaluation for {}: {}".format(
                            test_run_name, average_reward_eval
                        )
                    )
                    logger.info(
                        "Avg. reward history during training for {}: {}".format(
                            test_run_name, average_reward_train
                        )
                    )
                    return average_reward_train, num_episodes_train, average_reward_eval, num_episodes_eval, timesteps_history, trainer, predictor, gym_env

            if max_steps and ep_timesteps >= max_steps:
                break
        reward_hist.append(reward_sum)

        # Always eval on last episode if previous eval loop didn't return.
        if ep_i == num_episodes - 1:
            avg_ep_count, avg_rewards = gym_env.run_n_steps(
                avg_over_num_steps, predictor, test=True
            )

            # save Tensorboard statistics
            timesteps_history.append(ts)
            avg_train_reward = sum(reward_hist) / len(reward_hist)
            average_reward_train.append(avg_train_reward)
            num_episodes_train.append(len(reward_hist))
            average_reward_eval.append(avg_rewards)
            num_episodes_eval.append(avg_ep_count)

            logger.info(
                "Achieved an average reward score of {} over {} evaluations."
                " Total episodes: {}, total timesteps: {}.".format(
                    avg_rewards, avg_ep_count, ep_i + 1, ts
                )
            )

            logger.info(
                "Achieved an average reward score of {} during {} training episodes."
                " Total episodes: {}, total timesteps: {}.".format(
                    avg_train_reward, len(
                        reward_hist), ep_i + 1, ts
                )
            )
            reward_hist.clear()

        gym_env.decay_epsilon()
        ep_i += 1

    logger.info("Avg. reward history during evaluation for {}: {}".format(
        test_run_name, average_reward_eval
    )
    )
    logger.info("Avg. reward history during training for {}: {}".format(
        test_run_name, average_reward_train
    )
    )
    return average_reward_train, num_episodes_train, average_reward_eval, num_episodes_eval, timesteps_history, trainer, predictor, gym_env
