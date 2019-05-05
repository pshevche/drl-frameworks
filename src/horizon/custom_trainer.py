import logging
import numpy as np
import io

from Horizon.ml.rl.test.gym import run_gym as horizon_runner
from Horizon.ml.rl.test.gym.open_ai_gym_environment import (
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
    timesteps_total=1000,
    checkpoint_after_ts=1,
    num_propagation_steps=None
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
):
    """Train off of dynamic set of transitions generated on-policy."""
    
    total_timesteps = 0
    avg_test_history, avg_reward_history, timestep_history, test_episodes, eval_episodes = [], [], [],[],[]
    best_episode_score_seeen = -1e20
    episodes_since_solved = 0
    solved = False
    policy_id = 0
    reward_hist = list()
    i = 0
    while i < num_episodes and total_timesteps < timesteps_total:
        if (
            max_episodes_to_run_after_solved is not None
            and episodes_since_solved > max_episodes_to_run_after_solved
        ):
            break

        if solved:
            episodes_since_solved += 1

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
            total_timesteps += 1
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
                total_timesteps % checkpoint_after_ts == 0 or total_timesteps == timesteps_total) and (
                start_saving_from_score is None
                or best_episode_score_seeen >= start_saving_from_score
            ):
                save_timesteps_to_dataset.insert(
                    mdp_id=i,
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
                total_timesteps % train_every_ts == 0
                and total_timesteps > train_after_ts
                and len(replay_buffer.replay_memory) >= trainer.minibatch_size
                and not (stop_training_after_solved and solved)
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
            if total_timesteps % test_every_ts == 0 and total_timesteps > test_after_ts:
                avg_test_reward= sum(reward_hist) / len(reward_hist)
                avg_test_history.append(avg_test_reward)
                
                avg_rewards, avg_discounted_rewards = gym_env.run_ep_n_times(
                    avg_over_num_episodes, predictor, test=True
                )
                if avg_rewards > best_episode_score_seeen:
                    best_episode_score_seeen = avg_rewards

                if (
                    solved_reward_threshold is not None
                    and best_episode_score_seeen > solved_reward_threshold
                ):
                    solved = True
                avg_reward_history.append(avg_rewards)
                test_episodes.append(len(reward_hist))
                eval_episodes.append(avg_over_num_episodes)
                timestep_history.append(total_timesteps)
                logger.info(
                    "Achieved an average reward score of {} over {} evaluations."
                    " Total episodes: {}, total timesteps: {}.".format(
                        avg_rewards, avg_over_num_episodes, i + 1, total_timesteps
                    )
                )
                logger.info(
                    "Achieved an average reward score of {} during {} training episodes."
                    " Total episodes: {}, total timesteps: {}.".format(
                        avg_test_reward, len(reward_hist), i + 1, total_timesteps
                    )
                )
                reward_hist.clear()
                if score_bar is not None and avg_rewards > score_bar:
                    logger.info(
                        "Avg. reward history during evaluation for {}: {}".format(
                            test_run_name, avg_reward_history
                        )
                    )
                    logger.info(
                        "Avg. reward history during training for {}: {}".format(
                            test_run_name, avg_test_history
                        )
                    )
                    return avg_reward_history, avg_test_history, timestep_history, test_episodes, eval_episodes, trainer, predictor

            if max_steps and ep_timesteps >= max_steps:
                break
        reward_hist.append(reward_sum)

        # Always eval on last episode if previous eval loop didn't return.
        if i == num_episodes - 1:
            avg_rewards, avg_discounted_rewards = gym_env.run_ep_n_times(
                avg_over_num_episodes, predictor, test=True
            )
            avg_reward_history.append(avg_rewards)
            timestep_history.append(total_timesteps)
            logger.info(
                "Achieved an average reward score of {} over {} evaluations."
                " Total episodes: {}, total timesteps: {}.".format(
                    avg_rewards, avg_over_num_episodes, i + 1, total_timesteps
                )
            )

            avg_test_reward= sum(reward_hist) / len(reward_hist)
            avg_test_history.append(avg_test_reward)
                
            logger.info(
                    "Achieved an average reward score of {} during {} training episodes."
                    " Total episodes: {}, total timesteps: {}.".format(
                        avg_test_reward, len(reward_hist), i + 1, total_timesteps
                    )
            )
            reward_hist.clear()


        if solved:
            gym_env.epsilon = gym_env.minimum_epsilon
        else:
            gym_env.decay_epsilon()

        i += 1

    logger.info("Avg. reward history during evaluation for {}: {}".format(
               test_run_name, avg_reward_history
              )
    )
    logger.info("Avg. reward history during training for {}: {}".format(
               test_run_name, avg_test_history
              )
    )
    return avg_reward_history, avg_test_history, timestep_history, test_episodes, eval_episodes, trainer, predictor
