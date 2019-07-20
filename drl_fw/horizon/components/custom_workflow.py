import logging
import numpy as np
import io
import torch
from typing import Dict

from ml.rl.test.gym import run_gym as horizon_runner
from ml.rl.test.gym.open_ai_gym_environment import (
    ModelType,
)
from ml.rl.thrift.core.ttypes import (
    CNNParameters,
    ContinuousActionModelParameters,
    DDPGModelParameters,
    DDPGNetworkParameters,
    DDPGTrainingParameters,
    DiscreteActionModelParameters,
    FeedForwardParameters,
    OptimizerParameters,
    RainbowDQNParameters,
    RLParameters,
    SACModelParameters,
    SACTrainingParameters,
    TrainingParameters,
)
from ml.rl.models.dueling_q_network import DuelingQNetwork
from ml.rl.preprocessing.normalization import (
    NormalizationParameters,
    get_num_output_features,
)
from ml.rl.models.dqn import FullyConnectedDQN
from ml.rl.training.ddpg_trainer import ActorNetModel, CriticNetModel, DDPGTrainer
from ml.rl.test.gym.gym_predictor import (
    GymDDPGPredictor,
    GymDQNPredictor,
    GymSACPredictor,
)
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.workflow.transitional import (
    create_parametric_dqn_trainer_from_params,
)

from drl_fw.horizon.components.park_trainers import (
    ParkDQNTrainer,
)
from drl_fw.horizon.components.park_predictors import ParkDQNPredictor

logger = logging.getLogger(__name__)

# CUSTOM TRAINER


def create_park_predictor(trainer, model_type, use_gpu, action_dim=None):
    if model_type == ModelType.CONTINUOUS_ACTION.value:
        predictor = GymDDPGPredictor(trainer, action_dim)
    elif model_type == ModelType.SOFT_ACTOR_CRITIC.value:
        predictor = GymSACPredictor(trainer, action_dim)
    elif model_type in (
        ModelType.PYTORCH_DISCRETE_DQN.value,
        ModelType.PYTORCH_PARAMETRIC_DQN.value,
    ):
        predictor = ParkDQNPredictor(trainer, action_dim)
    else:
        raise NotImplementedError()
    return predictor


def create_park_trainer(model_type, params, rl_parameters, use_gpu, env):
    if model_type == ModelType.PYTORCH_DISCRETE_DQN.value:
        training_parameters = params["training"]
        if isinstance(training_parameters, dict):
            training_parameters = TrainingParameters(**training_parameters)
        rainbow_parameters = params["rainbow"]
        if isinstance(rainbow_parameters, dict):
            rainbow_parameters = RainbowDQNParameters(**rainbow_parameters)
        if env.img:
            assert (
                training_parameters.cnn_parameters is not None
            ), "Missing CNN parameters for image input"
            if isinstance(training_parameters.cnn_parameters, dict):
                training_parameters.cnn_parameters = CNNParameters(
                    **training_parameters.cnn_parameters
                )
            training_parameters.cnn_parameters.conv_dims[0] = env.num_input_channels
            training_parameters.cnn_parameters.input_height = env.height
            training_parameters.cnn_parameters.input_width = env.width
            training_parameters.cnn_parameters.num_input_channels = (
                env.num_input_channels
            )
        else:
            assert (
                training_parameters.cnn_parameters is None
            ), "Extra CNN parameters for non-image input"
        trainer_params = DiscreteActionModelParameters(
            actions=env.actions,
            rl=rl_parameters,
            training=training_parameters,
            rainbow=rainbow_parameters,
        )
        trainer = create_park_dqn_trainer_from_params(
            model=trainer_params, normalization_parameters=env.normalization, use_gpu=use_gpu,
            env=env.env
        )
    elif model_type == ModelType.PYTORCH_PARAMETRIC_DQN.value:
        training_parameters = params["training"]
        if isinstance(training_parameters, dict):
            training_parameters = TrainingParameters(**training_parameters)
        rainbow_parameters = params["rainbow"]
        if isinstance(rainbow_parameters, dict):
            rainbow_parameters = RainbowDQNParameters(**rainbow_parameters)
        if env.img:
            assert (
                training_parameters.cnn_parameters is not None
            ), "Missing CNN parameters for image input"
            training_parameters.cnn_parameters.conv_dims[0] = env.num_input_channels
        else:
            assert (
                training_parameters.cnn_parameters is None
            ), "Extra CNN parameters for non-image input"
        trainer_params = ContinuousActionModelParameters(
            rl=rl_parameters, training=training_parameters, rainbow=rainbow_parameters
        )
        trainer = create_parametric_dqn_trainer_from_params(
            trainer_params, env.normalization, env.normalization_action, use_gpu,
            env=env.env
        )
    elif model_type == ModelType.CONTINUOUS_ACTION.value:
        training_parameters = params["shared_training"]
        if isinstance(training_parameters, dict):
            training_parameters = DDPGTrainingParameters(**training_parameters)

        actor_parameters = params["actor_training"]
        if isinstance(actor_parameters, dict):
            actor_parameters = DDPGNetworkParameters(**actor_parameters)

        critic_parameters = params["critic_training"]
        if isinstance(critic_parameters, dict):
            critic_parameters = DDPGNetworkParameters(**critic_parameters)

        trainer_params = DDPGModelParameters(
            rl=rl_parameters,
            shared_training=training_parameters,
            actor_training=actor_parameters,
            critic_training=critic_parameters,
        )

        action_range_low = env.action_space.low.astype(np.float32)
        action_range_high = env.action_space.high.astype(np.float32)

        state_dim = get_num_output_features(env.normalization)
        action_dim = get_num_output_features(env.normalization_action)

        # Build Actor Network
        actor_network = ActorNetModel(
            layers=(
                [state_dim] + trainer_params.actor_training.layers[1:-1] + [action_dim]
            ),
            activations=trainer_params.actor_training.activations,
            fl_init=trainer_params.shared_training.final_layer_init,
            state_dim=state_dim,
            action_dim=action_dim,
            use_gpu=use_gpu,
            use_all_avail_gpus=False,
        )

        # Build Critic Network
        critic_network = CriticNetModel(
            # Ensure dims match input state and scalar output
            layers=[state_dim] + \
            trainer_params.critic_training.layers[1:-1] + [1],
            activations=trainer_params.critic_training.activations,
            fl_init=trainer_params.shared_training.final_layer_init,
            state_dim=state_dim,
            action_dim=action_dim,
            use_gpu=use_gpu,
            use_all_avail_gpus=False,
        )

        trainer = DDPGTrainer(
            actor_network,
            critic_network,
            trainer_params,
            env.normalization,
            env.normalization_action,
            torch.from_numpy(action_range_low).unsqueeze(dim=0),
            torch.from_numpy(action_range_high).unsqueeze(dim=0),
            use_gpu,
        )

    elif model_type == ModelType.SOFT_ACTOR_CRITIC.value:
        value_network = None
        value_network_optimizer = None
        if params["sac_training"]["use_value_network"]:
            value_network = FeedForwardParameters(
                **params["sac_value_training"])
            value_network_optimizer = OptimizerParameters(
                **params["sac_training"]["value_network_optimizer"]
            )

        trainer_params = SACModelParameters(
            rl=rl_parameters,
            training=SACTrainingParameters(
                minibatch_size=params["sac_training"]["minibatch_size"],
                use_2_q_functions=params["sac_training"]["use_2_q_functions"],
                use_value_network=params["sac_training"]["use_value_network"],
                q_network_optimizer=OptimizerParameters(
                    **params["sac_training"]["q_network_optimizer"]
                ),
                value_network_optimizer=value_network_optimizer,
                actor_network_optimizer=OptimizerParameters(
                    **params["sac_training"]["actor_network_optimizer"]
                ),
                entropy_temperature=params["sac_training"]["entropy_temperature"],
            ),
            q_network=FeedForwardParameters(**params["sac_q_training"]),
            value_network=value_network,
            actor_network=FeedForwardParameters(
                **params["sac_actor_training"]),
        )
        trainer = horizon_runner.get_sac_trainer(env, trainer_params, use_gpu)

    else:
        raise NotImplementedError(
            "Model of type {} not supported".format(model_type))

    return trainer


def create_park_dqn_trainer_from_params(model: DiscreteActionModelParameters,
                                        normalization_parameters: Dict[int, NormalizationParameters],
                                        use_gpu: bool = False,
                                        use_all_avail_gpus: bool = False,
                                        metrics_to_score=None,
                                        env=None
                                        ):
    metrics_to_score = metrics_to_score or []
    if model.rainbow.dueling_architecture:
        q_network = DuelingQNetwork(
            layers=[get_num_output_features(normalization_parameters)]
            + model.training.layers[1:-1]
            + [len(model.actions)],
            activations=model.training.activations,
        )
    else:
        q_network = FullyConnectedDQN(
            state_dim=get_num_output_features(normalization_parameters),
            action_dim=len(model.actions),
            sizes=model.training.layers[1:-1],
            activations=model.training.activations[:-1],
            dropout_ratio=model.training.dropout_ratio,
        )

    if use_gpu and torch.cuda.is_available():
        q_network = q_network.cuda()

    q_network_target = q_network.get_target_network()

    reward_network, q_network_cpe, q_network_cpe_target = None, None, None
    if model.evaluation.calc_cpe_in_training:
        # Metrics + reward
        num_output_nodes = (len(metrics_to_score) + 1) * len(model.actions)
        reward_network = FullyConnectedDQN(
            state_dim=get_num_output_features(normalization_parameters),
            action_dim=num_output_nodes,
            sizes=model.training.layers[1:-1],
            activations=model.training.activations[:-1],
            dropout_ratio=model.training.dropout_ratio,
        )
        q_network_cpe = FullyConnectedDQN(
            state_dim=get_num_output_features(normalization_parameters),
            action_dim=num_output_nodes,
            sizes=model.training.layers[1:-1],
            activations=model.training.activations[:-1],
            dropout_ratio=model.training.dropout_ratio,
        )

        if use_gpu and torch.cuda.is_available():
            reward_network.cuda()
            q_network_cpe.cuda()

        q_network_cpe_target = q_network_cpe.get_target_network()

    if use_all_avail_gpus:
        q_network = q_network.get_distributed_data_parallel_model()
        reward_network = (
            reward_network.get_distributed_data_parallel_model()
            if reward_network
            else None
        )
        q_network_cpe = (
            q_network_cpe.get_distributed_data_parallel_model()
            if q_network_cpe
            else None
        )

    return ParkDQNTrainer(
        q_network,
        q_network_target,
        reward_network,
        model,
        use_gpu,
        q_network_cpe=q_network_cpe,
        q_network_cpe_target=q_network_cpe_target,
        metrics_to_score=metrics_to_score,
        env=env
    )


# TRAINING PROCESS


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
                avg_train_reward = sum(
                    reward_hist) / len(reward_hist) if len(reward_hist) != 0 else 0
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

                # During the evaluation phase, environment's state may change from the last saved state, so we do a hard reset
                terminal = False
                next_state = gym_env.transform_state(gym_env.env.reset())
                next_action, next_action_probability = gym_env.policy(
                    predictor, next_state, False
                )

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
            # During the evaluation phase, environment's state may change from the last saved state, so we do a hard reset
            terminal = False
            next_state = gym_env.transform_state(gym_env.env.reset())
            next_action, next_action_probability = gym_env.policy(
                predictor, next_state, False
            )

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
