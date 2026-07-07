import gymnasium as gym
from gymnasium.wrappers import FrameStack, FlattenObservation
try:
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN # type: ignore
except Exception:  # MetaWorld is not installed on the Franka path
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN = {}, {}

import pickle
import numpy as np
import wandb
import datetime
import tempfile
import argparse
import os

import torch
from torch.utils.data import DataLoader

# PyTorch 2.6+ changed weights_only default to True, but SB3 policy files embed
# gymnasium/numpy/imitation globals that are not allowlisted.  All checkpoints in
# this repo are produced by build_base_policy.py (trusted source), so False is safe.
_torch_load_orig = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_compat

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
import stable_baselines3.common.logger as sb_logger
from stable_baselines3.common.utils import get_schedule_fn

from imitation.util import util
from imitation.util.logger import HierarchicalLogger, _build_output_formats
from imitation.policies.base import NormalizeFeaturesExtractor
from imitation.util.networks import RunningNorm

from mile.utils import prepare_dataset, read_config, DictDataset, Logger, log_to_file
from mile.algorithm import InterventionTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rand = np.random.randint(0, 1000)


def _features_extractor_class(env_name):
    """Plain normalizing extractor everywhere.

    The Franka path used to mask dead dims (cube quats + bottom_z); the reduced observation
    now excludes them outright, so mental models / policies match the base policy
    (mile_franka.policies.bc) with the plain extractor and no masking is needed.
    """
    return NormalizeFeaturesExtractor


def build_franka_or_metaworld_env(env_name):
    """Build the wrapped (FrameStack+Flatten) training env for either backend."""
    if env_name + '-goal-observable' in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + '-goal-observable']()
        env._freeze_rand_vec = False
        env = FrameStack(env, 4)
        env = FlattenObservation(env)
    elif env_name.startswith('Franka'):
        from mile_franka.envs.registration import register_franka_envs, make_franka_env
        register_franka_envs()
        env = make_franka_env(gym.make(env_name))
    else:
        env = gym.make(env_name)
    return Monitor(env)


def offline_training(config):
    env_name = config['experiment']['env_name']
    if env_name+'-goal-observable' in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+'-goal-observable']()
        env._freeze_rand_vec = False
        env = FrameStack(env, 4)
        env = FlattenObservation(env)
    else:
        env = gym.make(env_name)
    env = Monitor(env)
    env.reset()

    with open(config['experiment']['dataset_path'], 'rb') as f:
        dataset = pickle.load(f)    
    train_set, valid_set = prepare_dataset(dataset)
    train_set = DictDataset(train_set)
    valid_set = DictDataset(valid_set)
    print(f"Dataset size: {len(train_set)}")
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=config['train']['batch_size'], shuffle=False)

    if config['experiment']['policy_type'] == 'sac':
        policy = SACPolicy.load(config['experiment']['policy_path'])
    elif config['experiment']['policy_type'] == 'qnetwork':
        policy = QNetwork.load(config['experiment']['policy_path'])
    elif config['experiment']['policy_type'] == 'bc':
        policy = ActorCriticPolicy.load(config['experiment']['policy_path'])

    if config['experiment']['mental_model_type'] == 'bc':
        mental_model = ActorCriticPolicy(observation_space=env.observation_space,
                                        action_space=env.action_space,
                                        lr_schedule=get_schedule_fn(1),
                                        net_arch=[256, 256],
                                        features_extractor_class=_features_extractor_class(env_name),
                                        features_extractor_kwargs=dict(normalize_class=RunningNorm),
                                        )
    elif config['experiment']['mental_model_type'] == 'qnetwork':
        mental_model = DQNPolicy(observation_space=env.observation_space,
                                action_space=env.action_space,
                                lr_schedule=get_schedule_fn(1),
                                net_arch=[256, 256],
                                features_extractor_class=NormalizeFeaturesExtractor,
                                features_extractor_kwargs=dict(normalize_class=RunningNorm),
                                ).q_net

    policy.to(device)
    mental_model.to(device)
    if config['experiment']['use_warm_start']:
        mental_model = ActorCriticPolicy.load(config['experiment']['warm_start_path'])

    print(config)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    format_strs = ["stdout"]
    if config['experiment']['logging']['terminal_output_to_txt']:
        format_strs.append("log")
    if config['experiment']['logging']['log_tb']:
        format_strs.append("tensorboard")
    if config['experiment']['logging']['log_wandb']:
        format_strs.append("wandb")
        EXPERIMENT_NAME = config['experiment']['name']+'-'+timestamp
        run = wandb.init(project=config['experiment']['name'], name=EXPERIMENT_NAME)
    tempdir = util.parse_path(tempfile.gettempdir())
    folder = tempdir / timestamp
    output_formats = _build_output_formats(folder, format_strs)
    custom_logger = sb_logger.Logger(str(folder), list(output_formats))
    hier_format_strs = [f for f in format_strs if (f != "wandb" and f != "tensorboard")]
    hier_logger = HierarchicalLogger(custom_logger, hier_format_strs)

    logger = Logger(hier_logger)

    trainer = InterventionTrainer(policy=policy, 
                                  mental_model=mental_model, 
                                  env=env,
                                  logger=logger,
                                  config=config,
                                  **config['train'])
    trainer.train(train_loader, val_loader)

    if config['experiment']['logging']['log_wandb']:
        run.finish()

def iterative_training(config):
    num_rounds = config['experiment']['num_rounds']
    episodes_per_round = config['experiment']['episodes_per_round']

    env_name = config['experiment']['env_name']
    env = build_franka_or_metaworld_env(env_name)
    env.reset()

    assert config['experiment']['policy_type'] in ['sac', 'qnetwork', 'bc'], 'Invalid policy type, choose from sac, qnetwork, bc'

    if config['experiment']['policy_type'] == 'sac':
        policy = SACPolicy.load(config['experiment']['policy_path'])
        intervention_policy = None
        if config['experiment'].get('collector', 'synthetic') == 'synthetic':
            intervention_policy = SACPolicy.load(config['experiment']['intervention_policy_path'])
            intervention_policy.eval()
    elif config['experiment']['policy_type'] == 'qnetwork':
        policy = QNetwork.load(config['experiment']['policy_path'])
        intervention_policy = None
        if config['experiment'].get('collector', 'synthetic') == 'synthetic':
            intervention_policy = QNetwork.load(config['experiment']['intervention_policy_path'])
            intervention_policy.eval()
    elif config['experiment']['policy_type'] == 'bc':
        policy = ActorCriticPolicy.load(config['experiment']['policy_path'])
        intervention_policy = None
        if config['experiment'].get('collector', 'synthetic') == 'synthetic':
            intervention_policy = ActorCriticPolicy.load(config['experiment']['intervention_policy_path'])
            intervention_policy.eval()

    if config['experiment']['mental_model_type'] == 'bc':
        mental_model = ActorCriticPolicy(observation_space=env.observation_space,
                                        action_space=env.action_space,
                                        lr_schedule=get_schedule_fn(1),
                                        net_arch=[256, 256],
                                        features_extractor_class=_features_extractor_class(env_name),
                                        features_extractor_kwargs=dict(normalize_class=RunningNorm),
                                        )
        gt_mental_model = None
        if config['experiment'].get('collector', 'synthetic') == 'synthetic':
            gt_mental_model = ActorCriticPolicy.load(config['experiment']['gt_mental_model_path'])
            gt_mental_model.eval()
    elif config['experiment']['mental_model_type'] == 'qnetwork':
        mental_model = DQNPolicy(observation_space=env.observation_space,
                                action_space=env.action_space,
                                lr_schedule=get_schedule_fn(1),
                                net_arch=[256, 256],
                                features_extractor_class=NormalizeFeaturesExtractor,
                                features_extractor_kwargs=dict(normalize_class=RunningNorm),
                                ).q_net
        gt_mental_model = None
        if config['experiment'].get('collector', 'synthetic') == 'synthetic':
            gt_mental_model = QNetwork.load(config['experiment']['gt_mental_model_path'])
            gt_mental_model.eval()

    policy.to(device)
    if intervention_policy is not None:
        intervention_policy.to(device)
    mental_model.to(device)
    if gt_mental_model is not None:
        gt_mental_model.to(device)

    if config['experiment']['use_warm_start']:
        mental_model = ActorCriticPolicy.load(config['experiment']['warm_start_path'])

    print(config)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    EXPERIMENT_NAME = config['experiment']['name']+'-'+timestamp
    format_strs = ["stdout"]
    if config['experiment']['logging']['terminal_output_to_txt']:
        format_strs.append("log")
    if config['experiment']['logging']['log_tb']:
        format_strs.append("tensorboard")
    if config['experiment']['logging']['log_wandb']:
        format_strs.append("wandb")
        run = wandb.init(project=config['experiment']['name'], name=EXPERIMENT_NAME)
    tempdir = util.parse_path(tempfile.gettempdir())
    folder = tempdir / timestamp
    output_formats = _build_output_formats(folder, format_strs)
    custom_logger = sb_logger.Logger(str(folder), list(output_formats))
    hier_format_strs = [f for f in format_strs if (f != "wandb" and f != "tensorboard")]
    hier_logger = HierarchicalLogger(custom_logger, hier_format_strs)

    logger = Logger(hier_logger)

    trainer = InterventionTrainer(policy=policy, 
                                  mental_model=mental_model, 
                                  env=env,
                                  logger=logger,
                                  config=config,
                                  **config['train'])

    collector_type = config['experiment'].get('collector', 'synthetic')
    real_collector = None
    intervener = None
    if collector_type == 'real':
        from mile_franka.collect import Collector, ScriptedIntervener, TeleopIntervener
        real_collector = Collector(env)
        which = config['experiment'].get('intervener', 'scripted')
        if which == 'scripted':
            from mile_franka.policies.scripted import ScriptedStackPolicy
            task_config = getattr(env.unwrapped, 'config', None)
            intervener = ScriptedIntervener(
                ScriptedStackPolicy(task_config, mediocre=False), env=env)
        elif which == 'spacemouse':
            from mile_franka.teleop.spacemouse import SpaceMouseDevice
            intervener = TeleopIntervener(SpaceMouseDevice())
        elif which == 'joystick':
            from mile_franka.teleop.joystick import JoystickDevice
            joystick_translation_scale = config['experiment'].get('joystick_translation_scale', 0.5)
            # Xbox 360: left-stick-forward=axis1(-=fwd), left-stick-left=axis0(-=left), right-stick-up=axis4(-=up)
            # clutch=RB(5), gripper=A(0), done=Start(7)
            intervener = TeleopIntervener(JoystickDevice(
                ax_x=1, ax_x_sign=-1.0,
                ax_y=0, ax_y_sign=-1.0,
                ax_z=4, ax_z_sign=-1.0,
                clutch_button=5, gripper_button=0, done_button=7, discard_button=6,
                gripper_toggle=True,
                translation_scale=joystick_translation_scale,
            ))
        elif which == 'keyboard':
            from mile_franka.teleop.keyboard import KeyboardDevice
            kb_scale = config['experiment'].get('keyboard_translation_scale', 0.02)
            intervener = TeleopIntervener(KeyboardDevice(translation_scale=kb_scale))
        elif which == 'vive':
            from mile_franka.teleop.vive import ViveDevice
            vive_scale = config['experiment'].get('vive_translation_scale', 1.0)
            # HTC Vive wand: grip=segment toggle, menu=gripper toggle, trackpad=done,
            # trigger=discard (see mile_franka/teleop/vive.py for the button mapping).
            intervener = TeleopIntervener(ViveDevice(translation_scale=vive_scale))
        else:
            raise ValueError(f'Unknown intervener: {which}')

    if config['experiment']['include_offline_dataset']:
        with open(config['experiment']['offline_dataset_path'], 'rb') as f:
            dataset = pickle.load(f)    
    else:
        if isinstance(env.action_space, gym.spaces.Box):
            dataset = dict(state=[], 
                        rollout_action=[], 
                        action=[], 
                        intervention_prob=[], 
                        intervention=[], 
                        reward=[], 
                        next_state=[], 
                        done=[])
        elif isinstance(env.action_space, gym.spaces.Discrete):
            dataset = dict(state=[], 
                    rollout_action=[],
                    intervention=[],
                    human_action=[],
                    human_action_prob=[], 
                    next_state=[], 
                    reward=[], 
                    done=[])

    for round in range(num_rounds):
        log_to_file('Round: {}'.format(round), EXPERIMENT_NAME+'_log.txt')
        print('Collecting intervention data...')
        if collector_type == 'real':
            additional_data, mean_score, mean_success_rate = real_collector.collect_intervention(
                policy=trainer.policy, intervener=intervener, n_episodes=episodes_per_round,
                max_t=10_000)
        else:
            from collect_synthetic_interventions import collect_synthetic_data
            additional_data, mean_score, mean_success_rate = collect_synthetic_data(env=env,
                                                                                    n_episodes=episodes_per_round,
                                                                                    cost=trainer.intervention_cost,
                                                                                    cdf_scale=trainer.intervention_scale,
                                                                                    rollout_policy=trainer.policy,
                                                                                    intervention_policy=intervention_policy,
                                                                                    mental_model=gt_mental_model)
        log_to_file('Dataset size: {}'.format(len(additional_data['state'])), EXPERIMENT_NAME+'_log.txt')
        log_to_file(f"Percentage of no-intervention: {additional_data['intervention'].count(0)/len(additional_data['intervention'])}", EXPERIMENT_NAME+'_log.txt')
        log_to_file(f"Success rate: {mean_success_rate}", EXPERIMENT_NAME+'_log.txt')

        # Save per-round raw intervention data for reproducibility/debugging.
        save_dir = config['experiment']['save']['outdir']
        os.makedirs(save_dir, exist_ok=True)
        round_data_path = os.path.join(save_dir, f'intervention_data_round{round}.pkl')
        with open(round_data_path, 'wb') as f:
            pickle.dump(additional_data, f)
        log_to_file(f'Saved round data -> {round_data_path}', EXPERIMENT_NAME+'_log.txt')

        for key in dataset.keys():
            if round == 0:
                dataset[key].extend(additional_data[key])
            else:
                dataset[key] = np.concatenate((dataset[key], additional_data[key]), axis=0)
        train_set, valid_set = prepare_dataset(dataset)
        print("Current dataset size: {}".format(len(train_set['state'])))
        train_set = DictDataset(train_set)
        valid_set = DictDataset(valid_set)
        train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True)
        val_loader = DataLoader(valid_set, batch_size=config['train']['batch_size'], shuffle=False)
        trainer.train(train_loader, val_loader, round)

        # Save accumulated dataset after this round's training for reproducibility.
        accum_path = os.path.join(save_dir, f'accumulated_dataset_round{round}.pkl')
        with open(accum_path, 'wb') as f:
            pickle.dump(dataset, f)
        log_to_file(f'Saved accumulated dataset -> {accum_path}', EXPERIMENT_NAME+'_log.txt')

    if config['experiment']['logging']['log_wandb']:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description='Intervention training')
    parser.add_argument('--config', type=str, default='config/metaworld.yaml', help='Path to the config file')
    args = parser.parse_args()
    config = read_config(args.config)
    if config['experiment']['save']['enabled']:
        os.makedirs(config['experiment']['save']['outdir'], exist_ok=True)

    if config['experiment']['mode'] == 'offline':
        offline_training(config)
    elif config['experiment']['mode'] == 'iterative':
        iterative_training(config)
    else:
        raise ValueError('Invalid mode')


if __name__ == "__main__":
    main()

