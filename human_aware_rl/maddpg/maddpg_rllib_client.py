# All imports except rllib
import argparse, os, sys
import numpy as np

# environment variable that tells us whether this code is running on the server or not
LOCAL_TESTING = os.getenv('RUN_ENV', 'production') == 'local'

# Sacred setup (must be before rllib imports)
from sacred import Experiment

ex = Experiment("MADDPG RLLib")

# Necessary work-around to make sacred pickling compatible with rllib
from sacred import SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Slack notification configuration
from sacred.observers import SlackObserver

if os.path.exists('slack.json') and not LOCAL_TESTING:
    slack_obs = SlackObserver.from_config('slack.json')
    ex.observers.append(slack_obs)

    # Necessary for capturing stdout in multiprocessing setting
    SETTINGS.CAPTURE_MODE = 'sys'

# rllib and rllib-dependent imports
# Note: tensorflow and tensorflow dependent imports must also come after rllib imports
# This is because rllib disables eager execution. Otherwise, it must be manually disabled
import ray
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.registry import register_env
from human_aware_rl.ppo.ppo_rllib import RllibPPOModel, RllibLSTMPPOModel
from human_aware_rl.rllib.rllib import OvercookedMultiAgent, save_trainer, gen_maddpg_trainer_from_params
from human_aware_rl.imitation.behavior_cloning_tf2 import BehaviorCloningPolicy, BC_SAVE_DIR


###################### Temp Documentation #######################
#   run the following command in order to train a PPO self-play #
#   agent with the static parameters listed in my_config        #
#                                                               #
#   python maddpg_rllib_client.py                                  #
#                                                               #
#   In order to view the results of training, run the following #
#   command                                                     #
#                                                               #
#   tensorboard --log-dir ~/ray_results/                        #
#                                                               #
#################################################################

# Dummy wrapper to pass rllib type checks
def _env_creator(env_config):
    # Re-import required here to work with serialization
    from human_aware_rl.rllib.rllib import OvercookedMultiAgent
    return OvercookedMultiAgent.from_config(env_config)


@ex.config
def my_config():
    ### Model params ###

    # Whether dense reward should come from potential function or not
    use_phi = False

    # whether to use recurrence in ppo model
    use_lstm = False

    # Base model params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3

    # LSTM memory cell size (only used if use_lstm=True)
    CELL_SIZE = 256

    # whether to use D2RL https://arxiv.org/pdf/2010.09163.pdf (concatenation the result of last conv layer to each hidden layer); works only when use_lstm is False
    D2RL = False
    ### Training Params ###

    num_workers = 1 if not LOCAL_TESTING else 2

    # list of all random seeds to use for experiments, used to reproduce results
    seeds = [0]

    # Placeholder for random for current trial
    seed = None

    # Number of gpus the central driver should use
    num_gpus = 0 if LOCAL_TESTING else 1

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    train_batch_size = 20000 if not LOCAL_TESTING else 800 #-> 4k + 5k iter

    sample_batch_size = 2000 if not LOCAL_TESTING else 800

    # Whether all PPO agents should share the same policy network
    shared_policy = True

    # Number of training iterations to run
    num_training_iters = 1200 if not LOCAL_TESTING else 4

    # Stepsize of SGD.
    lr = 1e-3

    gamma = 0.95

    tau = 0.01

    buffer_size = 1000000

    # How many trainind iterations (calls to trainer.train()) to run before saving model checkpoint
    save_freq = 25

    # How many training iterations to run between each evaluation
    evaluation_interval = 50 if not LOCAL_TESTING else 1

    # How many timesteps should be in an evaluation episode
    evaluation_ep_length = 400

    # Number of games to simulation each evaluation
    evaluation_num_games = 1

    # Whether to display rollouts in evaluation
    evaluation_display = False

    # Where to log the ray dashboard stats
    temp_dir = os.path.join(os.path.abspath(os.sep), "tmp", "ray_tmp")

    # Where to store model checkpoints and training stats
    results_dir = DEFAULT_RESULTS_DIR

    # Whether tensorflow should execute eagerly or not
    eager = False

    # Whether to log training progress and debugging info
    verbose = True

    ### Environment Params ###
    # Which overcooked level to use
    layout_name = "cramped_room"

    # all_layout_names = '_'.join(layout_names)

    # Linearly anneal the reward shaping factor such that it reaches zero after this number of timesteps
    reward_shaping_horizon = float('inf')

    # Constant by which shaped rewards are multiplied by when calculating total reward
    reward_shaping_factor = 1.0

    # Name of directory to store training results in (stored in ~/ray_results/<experiment_name>)
    params_str = str(use_phi) + "_tbs=%d_sbs=%d_iter=%d_rsh=%f_rsf=%f_lr=%f_gamma=%f_tau=%f" % (
        train_batch_size,
        sample_batch_size,
        num_training_iters,
        reward_shaping_horizon,
        reward_shaping_factor,
        lr,
        gamma,
        tau
    )

    experiment_name = "{0}_{1}_{2}".format("MADDPG", layout_name, params_str)

    # Rewards the agent will receive for intermediate actions
    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }

    # Max episode length
    horizon = 400


    # TODO! Custom model -> should not be the case for us
    # To be passed into rl-lib model/custom_options config
    model_params = {
        "use_lstm": use_lstm,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "CELL_SIZE": CELL_SIZE,
        "D2RL": D2RL
    }

    # TODO! What training params to we need to MADDPG?
    # to be passed into the rllib.PPOTrainer class
    training_params = {
        "num_workers": num_workers,
        "num_gpus": 0,
        "num_gpus_per_worker": 0,
        "num_envs_per_worker": 1,

        # === Policy Config ===
        # --- Model ---
        "good_policy": "maddpg",
        "adv_policy": "maddpg",
        "actor_hiddens": [64, 64, 64],
        "actor_hidden_activation": "relu",
        "critic_hiddens": [64, 64, 64],
        "critic_hidden_activation": "relu",
        "n_step": 1,
        "gamma": gamma,

        # --- Exploration ---
        "tau": tau,

        # --- Replay buffer ---
        "buffer_size": 1000000,

        # --- Optimization ---
        "actor_lr": lr,
        "critic_lr": lr,
        "learning_starts": train_batch_size * sample_batch_size,
        "sample_batch_size": sample_batch_size,
        "train_batch_size": train_batch_size,

        "seed": seed,
        "evaluation_interval": evaluation_interval,
        "eager": eager,
        "log_level": "WARN" if verbose else "ERROR"
    }

    # To be passed into AgentEvaluator constructor and _evaluate function
    evaluation_params = {
        "ep_length": evaluation_ep_length,
        "num_games": evaluation_num_games,
        "display": evaluation_display
    }

    bc_schedule = OvercookedMultiAgent.self_play_bc_schedule

    environment_params = {
        # To be passed into OvercookedGridWorld constructor

        "mdp_params": {
            "layout_name": layout_name,
            "rew_shaping_params": rew_shaping_params
        },
        # To be passed into OvercookedEnv constructor
        "env_params": {
            "horizon": horizon
        },

        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params": {
            "reward_shaping_factor": reward_shaping_factor,
            "reward_shaping_horizon": reward_shaping_horizon,
            "use_phi": use_phi,
            "bc_schedule": bc_schedule,
        }
    }

    # TODO? is this where they enforce the usage of their PPO instead of the library one?
    ray_params = {
        "custom_model_id": "MyPPOModel",
        "custom_model_cls": RllibLSTMPPOModel if model_params['use_lstm'] else RllibPPOModel,
        "temp_dir": temp_dir,
        "env_creator": _env_creator
    }

    ### BC Params ###
    # path to pickled policy model for behavior cloning
    bc_model_dir = os.path.join(BC_SAVE_DIR, "default")

    # Whether bc agents should return action logit argmax or sample
    bc_stochastic = True

    bc_schedule = OvercookedMultiAgent.self_play_bc_schedule

    bc_params = {
        "bc_policy_cls": BehaviorCloningPolicy,
        "bc_config": {
            "model_dir": bc_model_dir,
            "stochastic": bc_stochastic,
            "eager": eager
        }
    }

    params = {
        "model_params": model_params,
        "training_params": training_params,
        "environment_params": environment_params,
        "bc_params": bc_params,
        "shared_policy": shared_policy,
        "num_training_iters": num_training_iters,
        "evaluation_params": evaluation_params,
        "experiment_name": experiment_name,
        "save_every": save_freq,
        "seeds": seeds,
        "results_dir": results_dir,
        "ray_params": ray_params,
        "verbose": verbose
    }


def run(params):
    # Retrieve the tune.Trainable object that is used for the experiment
    trainer = gen_maddpg_trainer_from_params(params)

    # Object to store training results in
    result = {}

    # Training loop
    for i in range(params['num_training_iters']):
        if params['verbose']:
            print("Starting training iteration", i)
        result = trainer.train()

        if i % params['save_every'] == 0:
            save_path = save_trainer(trainer, params)
            if params['verbose']:
                print("saved trainer at", save_path)

    # Save the state of the experiment at end
    save_path = save_trainer(trainer, params)
    if params['verbose']:
        print("saved trainer at", save_path)

    return result


@ex.automain
def main(params):
    # List of each random seed to run
    seeds = params['seeds']
    del params['seeds']

    # List to store results dicts (to be passed to sacred slack observer)
    results = []

    # Train an agent to completion for each random seed specified
    for seed in seeds:
        # Override the seed
        params['training_params']['seed'] = seed

        # Do the thing
        result = run(params)
        results.append(result)
    for res in results:
        print(res['custom_metrics'])
    # Return value gets sent to our slack observer for notification
    average_sparse_reward = np.mean([res['custom_metrics']['sparse_reward_mean'] for res in results])
    average_episode_reward = np.mean([res['episode_reward_mean'] for res in results])
    return {"average_sparse_reward": average_sparse_reward, "average_total_reward": average_episode_reward}