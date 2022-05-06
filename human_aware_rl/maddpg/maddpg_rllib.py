import ray
from ray.rllib.agents import pg
from ray.tune import run_experiments
from ray.tune.registry import register_trainable, register_env
import ray.rllib.contrib.maddpg.maddpg as maddpg
import argparse
from human_aware_rl.ppo.ppo_rllib_client import my_config

import os

from human_aware_rl.rllib.rllib import OvercookedMultiAgent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CustomStdOut(object):
    def _log_result(self, result):
        if result["training_iteration"] % 50 == 0:
            try:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    result["timesteps_total"],
                    result["episodes_total"],
                    result["episode_reward_mean"],
                    result["policy_reward_mean"],
                    round(result["time_total_s"] - self.cur_time, 3)
                ))
            except:
                pass

            self.cur_time = result["time_total_s"]


def parse_args():
    parser = argparse.ArgumentParser("MADDPG with OpenAI MPE")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple",
                        choices=['simple', 'simple_speaker_listener',
                                 'simple_crypto', 'simple_push',
                                 'simple_tag', 'simple_spread', 'simple_adversary'],
                        help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25,
                        help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=3,
                        help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0,
                        help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg",
                        help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg",
                        help="policy of adversaries")

    parser.add_argument("--environment_params", type=dict, default={
        # To be passed into OvercookedGridWorld constructor

        "mdp_params": {
            "layout_name": "cramped_room",
            "rew_shaping_params": {
                "PLACEMENT_IN_POT_REW": 3,
                "DISH_PICKUP_REWARD": 3,
                "SOUP_PICKUP_REWARD": 5,
                "DISH_DISP_DISTANCE_REW": 0,
                "POT_DISTANCE_REW": 0,
                "SOUP_DISTANCE_REW": 0,
            }
        },
        # To be passed into OvercookedEnv constructor
        "env_params": {
            "horizon": 400  # Max episode length
        },

        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params": {
            "reward_shaping_factor": 1.0,  # Constant by which shaped rewards are multiplied by when calculating total reward
            "reward_shaping_horizon": float('inf'),  # Linearly anneal the reward shaping factor such that it reaches zero after this number of timesteps
            "use_phi": True,  # Whether dense reward should come from potential function or not
            "bc_schedule": OvercookedMultiAgent.self_play_bc_schedule
        }
    }, help="environemnt parameters")


    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    # NOTE: 1 iteration = sample_batch_size * num_workers timesteps * num_envs_per_worker
    parser.add_argument("--sample-batch-size", type=int, default=25,
                        help="number of data points sampled /update /worker")
    parser.add_argument("--train-batch-size", type=int, default=1024,
                        help="number of data points /update")
    parser.add_argument("--n-step", type=int, default=1,
                        help="length of multistep value backup")
    parser.add_argument("--num-units", type=int, default=64,
                        help="number of units in the mlp")
    parser.add_argument("--replay-buffer", type=int, default=1000000,
                        help="size of replay buffer in training")

    # Checkpoint
    parser.add_argument("--checkpoint-freq", type=int, default=7500,
                        help="save model once every time this many iterations are completed")
    parser.add_argument("--local-dir", type=str, default="./ray_results",
                        help="path to save checkpoints")
    parser.add_argument("--restore", type=str, default=None,
                        help="directory in which training state and model are loaded")

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-envs-per-worker", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=0)

    return parser.parse_args()

environment_params = my_config()

# Dummy wrapper to pass rllib type checks
def _env_creator(env_config):
    # Re-import required here to work with serialization
    from human_aware_rl.rllib.rllib import OvercookedMultiAgent
    return OvercookedMultiAgent.from_config(env_config)


def main(args):
    ray.init()
    MADDPGAgent = maddpg.MADDPGTrainer.with_updates(
        mixins=[CustomStdOut]
    )
    register_trainable("MADDPG", MADDPGAgent)

    register_env("overcooked_multi_agent", _env_creator)

    # TODO send env params
    environment_params = args.environment_params
    env = OvercookedMultiAgent.from_config(environment_params)
    print(env.ppo_observation_space)
    print(env.action_space)
    def gen_policy(i):
        return (
            None,
            env.ppo_observation_space,
            env.action_space,
            {
                "agent_id": i,
                "use_local_critic": True,
                "obs_space_dict": env.ppo_observation_space,
                "act_space_dict": env.action_space,
            }
        )

    policies = {"policy_%d" %i: gen_policy(i) for i in [1, 2]}
    policy_ids = list(policies.keys())

    run_experiments({
        "MADDPG_RLLib": {
            "run": "contrib/MADDPG",
            "env": "overcooked_multi_agent",
            "stop": {
                "episodes_total": args.num_episodes,
            },
            "checkpoint_freq": args.checkpoint_freq,
            "local_dir": args.local_dir,
            "restore": args.restore,
            "config": {
                # === Log ===
                "log_level": "ERROR",

                # === Environment ===
                "env_config": environment_params,
                "num_envs_per_worker": args.num_envs_per_worker,
                "horizon": args.max_episode_len,

                # === Policy Config ===
                # --- Model ---
                "good_policy": args.good_policy,
                "adv_policy": args.adv_policy,
                "actor_hiddens": [args.num_units] * 2,
                "actor_hidden_activation": "relu",
                "critic_hiddens": [args.num_units] * 2,
                "critic_hidden_activation": "relu",
                "n_step": args.n_step,
                "gamma": args.gamma,

                # --- Exploration ---
                "tau": 0.01,

                # --- Replay buffer ---
                "buffer_size": args.replay_buffer,

                # --- Optimization ---
                "actor_lr": args.lr,
                "critic_lr": args.lr,
                "learning_starts": args.train_batch_size * args.max_episode_len,
                "sample_batch_size": args.sample_batch_size,
                "train_batch_size": args.train_batch_size,
                "batch_mode": "truncate_episodes",

                # --- Parallelism ---
                "num_workers": args.num_workers,
                "num_gpus": args.num_gpus,
                "num_gpus_per_worker": 0,

                # === Multi-agent setting ===
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": ray.tune.function(
                        lambda i: policy_ids[i]
                    )
                },
            },
        },
    }, verbose=3)


if __name__ == '__main__':
    args = parse_args()
    main(args)