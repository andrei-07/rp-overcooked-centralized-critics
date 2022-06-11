import json
import os

from human_aware_rl.experiments.plotting import simple_plot
from human_aware_rl.rllib.rllib import load_agent, load_agent_pair, RlLibAgent
from human_aware_rl.imitation.behavior_cloning_tf2 import load_bc_model, BehaviorCloningPolicy
import numpy as np
#from human_aware_rl.rllib.rllib import load_agent
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair
from human_aware_rl.rllib.utils import get_base_ae
from overcooked_ai_py.utils import mean_and_std_err

evaluation_params = {
    "ep_length": 400,
    "num_games": 100,
    "display": False,
    "display_phi": False,
    "outer_shape": (5,4)
}

mdp_params = {
    "layout_name": "cramped_room",
    "rew_shaping_params": {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }
}


def create_bc_agent_and_evaluator(layout):
    bc_agent_model, bc_params = load_bc_model(
        '/Users/andreimija/Documents/University/human_aware_rl/human_aware_rl/imitation/bc_runs/default/' + layout)
    agent_0_policy = BehaviorCloningPolicy.from_model(bc_agent_model, bc_params, stochastic=True)

    evaluator = get_base_ae(bc_params['mdp_params'], bc_params['env_params'])
    base_env = evaluator.env

    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)

    bc_agent = RlLibAgent(agent_0_policy, 1, featurize_fn=featurize_fn)

    return bc_agent, evaluator

def evalaute_ultimate():

    cr_bc, cr_evaluator = create_bc_agent_and_evaluator('cramped_room')
    aa_bc, aa_evaluator = create_bc_agent_and_evaluator('asymmetric_advantages')
    cc_bc, cc_evaluator = create_bc_agent_and_evaluator('coordination_ring')

    ppo_cr_2229 = '/Users/andreimija/ray_results/PPO_cramped_room_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=2229/checkpoint_165/checkpoint-165'
    ppo_cr_7225 = '/Users/andreimija/ray_results/PPO_cramped_room_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7225/checkpoint_165/checkpoint-165'
    ppo_cr_7649 = '/Users/andreimija/ray_results/PPO_cramped_room_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7649/checkpoint_165/checkpoint-165'

    mappo_cr_2229 = '/Users/andreimija/ray_results/PPO_cramped_room_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=2229_cc/checkpoint_165/checkpoint-165'
    mappo_cr_7225 = '/Users/andreimija/ray_results/PPO_cramped_room_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7225_cc/checkpoint_165/checkpoint-165'
    mappo_cr_7649 = '/Users/andreimija/ray_results/PPO_cramped_room_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7649_cc/checkpoint_165/checkpoint-165'

    ppo_aa_2229 = '/Users/andreimija/ray_results/PPO_asymmetric_advantages_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=2229/checkpoint_165/checkpoint-165'
    ppo_aa_7225 = '/Users/andreimija/ray_results/PPO_asymmetric_advantages_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7225/checkpoint_165/checkpoint-165'
    ppo_aa_7649 = '/Users/andreimija/ray_results/PPO_asymmetric_advantages_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7649/checkpoint_165/checkpoint-165'

    mappo_aa_2229 = '/Users/andreimija/ray_results/PPO_asymmetric_advantages_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=2229_cc/checkpoint_165/checkpoint-165'
    mappo_aa_7225 = '/Users/andreimija/ray_results/PPO_asymmetric_advantages_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7225_cc/checkpoint_165/checkpoint-165'
    mappo_aa_7649 = '/Users/andreimija/ray_results/PPO_asymmetric_advantages_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7649_cc/checkpoint_165/checkpoint-165'

    ppo_cc_2229 = '/Users/andreimija/ray_results/PPO_coordination_ring_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=2229/checkpoint_165/checkpoint-165'
    ppo_cc_7225 = '/Users/andreimija/ray_results/PPO_coordination_ring_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7225/checkpoint_165/checkpoint-165'
    ppo_cc_7649 = '/Users/andreimija/ray_results/PPO_coordination_ring_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7649/checkpoint_165/checkpoint-165'

    mappo_cc_2229 = '/Users/andreimija/ray_results/PPO_coordination_ring_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=2229_cc/checkpoint_165/checkpoint-165'
    mappo_cc_7225 = '/Users/andreimija/ray_results/PPO_coordination_ring_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7225_cc/checkpoint_165/checkpoint-165'
    mappo_cc_7649 = '/Users/andreimija/ray_results/PPO_coordination_ring_False_nw=2_mini_bs=2000_lr=0.001_gamma=0.9_lambda=0.9_vf=0.5_kl=0.2_clip=0.2_sgd=8_2229_2022-05-31_seed=7649_cc/checkpoint_165/checkpoint-165'

    ppo_cr = [ppo_cr_2229, ppo_cr_7225, ppo_cr_7649]
    mappo_cr = [mappo_cr_2229, mappo_cr_7225, mappo_cr_7649]
    ppo_aa = [ppo_aa_2229, ppo_aa_7225, ppo_aa_7649]
    mappo_aa = [mappo_aa_2229, mappo_aa_7225, mappo_aa_7649]
    ppo_cc = [ppo_cc_2229, ppo_cc_7225, ppo_cc_7649]
    mappo_cc = [mappo_cc_2229, mappo_cc_7225, mappo_cc_7649]

    rooms = {
        'cr': {
            'ppo': ppo_cr,
            'mappo': mappo_cr
        },
        'aa': {
            'ppo': ppo_aa,
            'mappo': mappo_aa
        },
        'cc': {
            'ppo': ppo_cc,
            'mappo': mappo_cc
        }
    }

    room_res = {
        'cr': {
            'ppo': [],
            'mappo': [],
        },
        'aa': {
            'ppo': [],
            'mappo': [],
        },
        'cc': {
            'ppo': [],
            'mappo': [],
        }
    }

    for room in rooms:
        print("starting room " + room)
        cc = False
        bc_agent, evaluator = (cr_bc, cr_evaluator) if room == 'cr' else (aa_bc, aa_evaluator) if room == 'aa' else (cc_bc, cc_evaluator)
        for agent_alg_key in rooms[room]:
            print("starting alg " + agent_alg_key + " in room " + room + str(cc))
            agent_alg = rooms[room][agent_alg_key]
            bc_res_mean = [0, 0, 0]
            sp_res_mean = [0, 0, 0]

            for agent_seed in agent_alg:
                agent = load_agent(agent_seed, cc=cc)

                agent_bc_pair = AgentPair(agent, bc_agent)
                agent_pair = load_agent_pair(agent_seed, cc=cc)
                print("starting bc with alg " + agent_alg_key + " in room " + room + str(cc))
                bc_local_res = evaluator.evaluate_agent_pair(agent_bc_pair,
                                              num_games=evaluation_params['num_games'],
                                              display=evaluation_params['display'],
                                              display_phi=evaluation_params[
                                                  'display_phi'])

                avg_rew, se = mean_and_std_err(bc_local_res["ep_returns"])
                std = np.std(bc_local_res["ep_returns"])
                bc_res_mean[0] += avg_rew
                bc_res_mean[1] += std
                bc_res_mean[2] += se
                print("starting sp with alg " + agent_alg_key + " in room " + room + str(cc))
                sp_local_res = evaluator.evaluate_agent_pair(agent_pair,
                                                       num_games=evaluation_params['num_games'],
                                                       display=evaluation_params['display'],
                                                       display_phi=evaluation_params[
                                                           'display_phi'])

                avg_rew, se = mean_and_std_err(sp_local_res["ep_returns"])
                std = np.std(sp_local_res["ep_returns"])
                sp_res_mean[0] += avg_rew
                sp_res_mean[1] += std
                sp_res_mean[2] += se

            room_res[room][agent_alg_key] = [[sp_res / 3 for sp_res in sp_res_mean], [bc_res / 3 for bc_res in bc_res_mean]]
            print(room_res[room][agent_alg_key][0])
            print(room_res[room][agent_alg_key][0][0])
            cc = True

    with open("resultss.json", "w") as write_file:
        json.dump(room_res, write_file, indent=4)

    return room_res


    # PPO_path = '/Users/andreimija/ray_results/PPO_cramped_room_False_nw=2_mini=2k_lr=0.001_gamma=0.9_lambda=0.95_vf=0.5_kl=0.2_clip=0.2_sgd=8/checkpoint_250/checkpoint-250'
    # agent_PPO = load_agent(PPO_path, cc=False)
    #
    # MAPPO_path = '/Users/andreimija/ray_results/PPO_cramped_room_cc_False_nw=2_mini=2k_lr=0.001_gamma=0.9_lambda=0.95_vf=0.5_kl=0.2_clip=0.2_sgd=8/checkpoint_250/checkpoint-250'
    # agent_MAPPO = load_agent(MAPPO_path, cc=True)
    #
    # agent_pairs = [
    #     ('ppo_pair', load_agent_pair(PPO_path, cc=False)),
    #     ('ppo+bc', AgentPair(agent_PPO, bc_agent)),
    #     ('mappo_pair', load_agent_pair(MAPPO_path, cc=True)),
    #     ('mappo+bc', AgentPair(agent_MAPPO, bc_agent)),
    # ]
    #
    # evaluator_results = []
    # for agent_pair in agent_pairs:
    #     evaluator_results.append((agent_pair[0], evaluator.evaluate_agent_pair(agent_pair[1],
    #                                                                            num_games=evaluation_params['num_games'],
    #                                                                            display=evaluation_params['display'],
    #                                                                            display_phi=evaluation_params[
    #                                                                                'display_phi'])))
    #
    # return evaluator_results

if __name__ == '__main__':
    result = evalaute_ultimate()
    simple_plot(result)
    # for result in results:
    #     print(result[1].keys())
    #     print('Avg mean for ' + result[0], np.mean(result[1]['ep_returns']))