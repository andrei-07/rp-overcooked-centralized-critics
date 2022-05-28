import os
from human_aware_rl.rllib.rllib import load_agent, load_agent_pair, RlLibAgent
from human_aware_rl.imitation.behavior_cloning_tf2 import load_bc_model, BehaviorCloningPolicy
import numpy as np
#from human_aware_rl.rllib.rllib import load_agent
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair
from human_aware_rl.rllib.utils import get_base_ae

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

path_to_agent = '/Users/andreimija/ray_results/MADDPG_cramped_room_True_nw=30_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_0_2022-05-14_01-23-466zunfh0r/checkpoint_116476/checkpoint-116476'

def evaluate_maddpg():
    # get maddpg agent
    # get maddpg agent2
    path_to_agent1 = '/Users/andreimija/ray_results/MADDPG_cramped_room_True_nw=2_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_0_2022-05-07_17-11-33wllq4j_f/checkpoint_2/checkpoint-2'
    agent1 = load_agent(path_to_agent1)

    path_to_agent2 = '/Users/andreimija/ray_results/MADDPG_cramped_room_True_nw=2_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_0_2022-05-07_17-40-24sko6xdlt/checkpoint_2/checkpoint-2'
    agent2 = load_agent(path_to_agent2)
    # create evlautor
    evaluator = AgentEvaluator.from_layout_name(mdp_params={"layout_name" : "cramped_room"}, env_params={"horizon" : 400})
    # evaluate agent pair
    # print some data
    return evaluator.evaluate_agent_pair(AgentPair(agent1, agent2), 10)

def evaluate_maddpg_pair():
    agent_pair = load_agent_pair(path_to_agent)
    # create evlautor
    evaluator = AgentEvaluator.from_layout_name(mdp_params={"layout_name": "cramped_room"}, env_params={"horizon": 400})
    # evaluate agent pair
    # print some data
    return evaluator.evaluate_agent_pair(agent_pair, 100)

def evluate_advanced_maddpg_pair():
    agent_pair = load_agent_pair(path_to_agent)

    evaluator = get_base_ae(mdp_params, {"horizon": evaluation_params['ep_length'], "num_mdp": 1}, evaluation_params['outer_shape'])
    return evaluator.evaluate_agent_pair(agent_pair,
                                            num_games=evaluation_params['num_games'],
                                            display=evaluation_params['display'],
                                            display_phi=evaluation_params['display_phi'])

def _get_base_ae(bc_params):
    return get_base_ae(bc_params['mdp_params'], bc_params['env_params'])

def evaluate_with_human():
    dense_path = 'PPO_cramped_room_tbs=12k_mbs=2000_iter=300_rsh=inf_rsf=0=sparse'
    anneal_path = 'PPO_cramped_room_tbs=12k_mbs=2000_iter=300_rsh=2.5e6_rsf=1=anneal'
    sparse_path = 'PPO_cramped_room_tbs=12k_mbs=2000_iter=800_rsh=inf_rsf=1=dense'

    bc_agent_model, bc_params = load_bc_model('/Users/andreimija/Documents/University/human_aware_rl/human_aware_rl/imitation/bc_runs/default')
    agent_0_policy = BehaviorCloningPolicy.from_model(bc_agent_model, bc_params, stochastic=True)

    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env

    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)

    bc_agent = RlLibAgent(agent_0_policy, 1, featurize_fn=featurize_fn)

    ppo_agent_dense = load_agent('/Users/andreimija/ray_results/PPO_cramped_room_True_nw=4_vf=0.500000_es=0.200000_en=0.100000_kl=0.200000_0_2022-05-19_11-41-16trh09i73/checkpoint_800/checkpoint-800', policy_id='ppo')

    ppo_agent_sparse = load_agent('/Users/andreimija/ray_results/PPO_cramped_room_tbs=12k_mbs=2000_iter=300_rsh=inf_rsf=0/checkpoint_676/checkpoint-676', policy_id='ppo')

    ppo_agent_anneal = load_agent('/Users/andreimija/ray_results/PPO_cramped_room_tbs=12k_mbs=2000_iter=300_rsh=2.5e6_rsf=1/checkpoint_800/checkpoint-800', policy_id='ppo')

    evaluator = get_base_ae(mdp_params, {"horizon": evaluation_params['ep_length'], "num_mdp": 1},
                            evaluation_params['outer_shape'])

    # evaluator = get_base_ae(bc_params["mdp_params"], {"horizon": evaluation_params['ep_length'], "num_mdp": 1},
    #                         evaluation_params['outer_shape'])

    ppo_dense_pair = load_agent_pair('/Users/andreimija/ray_results/PPO_cramped_room_True_nw=4_vf=0.500000_es=0.200000_en=0.100000_kl=0.200000_0_2022-05-19_11-41-16trh09i73/checkpoint_800/checkpoint-800')
    ppo_anneal_pair = load_agent_pair('/Users/andreimija/ray_results/PPO_cramped_room_tbs=12k_mbs=2000_iter=300_rsh=2.5e6_rsf=1/checkpoint_800/checkpoint-800')
    ppo_sparse_pair = load_agent_pair('/Users/andreimija/ray_results/PPO_cramped_room_tbs=12k_mbs=2000_iter=300_rsh=inf_rsf=0/checkpoint_676/checkpoint-676')

    agent_pairs = [('dense_PPO+BC', AgentPair(ppo_agent_dense, bc_agent)), ('anneal_PPO+BC', AgentPair(ppo_agent_anneal, bc_agent)), ('sparse_PPO+BC', AgentPair(ppo_agent_sparse, bc_agent)),
                   ('dense_PPO', ppo_dense_pair), ('anneal_PPO', ppo_anneal_pair), ('sparse_PPO', ppo_sparse_pair)]

    evaluator_results = []
    for agent_pair in agent_pairs:
        evaluator_results.append((agent_pair[0], evaluator.evaluate_agent_pair(agent_pair[1],
                                         num_games=evaluation_params['num_games'],
                                         display=evaluation_params['display'],
                                         display_phi=evaluation_params['display_phi'])))
    return evaluator_results

def evaluate_final():
    evaluator = get_base_ae(mdp_params, {"horizon": evaluation_params['ep_length'], "num_mdp": 1},
                            evaluation_params['outer_shape'])

    bc_agent_model, bc_params = load_bc_model(
        '/Users/andreimija/Documents/University/human_aware_rl/human_aware_rl/imitation/bc_runs/default')
    agent_0_policy = BehaviorCloningPolicy.from_model(bc_agent_model, bc_params, stochastic=True)

    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env

    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)

    bc_agent = RlLibAgent(agent_0_policy, 1, featurize_fn=featurize_fn)

    PPO_path = '/Users/andreimija/ray_results/PPO_cramped_room_tbs=12k_mbs=2000_iter=500_rsh=inf_rsf=1=dense_no_phi_bc_features/checkpoint_500/checkpoint-500'
    agent_PPO = load_agent(PPO_path, cc=False)

    MAPPO_path = '/Users/andreimija/ray_results/PPO_cramped_room_cc_False_nw=2_vf=0.5_es=0.4_en=0.1_kl=0.2_dense_customCC/checkpoint_500/checkpoint-500'
    agent_MAPPO = load_agent(MAPPO_path, cc=True)

    MAPPO_anneal_path = '/Users/andreimija/ray_results/PPO_cramped_room_cc_False_nw=2_vf=0.5_es=0.4_en=0.1_kl=0.2_anneal?_normal_critics/checkpoint_500/checkpoint-500'
    agent_MAPPO_anneal = load_agent(MAPPO_path, cc=True)

    MAPPO_dense2_path = '/Users/andreimija/ray_results/PPO_cramped_room_cc_False_nw=2_vf=0.5_es=0.5_en=0.1_kl=0.2_dense2/checkpoint_500/checkpoint-500'
    agent_dense2_anneal = load_agent(MAPPO_path, cc=True)

    agent_pairs = [
        ('ppo_pair', load_agent_pair(PPO_path, cc=False)),
        ('ppo+bc', AgentPair(agent_PPO, bc_agent)),
        ('mappo_pair', load_agent_pair(MAPPO_path, cc=True)),
        ('mappo+bc', AgentPair(agent_MAPPO, bc_agent)),
        ('mappo_anneal_pair', load_agent_pair(MAPPO_anneal_path, cc=True)),
        ('mappo_anneal+bc', AgentPair(agent_MAPPO_anneal, bc_agent)),
        ('mappo_dense2_pair', load_agent_pair(MAPPO_dense2_path, cc=True)),
        ('mappo_dense2+bc', AgentPair(MAPPO_dense2_path, bc_agent)),
    ]

    evaluator_results = []
    for agent_pair in agent_pairs:
        evaluator_results.append((agent_pair[0], evaluator.evaluate_agent_pair(agent_pair[1],
                                                                               num_games=100,
                                                                               display=evaluation_params['display'],
                                                                               display_phi=evaluation_params[
                                                                                   'display_phi'])))
    return evaluator_results

if __name__ == '__main__':
    results = evaluate_final()
    for result in results:
        print(result[1].keys())
        print('Avg mean for ' + result[0], np.mean(result[1]['ep_returns']))