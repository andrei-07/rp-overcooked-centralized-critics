import os
from human_aware_rl.rllib.rllib_maddpg import load_agent, load_agent_pair
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

if __name__ == '__main__':
    results = evluate_advanced_maddpg_pair()
    print(results.keys())
    print(np.mean(results['ep_returns']))
    #evaluate_maddpg()