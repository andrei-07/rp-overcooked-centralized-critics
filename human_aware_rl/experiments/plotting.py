import numpy as np
import matplotlib.pyplot as plt


def simple_plot(result):
    # result = {
    #     'cr': {
    #         'ppo': [[200, 10],[20, 5]],
    #         'mappo': [[200, 10],[40, 5]]
    #     },
    #     'aa': {
    #         'ppo': [[200, 10],[20, 5]],
    #         'mappo': [[200, 10],[40, 5]]
    #     }
    # }

    layouts = ['Cramped rm.', 'Asymm. Adv.']

    # set width of bar
    barWidth = 0.2
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    PPO_PPO = [result['cr']['ppo'][0][0], result['aa']['ppo'][0][0]]
    PPO_PPO_std = [result['cr']['ppo'][0][1], result['aa']['ppo'][0][1]]

    PPO_BC = [result['cr']['ppo'][1][0], result['aa']['ppo'][1][0]]
    PPO_BC_std = [result['cr']['ppo'][1][1], result['aa']['ppo'][1][1]]

    MAPPO_MAPPO = [result['cr']['mappo'][0][0], result['aa']['mappo'][0][0]]
    MAPPO_MAPPO_std = [result['cr']['mappo'][0][1], result['aa']['mappo'][0][1]]

    MAPPO_BC = [result['cr']['mappo'][1][0], result['aa']['mappo'][1][0]]
    MAPPO_BC_std = [result['cr']['mappo'][1][1], result['aa']['mappo'][1][1]]

    # Set position of bar on X axis
    br1 = np.arange(len(layouts))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, PPO_PPO, yerr=PPO_PPO_std, color='#eaeaea', width=barWidth,
            edgecolor='grey', label='PPO+PPO')
    plt.bar(br2, PPO_BC, yerr=PPO_BC_std, color='#F79646', width=barWidth,
            edgecolor='grey', label='PPO+BC')
    plt.bar(br3, MAPPO_MAPPO, yerr=MAPPO_MAPPO_std, color='#4BACC6', width=barWidth,
            edgecolor='grey', label='MAPPO+MAPPO')
    plt.bar(br4, MAPPO_BC, yerr=MAPPO_BC_std, color='#2d6777', width=barWidth,
            edgecolor='grey', label='MAPPO+BC')

    # Adding Xticks
    plt.xlabel('Layout name', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(layouts))], layouts)
    plt.ylabel('Average reward per episode', fontsize=15)
    plt.title('Performance with human proxy model', fontsize=25)

    plt.legend()
    plt.show()
    plt.savefig('evaluation')

if __name__ == '__main__':
    simple_plot({})
