import numpy as np
import matplotlib.pyplot as plt


def simple_plot():
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    PPO_PPO = 216
    PPO_BC = 41.4
    MAPPO_MAPPO = 220
    MAPPO_BC = 79.4

    # Set position of bar on X axis
    br1 = np.arange(1)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, PPO_PPO, color='#eaeaea', width=barWidth,
            edgecolor='grey', label='PPO+PPO')
    plt.bar(br2, PPO_BC, color='#F79646', width=barWidth,
            edgecolor='grey', label='PPO+BC')
    plt.bar(br3, MAPPO_MAPPO, color='#4BACC6', width=barWidth,
            edgecolor='grey', label='MAPPO+MAPPO')
    plt.bar(br4, MAPPO_BC, color='#2d6777', width=barWidth,
            edgecolor='grey', label='MAPPO+BC')

    # Adding Xticks
    plt.xlabel('Layout name', fontsize=15)
    plt.xticks([r + barWidth for r in range(1)], ['Cramped rm.'])
    plt.ylabel('Average reward per episode', fontsize=15)
    plt.title('Performance with human proxy model', fontsize=25)

    plt.legend()
    plt.show()
    plt.savefig('evaluation')

if __name__ == '__main__':
    simple_plot()
