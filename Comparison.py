import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


#taking the final_values onwards, until len(eps)

eps = 2000
eps = range(eps)


Boltzmann = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Exploration_DQN/All_Results/Average_Cum_Rwd_Boltzmann_Exploration_.npy')
Epsilon = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Exploration_DQN/All_Results/Average_Cum_Rwd_Only_Epsilon_Greedy_.npy')



def single_plot_episode_stats(stats, eps,  smoothing_window=50, noshow=False):

    #higher the smoothing window, the better the differences can be seen

    ##Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd, = plt.plot(eps, rewards_smoothed, label="Deep Q Learning on Cart Pole")


    plt.legend(handles=[cum_rwd])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("DQN on Cart Pole - Single Run - Larger Network (Layer 1, 512 Units, Layer 2, 256 Units)")
    plt.show()

    return fig





def multiple_plot_episode_stats(stats1, stats2,  eps,  smoothing_window=200, noshow=False):

    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="Boltzmann Exploration")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Epsilon Greedy Exploration")    

    plt.legend(handles=[cum_rwd_1, cum_rwd_2])

    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Exploration Strategies in DQN")  
    plt.show()

    return fig






def main():
	multiple_plot_episode_stats(Boltzmann, Epsilon, eps)




if __name__ == '__main__':
	main()


