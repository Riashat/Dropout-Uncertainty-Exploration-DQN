import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


#taking the final_values onwards, until len(eps)

eps = 2000
eps = range(eps)


EpsGreedy = np.load('Cum_Rwd_Only_Epsilon_Greedy_0.npy')
Boltzmann = np.load('Cum_Rwd_Only_Boltzmann_0.npy')
Random = np.load('Cum_Rwd_Only_Random.npy')
Only_Dropout = np.load('Cum_Rwd_Only_Train_Dropout_0.npy')
Dropout_Boltzmann = np.load('Cum_Rwd_MC_Dropout_Boltzmann0.npy')
Dropout_Epsilon_Greedy = np.load('Cum_Rwd_MCDropout_Mean_EpsilonGreedy_0.npy')



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





def multiple_plot_episode_stats(stats1, stats3,  stats4, stats5, stats6, eps,  smoothing_window=200, noshow=False):


    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_6 = pd.Series(stats6   ).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="Epsilon Greedy")    
    # cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Dropout in DQN")    
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="Boltzmann") 
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="Random") 
    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, label="Dropout + Boltzmann") 
    cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, label="Dropout + Epsilon Greedy") 

    plt.legend(handles=[cum_rwd_1, cum_rwd_3, cum_rwd_4, cum_rwd_5, cum_rwd_6])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Exploration Strategies in DQN - Larger Network (Layer 1, 512 Units, Layer 2, 256 Units)")  
    plt.show()



    return fig




def multiple_plot_episode_stats_all(stats1, stats2, stats3,  stats4, stats5, stats6, eps,  smoothing_window=200, noshow=False):


    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_6 = pd.Series(stats6   ).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="Epsilon Greedy")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Dropout in DQN")    
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="Boltzmann") 
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="Random") 
    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, label="Dropout + Boltzmann") 
    cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, label="Dropout + Epsilon Greedy") 

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5, cum_rwd_6])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Exploration Strategies in DQN - Larger Network (Layer 1, 512 Units, Layer 2, 256 Units)")  
    plt.show()



    return fig






def main():
	multiple_plot_episode_stats(EpsGreedy, Boltzmann, Random, Dropout_Boltzmann, Dropout_Epsilon_Greedy, eps)
    #multiple_plot_episode_stats_all(EpsGreedy, Only_Dropout, Boltzmann, Random, Dropout_Boltzmann, Dropout_Epsilon_Greedy, eps)
    #single_plot_episode_stats(Dropout_Boltzmann, eps)



if __name__ == '__main__':
	main()


