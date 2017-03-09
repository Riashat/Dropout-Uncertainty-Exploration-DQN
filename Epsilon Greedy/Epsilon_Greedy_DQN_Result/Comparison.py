import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


#taking the final_values onwards, until len(eps)

eps = 2000
eps = range(eps)


CartPole_DQN = np.load('CartPole_V0_cumulative_reward_Q_Learning_DQN_Exploration_Param_0.1.npy')
CartPole_DQN_2 = np.load('CartPole_V0_cumulative_reward_Q_Learning_DQN_Exploration_Param_0.3.npy')
CartPole_DQN_3 = np.load('CartPole_V0_cumulative_reward_Q_Learning_DQN_Exploration_Param_0.5.npy')
CartPole_DQN_4 = np.load('CartPole_V0_cumulative_reward_Q_Learning_DQN_Exploration_Param_0.9.npy')




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





def multiple_plot_episode_stats(stats, stats2,  eps,  smoothing_window=50, noshow=False):


    fig = plt.figure(figsize=(30, 20))
    plt.subplot(211)
    rewards_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()
    cum_rwd, = plt.plot(eps, rewards_smoothed, label="Deep Q Learning on Cart Pole, Run 1")    
    plt.legend(handles=[cum_rwd])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("DQN on Cart Pole - Single Run - Larger Network (Layer 1, 512 Units, Layer 2, 256 Units)")  

    plt.subplot(212)
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Deep Q Learning on Cart Pole, Run 2")
    plt.legend(handles=[cum_rwd_2])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.show()

	#higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    # fig = plt.figure(figsize=(20, 10))
    # rewards_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()

    # cum_rwd, = plt.plot(eps, rewards_smoothed, label="Deep Q Learning on Cart Pole")
    # cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Deep Q Learning on Cart Pole")

    # plt.legend(handles=[cum_rwd, cum_rwd_2])
    # plt.xlabel("Epsiode")
    # plt.ylabel("Epsiode Reward (Smoothed)")
    # plt.title("DQN on Cart Pole")
    # plt.show()

    return fig





def same_figure_multiple_plot_episode_stats(stats, stats2, stats3, stats4,  eps,  smoothing_window=50, noshow=False):

    #Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean() 
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd, = plt.plot(eps, rewards_smoothed, label="Epsilon Parameter = 0.1")
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Epsilon Parameter = 0.3 ")
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="Epsilon Parameter = 0.5")
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="Epsilon Parameter = 0.9")


    plt.legend(handles=[cum_rwd, cum_rwd_2, cum_rwd_3, cum_rwd_4])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("DQN on Cart Pole")
    plt.show()

    return fig




def main():
	same_figure_multiple_plot_episode_stats(CartPole_DQN, CartPole_DQN_2, CartPole_DQN_3, CartPole_DQN_4,  eps)



if __name__ == '__main__':
	main()


