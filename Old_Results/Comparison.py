import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


#taking the final_values onwards, until len(eps)

eps = 2000
eps = range(eps)


Dropout_Boltzmann = np.load('Average_Cum_Rwd_MC_Dropout_Boltzmann.npy')
Dropout_EpsilonGreedy = np.load('Average_Cum_Rwd_Dropout_EpsilonGreedy_.npy')
Dropout_Highest_Variance = np.load('Average_Cum_Rwd_Dropout_Highest_Variance_.npy')
Dropout_Higehst_Variance_Decaying_Prob = np.load('Average_Cum_Rwd_Dropout_Highest_Variance_Decaying_Probability.npy')

Dropout_Thompson = np.load('Average_Cum_Rwd_Dropout_Thompson_.npy')

EpsilonGreedy = np.load('Average_Cum_Rwd_Only_Epsilon_Greedy_.npy')
Boltzmann = np.load('Average_Cum_Rwd_Boltzmann_Exploration_.npy')
Random = np.load('Average_Cum_Rwd_Only_Random.npy')
Thompson = np.load('Average_Cum_Rwd_Only_Thompson_.npy')



# def single_average(stats, eps,  smoothing_window=50, noshow=False):

#     fig = plt.figure(figsize=(20, 10))
#     rewards_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()

#     cum_rwd, = plt.plot(eps, rewards_smoothed, label="Boltzmann Exploration (Averaged)")

#     plt.legend(handles=[cum_rwd])
#     plt.xlabel("Epsiode")
#     plt.ylabel("Epsiode Reward (Smoothed)")
#     plt.title("Boltzmann Exploration - Averaged Results")
#     plt.show()

#     return fig



def compare_runs(stats0, stats1, stats2, stats3, stats4, stats5,  eps,  smoothing_window=200, noshow=False):

    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed0 = pd.Series(stats0).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd0, = plt.plot(eps, rewards_smoothed0, label="MC Dropout Maximum Variance")
    cum_rwd1, = plt.plot(eps, rewards_smoothed1, label="MC Dropout with Boltzmann Exploration")
    cum_rwd2, = plt.plot(eps, rewards_smoothed2, label="MC Dropout with Epsilon Greedy Exploration")
    cum_rwd3, = plt.plot(eps, rewards_smoothed3, label="MC Dropout with Thompson Sampling")
    cum_rwd4, = plt.plot(eps, rewards_smoothed4, label="Epsilon Greedy Exploration")
    cum_rwd5, = plt.plot(eps, rewards_smoothed5, label="Boltzmann Exploration")


    plt.legend(handles=[cum_rwd0, cum_rwd1, cum_rwd2, cum_rwd3, cum_rwd4, cum_rwd5])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Exploration Strategies in DQN with CartPole MDP")
    plt.show()

    return fig



def comparing_exploration(stats1, stats2,  stats3, stats4, stats5, stats6, stats7, stats8, stats9, eps,  smoothing_window=100, noshow=False):


    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_6 = pd.Series(stats6).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_7 = pd.Series(stats7).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_8 = pd.Series(stats8).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_9 = pd.Series(stats9).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="MC Dropout + Boltzmann")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="MC Dropout + Epsilon Greedy")    
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="MC Dropout + Highest Variance") 
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="MC Dropout + Thompson Sampling") 
    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, label="Epsilon Greedy") 
    cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, label="Boltzmann Exploration") 
    cum_rwd_7, = plt.plot(eps, rewards_smoothed_7, label="Random exploration") 
    cum_rwd_8, = plt.plot(eps, rewards_smoothed_8, label="Thompson Sampling Exploration") 
    cum_rwd_9, = plt.plot(eps, rewards_smoothed_9, label="MC Dropout Decaying Prob + Highest Variance") 


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5, cum_rwd_6, cum_rwd_7, cum_rwd_8, cum_rwd_9])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Exploration Strategies in DQN")  
    plt.show()



    return fig







def main():

    compare_runs(Dropout_Highest_Variance, Dropout_Boltzmann,  Dropout_EpsilonGreedy, Dropout_Thompson, EpsilonGreedy, Boltzmann, eps)

    #single_average(Dropout_Boltzmann, eps)



    #comparing_exploration( Dropout_Boltzmann, Dropout_EpsilonGreedy, Dropout_Highest_Variance, Dropout_Thompson, EpsilonGreedy, Boltzmann, Random, Thompson, Dropout_Higehst_Variance_Decaying_Prob, eps)





if __name__ == '__main__':
	main()


