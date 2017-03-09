import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


#taking the final_values onwards, until len(eps)

eps = 2000
eps = range(eps)



MCDropout_Epsilon = np.load('Average_Cum_Rwd_MCDropout_EpsilonGreedy_.npy')
MCDropout_Boltzmann = np.load('Average_Cum_Rwd_MC_Dropout_Boltzmann.npy')
Boltzmann = np.load('Average_Cum_Rwd_Only_Boltzmann.npy')
Epsilon = np.load('Average_Cum_Rwd_Only_Epsilon_Greedy_.npy')
OnlyDropout = np.load('Average_Cum_Rwd_Only_Train_Dropout_.npy')





def single_average(stats, eps,  smoothing_window=50, noshow=False):

    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd, = plt.plot(eps, rewards_smoothed, label="Boltzmann Exploration (Averaged)")

    plt.legend(handles=[cum_rwd])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Boltzmann Exploration - Averaged Results")
    plt.show()

    return fig



def compare_runs(stats, stats1, stats2,  eps,  smoothing_window=50, noshow=False):

    #higher the smoothing window, the better the differences can be seen

    ##Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
 
    cum_rwd, = plt.plot(eps, rewards_smoothed, label="Run 1")
    cum_rwd1, = plt.plot(eps, rewards_smoothed1, label="Run 2")
    cum_rwd2, = plt.plot(eps, rewards_smoothed2, label="Run 3")


    plt.legend(handles=[cum_rwd, cum_rwd1, cum_rwd2])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Boltzmann Exploration")
    plt.show()

    return fig



def comparing_exploration(stats1, stats2,  stats3, stats4, stats5, eps,  smoothing_window=200, noshow=False):


    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="MC Dropout + Epsilon Greedy")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="MC Dropout + Boltzmann")    
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="Boltzmann") 
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="Epsilon Greedy") 
    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, label="Only Dropout") 


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Exploration Strategies in DQN")  
    plt.show()



    return fig





def main():

    #compare_runs(Boltzmann, Boltzmann1, Boltzmann2, eps)

    # single_average(Boltzmann_Average, eps)

    comparing_exploration(MCDropout_Epsilon, MCDropout_Boltzmann, Boltzmann, Epsilon, OnlyDropout, eps)





if __name__ == '__main__':
	main()


