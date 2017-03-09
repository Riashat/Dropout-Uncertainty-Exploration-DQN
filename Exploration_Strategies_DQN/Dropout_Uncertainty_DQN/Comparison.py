import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


#taking the final_values onwards, until len(eps)

eps = 2000
eps = range(eps)



Dropout_Boltzmann = np.load('Cum_Rwd_MC_Dropout_Boltzmann0.npy')
Dropout_Epsilon_Greedy = np.load('Cum_Rwd_MCDropout_Mean_EpsilonGreedy_0.npy')



def multiple_plot_episode_stats(stats1, stats2, eps,  smoothing_window=200, noshow=False):


    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="Dropout_Boltzmann")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Dropout_Epsilon_Greedy")    

    plt.legend(handles=[cum_rwd_1, cum_rwd_2 ])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Dropout Uncertainty in DQN - Network (Layer 1, 512 Units, Layer 2, 256 Units)")  
    plt.show()



    return fig



def main():
	multiple_plot_episode_stats(Dropout_Boltzmann, Dropout_Epsilon_Greedy, eps)



if __name__ == '__main__':
	main()


