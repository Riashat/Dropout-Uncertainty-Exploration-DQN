'''
Only using training dropout in DQN

Using a 0.5 train time dropout probability

'''

import copy
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import time
from utils import *
from ReplayMemory import ReplayMemory
from agents import AgentEpsGreedy
from valuefunctions import ValueFunctionDQN
from valuefunctions import ValueFunctionDQN3
from valuefunctions import ValueFunctionDQN_TEST_TRAIN_DROPOUT
from valuefunctions import ValueFunctionDQN_TRAIN_DROPOUT
from lib import plotting



discount = 0.9
decay_eps = 0.9
batch_size = 64
max_n_ep = 1000    #originally defined! Don't change this.

min_avg_Rwd = 200000000  # Minimum average reward to consider the problem as solved
n_avg_ep = 100      # Number of consecutive episodes to calculate the average reward


def run_episode(env,
                agent,
                state_normalizer,
                memory,
                batch_size,
                discount,
                max_step=10000):
    state = env.reset()
    if state_normalizer is not None:
        state = state_normalizer.transform(state)[0]
    done = False
    total_reward = 0
    step_durations_s = np.zeros(shape=max_step, dtype=float)
    train_duration_s = np.zeros(shape=max_step-batch_size, dtype=float)
    progress_msg = "Step {:5d}/{:5d}. Avg step duration: {:3.1f} ms. Avg train duration: {:3.1f} ms. Loss = {:2.10f}."
    loss_v = 0
    w1_m = 0
    w2_m = 0
    w3_m = 0
    i = 0
    action = 0    

    #each step within an episode
    for i in range(max_step):
        t = time.time()
        if i > 0 and i % 200 == 0:
            print(progress_msg.format(i, max_step,
                                      np.mean(step_durations_s[0:i])*1000,
                                      np.mean(train_duration_s[0:i-batch_size])*1000,
                                      loss_v))
        if done:
            break
        
        action = agent.act(state)

        #take a, get s' and reward
        state_next, reward, done, info = env.step(action)
        total_reward += reward

        if state_normalizer is not None:
            state_next = state_normalizer.transform(state_next)[0]


        memory.add((state, action, reward, state_next, done))

        if len(memory.memory) > batch_size:  # DQN Experience Replay
            states_b, actions_b, rewards_b, states_n_b, done_b = zip(*memory.sample(batch_size))
            states_b = np.array(states_b)
            actions_b = np.array(actions_b)
            rewards_b = np.array(rewards_b)
            states_n_b = np.array(states_n_b)
            done_b = np.array(done_b).astype(int)


            q_n_b = agent.predict_q_values(states_n_b)  # Action values on the arriving state

            #target - Q-learning here - taking max_a over Q(s', a)
            targets_b = rewards_b + (1. - done_b) * discount * np.amax(q_n_b, axis=1)

            #target function for the agent - predict based on the trained Q Network
            targets = agent.predict_q_values(states_b)
            for j, action in enumerate(actions_b):
                targets[j, action] = targets_b[j]

            t_train = time.time()

            #training the agent based on the target function
            loss_v, w1_m, w2_m, w3_m = agent.train(states_b, targets)
            train_duration_s[i - batch_size] = time.time() - t_train



        state = copy.copy(state_next)
        step_durations_s[i] = time.time() - t  # Time elapsed during this step
        step_length = time.time() - t



    return loss_v, w1_m, w2_m, w3_m, total_reward, step_length





env = gym.make("CartPole-v0")

n_actions = env.action_space.n
state_dim = env.observation_space.high.shape[0]


max_n_ep = 2000      #number of episodes
#max_step - number of steps within an episod

Experiments = 1
Experiments_All_Rewards = np.zeros(shape=(max_n_ep))

# tau is the temperature parameter for Boltzmann Off Policy

for e in range(Experiments):


    #using a smaller network for the value function
    value_function = ValueFunctionDQN_TRAIN_DROPOUT(state_dim=state_dim, n_actions=n_actions, batch_size=batch_size)

    epsilon_parameter = 0.1

    agent = AgentEpsGreedy(n_actions=n_actions, value_function_model=value_function, eps=epsilon_parameter)
    memory = ReplayMemory(max_size=100000)
    

    loss_per_ep = []
    w1_m_per_ep = []
    w2_m_per_ep = []
    w3_m_per_ep = []
    total_reward = []


    ep = 0
    avg_Rwd = -np.inf
    episode_end_msg = 'loss={:2.10f}, w1_m={:3.1f}, w2_m={:3.1f}, w3_m={:3.1f}, total reward={}'

    stats = plotting.EpisodeStats(episode_lengths=np.zeros(max_n_ep),episode_rewards=np.zeros(max_n_ep))  

    while avg_Rwd < min_avg_Rwd and ep < max_n_ep:
        if ep >= n_avg_ep:
            avg_Rwd = np.mean(total_reward[ep-n_avg_ep:ep])
            print("EPISODE {}. Average reward over the last {} episodes: {}.".format(ep, n_avg_ep, avg_Rwd))
        else:
            print("EPISODE {}.".format(ep))


        loss_v, w1_m, w2_m, w3_m, cum_R, step_length = run_episode(env, agent, None, memory, batch_size=batch_size, discount=discount,
                                                      max_step=2000)
        print(episode_end_msg.format(loss_v, w1_m, w2_m, w3_m, cum_R))

        stats.episode_rewards[ep] = cum_R
        stats.episode_lengths[ep] = step_length



        if agent.eps > 0.0001:
             agent.eps *= decay_eps

        # Collect episode results
        loss_per_ep.append(loss_v)
        w1_m_per_ep.append(w1_m)
        w2_m_per_ep.append(w2_m)
        w3_m_per_ep.append(w3_m)
        total_reward.append(cum_R)

        ep += 1

    Experiments_All_Rewards = Experiments_All_Rewards + total_reward
    episode_length_over_time = stats.episode_lengths

    np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Exploration_DQN/All_Results/'  + 'Cum_Rwd_' + 'Only_Train_Dropout_' + str(e) + '.npy', total_reward)

env.close()

print('Saving Average Cumulative Rewards Over Experiments')

Average_Cum_Rwd = np.divide(Experiments_All_Rewards, Experiments)

np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Exploration_DQN/All_Results/'   + 'Average_Cum_Rwd_' + 'Only_Train_Dropout_' +  '.npy', Average_Cum_Rwd)


print "All Experiments DONE - Deep Q Learning"


print "Only Training Dropout DONE"



