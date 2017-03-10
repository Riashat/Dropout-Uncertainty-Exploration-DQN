import copy
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import time
from utils import *
from ReplayMemory import ReplayMemory
from agents import AgentEpsGreedy
# from valuefunctions import ValueFunctionDQN
# from valuefunctions import ValueFunctionDQN3
from valuefunctions import ValueFunctionDQN_TEST_TRAIN_DROPOUT
from lib import plotting
from sklearn.model_selection import train_test_split


discount = 0.9
decay_eps = 0.9
batch_size = 64
max_n_ep = 1000    #originally defined! Don't change this.

min_avg_Rwd = 200000000  # Minimum average reward to consider the problem as solved
n_avg_ep = 100      # Number of consecutive episodes to calculate the average reward



env = gym.make("CartPole-v0")    
# env = gym.make("MountainCar-v0")   
n_actions = env.action_space.n
state_dim = env.observation_space.high.shape[0]


def run_episode(env,
                agent,
                state_normalizer,
                memory,
                batch_size,
                discount,
                dropout_probability,
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


    """
    Optimise dropout probability for each episode
    """

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
    
        """
        Perform stochastic forward passses (MC Dropout) here to choose actions
        Choose actions based on epsilon greedy, with Q as the average of the posterior
        Key : Use the dropout probability which is optimised below 
        for Test Time Dropout
        Use the Q posterior here - select actions with highest variance/epsilon_greedy/Boltzmann
        Use selected p to do MC Dropout here
        """

        #do a single stochastic pass for selecting actions
        action = agent.get_action_stochastic_epsilon_greedy(state, dropout_probability)

        #take a, get s' and reward
        state_next, reward, done, info = env.step(action)
        total_reward += reward

        if state_normalizer is not None:
            state_next = state_normalizer.transform(state_next)[0]

        #add trajectory data to memory - for experience replay
        memory.add((state, action, reward, state_next, done))

 

        if len(memory.memory) > batch_size:  # DQN Experience Replay

            """
            Replay memory data
            """
            states_b, actions_b, rewards_b, states_n_b, done_b = zip(*memory.sample(batch_size))
            states_b = np.array(states_b)
            actions_b = np.array(actions_b)
            rewards_b = np.array(rewards_b)
            states_n_b = np.array(states_n_b)
            done_b = np.array(done_b).astype(int)

            """
            Split the set of experimence 
            into training and validation sets 
            Use these sets to evaluate the predictive log likelihood
            - for each p, evaluate the train and valid log likelihood
            - array of validation likelihood values for each p
            - Select the optimal p from here
            """

            actions_b = np.array([actions_b]).T
            rewards_b = np.array([rewards_b]).T
            done_b = np.array([done_b]).T

            #dataset of trajectories - length equal to the batch size for experience replay
            trajectory = np.concatenate((states_b, states_n_b, actions_b, rewards_b, done_b), axis=1)

            """
            Split the data into training and test sets
            """
            train = trajectory[0:batch_size-20, :]
            valid = trajectory[batch_size-20:, :]

            ## if we were to consider random shuffling of train and validation sets
            # msk = np.random.rand(len(trajectory)) < 0.8
            # train = trajectory[msk]
            # valid = trajectory[~msk]

            train_states_b = train[:, 0:4]
            train_states_n_b = train[:, 4:8]
            train_actions_b = train[:, 8]
            train_rewards_b = train[:, 9]
            train_done_b = train[:, 10]

            valid_states_b = valid[:, 0:4]
            valid_states_n_b = valid[:, 4:8]
            valid_actions_b = valid[:, 8]
            valid_rewards_b = valid[:, 9] 
            valid_done_b = valid[:, 10]

            """
            With Batch size 64 : 
            Train states  (44, 4)
            Valid States  (20, 4)
            Train set data (44, 11)
            Valid set data (20, 11)
            """

            p_values = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.01])

            train_loss = np.zeros(shape=len(p_values))
            valid_loss = np.zeros(shape=len(p_values))

            """
            for each value p of dropout probability
            train corresponding model with dropout network p on the train split (we basically maintain multiple Q networks in parallel)

            and then Evaluate each model M on validation set, choose M with best lml and continue to explore with model M

            """
            for p in range(len(p_values)):

                drop_prob = p_values[p]

                train_q_n_b = agent.predict_q_values(train_states_n_b)
                train_targets_b = train_rewards_b + (1. - train_done_b) * discount * np.amax(train_q_n_b, axis=1)
                train_targets = agent.predict_q_values(train_states_b)

                valid_q_n_b = agent.predict_q_values(valid_states_n_b)
                valid_targets_b = valid_rewards_b + (1. - valid_done_b) * discount * np.amax(valid_q_n_b, axis=1)
                valid_targets = agent.predict_q_values(valid_states_b)


                for t, act in enumerate(train_actions_b.astype(int)):
                    train_targets[t, act] = train_targets_b[t]

                for v, act_v in enumerate(valid_actions_b.astype(int)):
                    valid_targets[v, act_v] = valid_targets_b[v]


                """
                We want to do MC Dropout when computing the metric - and average multiple samples together
                """
                #### compute training and validation loss
                ### train_loss_v, _, _, _ = agent.eval_train(train_states_b, train_targets, drop_prob)                
                ### train_loss[p] = train_loss_v

                #### valid_loss_v, _, _, _ = agent.eval_valid(valid_states_b, valid_targets, drop_prob)                
                #### valid_loss[p] = valid_loss_v


                #Applying MC Dropout for computing the metric
                dropout_iterations = 5
                d_all_train_loss = np.array([0])
                d_all_valid_loss_v = np.array([0])

                for d in range(dropout_iterations):
                    d_train_loss_v, _, _, _ = agent.eval_train(train_states_b, train_targets, drop_prob)
                    d_all_train_loss = np.append(d_all_train_loss, d_train_loss_v)

                    d_valid_loss_v, _, _, _ = agent.eval_valid(valid_states_b, valid_targets, drop_prob)   
                    d_all_valid_loss = np.append(d_all_valid_loss_v, d_valid_loss_v)

                mean_d_all_train_loss = np.mean(d_all_train_loss[1:])
                mean_d_all_valid_loss = np.mean(d_all_valid_loss[1:])

                train_loss[p] = mean_d_all_train_loss
                valid_loss[p] = mean_d_all_valid_loss


            # choose M with best |m| and continue to explore with model M
            ind = np.argmin(valid_loss)
            dropout_probability = p_values[ind]

            states_b = trajectory[:, 0:4]
            states_n_b = trajectory[:, 4:8]
            actions_b = trajectory[:, 8]
            rewards_b = trajectory[:, 9]
            done_b = trajectory[:, 10]


            """         
            This Q which is used for the Target Q below
            should use the optimized dropout probability
            We train the Q network wrt to a Target Q
            which uses the optimised dropout probability
            QUESTION : Should the Target Q be also based on average of the Q posterior?
            """
            q_n_b = agent.predict_q_values(states_n_b)  # Action values on the arriving state
            targets_b = rewards_b + (1. - done_b) * discount * np.amax(q_n_b, axis=1)

            #target function for the agent - predict based on the trained Q Network
            targets = agent.predict_q_values(states_b)

            for j, action in enumerate(actions_b.astype(int)):
                targets[j, action] = targets_b[j]
            t_train = time.time()

            #training the agent based on the target function
            loss_v, w1_m, w2_m, w3_m = agent.train(states_b, targets)
            train_duration_s[i - batch_size] = time.time() - t_train



        state = copy.copy(state_next)
        step_durations_s[i] = time.time() - t  # Time elapsed during this step
        step_length = time.time() - t



    return loss_v, w1_m, w2_m, w3_m, total_reward, step_length




max_n_ep = 2000      #number of episodes
#max_step - number of steps within an episode

Experiments = 1
Experiments_All_Rewards = np.zeros(shape=(max_n_ep))


for e in range(Experiments):

    value_function = ValueFunctionDQN_TEST_TRAIN_DROPOUT(state_dim=state_dim, n_actions=n_actions, batch_size=batch_size)
    epsilon = 0.1
    #decay rate for the temperature parameter
    # discount = 0.9

    agent = AgentEpsGreedy(n_actions=n_actions, value_function_model=value_function, eps=epsilon)
    memory = ReplayMemory(max_size=100000)

    loss_per_ep = []
    w1_m_per_ep = []
    w2_m_per_ep = []
    w3_m_per_ep = []
    total_reward = []

    #at the start of episodes
    #initialise the Q network with dropout probability 0.9
    #set current model index M with p=0.9
    dropout_probability = 0.9
    ep = 0
    avg_Rwd = -np.inf
    episode_end_msg = 'loss={:2.10f}, w1_m={:3.1f}, w2_m={:3.1f}, w3_m={:3.1f}, total reward={}'

    stats = plotting.EpisodeStats(episode_lengths=np.zeros(max_n_ep),episode_rewards=np.zeros(max_n_ep))  


    #loop for the number of episodes
    while avg_Rwd < min_avg_Rwd and ep < max_n_ep:
        if ep >= n_avg_ep:
            avg_Rwd = np.mean(total_reward[ep-n_avg_ep:ep])
            print("EPISODE {}. Average reward over the last {} episodes: {}.".format(ep, n_avg_ep, avg_Rwd))
        else:
            print("EPISODE {}.".format(ep))

        """
        Steps within the episode - defined by run_episode
        """
        loss_v, w1_m, w2_m, w3_m, cum_R, step_length = run_episode(env, agent, None, memory, batch_size=batch_size, discount=discount, dropout_probability=dropout_probability, max_step=10000)
        print(episode_end_msg.format(loss_v, w1_m, w2_m, w3_m, cum_R))

        stats.episode_rewards[ep] = cum_R
        stats.episode_lengths[ep] = step_length


        """
        Decaying epsilon parameter - if needed
        """

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

    np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Exploration_DQN/All_Results/'  + 'Cum_Rwd_' + 'Optimized_Dropout_Epsilon_' + str(e) + '.npy', total_reward)

env.close()

print('Saving Average Cumulative Rewards Over Experiments')

Average_Cum_Rwd = np.divide(Experiments_All_Rewards, Experiments)

np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Exploration_DQN/All_Results/'   + 'Average_Cum_Rwd_' + 'Optimized_Dropout_Epsilon_' + '.npy', Average_Cum_Rwd)


print "All Experiments DONE - Deep Q Learning"

print "Optimizing Dropout Probabilities"

