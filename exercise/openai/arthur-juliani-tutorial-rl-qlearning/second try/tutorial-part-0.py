import os
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

import rl_utils as rl

# Change dir to this script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def q_table():
    #load environment
    env = gym.make("FrozenLake-v0")

    #initialize table (zeros)
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    
    #setup hyperparameters
    lr = 0.8  #learning rate
    gamma = 0.95 #discount rate, how much we valye future rewards
    num_episodes = 2000
    num_sim_steps = 200    

    #list to store total rewards and steps per episode (debugging)
    j_list = []
    r_list = []

    for i in range(num_episodes):
        #reset env
        s = env.reset()

        r_all = 0
        done = False

        #Q-Table learning algorithm
        for j in range(num_sim_steps):
            
            #Choose action by greedily (with noise) picking from QTable
            #Slowly reduce the random action change as the Q-Table improves
            a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n)  * (1./(1 + i)))

            #Get new state and reward
            s1, r, done, _ = env.step(a)

            #Update Q-Table with new knowledge
            max_future_reward = np.max(Q[s1, :])
            new_estimate = r + gamma * max_future_reward
            Q[s, a] = (1-lr) * Q[s, a] + lr * new_estimate

            r_all += r
            s = s1

            if (done):
                break
        
        #register the rewards
        j_list.append(j)
        r_list.append(r_all)

    #print score
    print("Q-Table success rate: " +  str(sum(r_list)/num_episodes) + "%")

    #plot
    plt.figure(1)

    plt.subplot(211)
    plt.plot(r_list, linewidth=0.3)

    plt.subplot(212)
    plt.plot(j_list, linewidth=0.3)

    plt.show()

def q_nn():
    #load environment
    env = gym.make("FrozenLake-v0")

    #setup tf nn
    tf.reset_default_graph()
    
    state_size = env.observation_space.n
    action_size = env.action_space.n

    #feedforward
    X = tf.placeholder(shape=[1, state_size], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([state_size, action_size], 0, 0.01))
    Q = tf.matmul(X, W)   #Q = Z1 = W.X
    pred = tf.argmax(Q, 1)

    #loss/cost
    Q_target = tf.placeholder(shape=[1, action_size], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(Q_target - Q))

        #backprop
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    #train
    init = tf.global_variables_initializer()
    gamma = 0.99
    e = 0.1
    num_episodes = 2000
    num_sim_steps = 200  

    j_list = []
    r_list = []

    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_episodes):
            #reset
            s = env.reset()
            r_all = 0
            done = False

            for j in range (num_sim_steps):
                #chose an action by greedily (with 'e' change of random action)
                x0 = np.identity(state_size)[s:s+1]
                a, Q_all = sess.run([pred, Q], feed_dict={X:x0})

                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                
                #get new state
                s1, r, done, _ = env.step(a[0])

                #obtain Q' by feeding the new state
                x1 = np.identity(state_size)[s1:s1+1]
                Q1 = sess.run(Q, feed_dict={X:x1})

                #obtain max Q'
                max_Q1 = np.max(Q1)
                target_Q = Q_all
                target_Q[0, a[0]] = r + gamma * max_Q1

                #train NN using the target and predicted Q
                _, W1 = sess.run([optimizer, W], feed_dict={X:x0, Q_target:target_Q})

                r_all += r
                s = s1

                if (done):
                    #reduce chance of random choice
                    e = 1./((i/50) + 10)
                    break

            j_list.append(j)
            r_list.append(r_all)

    print("Q-NN success rate: " +  str(sum(r_list)/num_episodes) + "%")
    plt.figure(1)

    plt.subplot(211)
    plt.plot(r_list, linewidth=0.3)

    plt.subplot(212)
    plt.plot(j_list, linewidth=0.3)

    plt.show()

def qtable_from_rlutils():
        #load environment
    env = gym.make("FrozenLake-v0")

    #initialize
    q_table = rl.QTable(env.observation_space.n, env.action_space.n, epsilon=0.3)
    
    #setup hyperparameters
    num_episodes = 2000
    num_sim_steps = 200    

    #list to store total rewards and steps per episode (debugging)
    j_list = []
    r_list = []

    for i in range(num_episodes):
        #reset env
        s = env.reset()

        r_all = 0
        done = False

        #Q-Table learning algorithm
        for j in range(num_sim_steps):
            
            #Choose action by greedily (with noise) picking from QTable
            #Slowly reduce the random action change as the Q-Table improves
            a = q_table.get_action(s, episode=i)

            #Get new state and reward
            s1, r, done, _ = env.step(a)

            #Update Q-Table with new knowledge
            q_table.update_q(s=s, a=a, s1=s1, r=r)

            r_all += r
            s = s1

            if (done):
                break
        
        #register the rewards
        j_list.append(j)
        r_list.append(r_all)

    #print score
    print("Q-Table (rl-utils) success rate: " +  str(sum(r_list)/num_episodes) + "%")

    #plot
    plt.figure(1)

    plt.subplot(211)
    plt.plot(r_list, linewidth=0.3)

    plt.subplot(212)
    plt.plot(j_list, linewidth=0.3)

    plt.show()


if __name__ == "__main__":

    #q_table()
    #q_nn()
    qtable_from_rlutils()