#based on https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from datetime import datetime

#----------------------------------------------------------
def do_main():
    env = gym.make('FrozenLake-v0')

    ###QLearning - Table Implementation
    #Initialize table with all zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Set learning parameters
    lr = .8 #learning rate, 
    gamma = .95  #discount rate, how much we value future reward
    num_simulations = 2000
    num_sim_steps = 100

    #create lists to contain total rewards and steps per simulation
    j_list = []
    r_list = []
    
    startTime = datetime.now()
    for i in range(num_simulations):
        #Reset environment and get first new observation
        s = env.reset()
        r_total = 0
        j = 0

        #The Q-Table learning algorithm
        for j in range(num_sim_steps):
            #env.render()
            
            #Choose an action by greedily (with noise, so we dont get stuck on local maximums) picking from Q table
            #Slowly reduce the random action change as the Q-Table improves
            a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
            
            #Get new state and reward from environment
            s1, r, done, _ = env.step(a)

            #Update/Improve Q-Table with new knowledge
            max_future_reward = np.max(Q[s1, :]) # get the max reward using the QTable data for the next state s1
            improved_estimation = r + gamma * max_future_reward #improve estimation reward, adding the future reward parameter as 
            Q[s, a] = (1-lr) * Q[s,a] + lr *  improved_estimation #Update q_table with the new reward

            #Update state
            r_total += r
            s = s1

            if (done):
                break
        
        #append simulation end data
        j_list.append(j)
        r_list.append(r_total)
    
    print("Q-Table success rate: " +  str(sum(r_list)/num_simulations) + "%")
    print("Time taken:", datetime.now() - startTime)

    plt.figure(1)

    plt.subplot(211)
    plt.plot(r_list, linewidth=0.3)

    plt.subplot(212)
    plt.plot(j_list, linewidth=0.3)

    plt.show()

    ###QLearning - NeuralNetwork Implementation
    ##Implementing the Network
    #one layer network, 1x16 (states) and 4 Q-value outputs, one for each action
    tf.reset_default_graph()
    layer_0_dim = env.observation_space.n
    output_dim = env.action_space.n

    #These lines establish the feed-forward part of the network used to choose actions
    inputs1 = tf.placeholder(shape=[1, layer_0_dim], dtype=tf.float32) #inputs are the possible states
    W = tf.Variable(tf.random_uniform([layer_0_dim, output_dim], 0, 0.01))
    
    with tf.device('/gpu:0'): #'/fpu:0' '/cpu:0' #force GPU/CPU, only for NVidia
        Q_out = tf.matmul(inputs1, W)
        predict = tf.argmax(Q_out, 1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        #TargetQ is equivalent to the best possible nextQ (reusing the leaner model to predict the future, like in the Table version)  
        next_Q = tf.placeholder(shape=[1, output_dim], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(next_Q - Q_out))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        update_model = trainer.minimize(loss)

    ##Training the Network
    init = tf.global_variables_initializer()

    # Set learning parameters
    gamma = .99
    epsilon = 0.1
    num_simulations = 2000

    #create lists to contain total rewards and steps per simulation
    j_list = []
    r_list = []

    startTime = datetime.now()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)
        for i in range(num_simulations):
            #Reset environment and get first new observation
            s = env.reset()
            r_total = 0
            j = 0

            #The Q-Network learning algorithm
            for j in range(num_sim_steps):
                #Choose an action by greedily (with epsilon chance of random action) from the Q-network
                a, all_Q = sess.run([predict, Q_out], feed_dict={inputs1:np.identity(layer_0_dim)[s:s+1]})
                if np.random.rand(1) < epsilon: #random change is reduced per iteration later on
                    a[0] = env.action_space.sample() 

                #Get new state and reward from environment
                s1, r , done, _ = env.step(a[0])

                #Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Q_out, feed_dict={inputs1:np.identity(layer_0_dim)[s1:s1+1]})

                #Obtain maxQ' and set our target value for chosen action. (as best possible future action)
                max_Q1 = np.max(Q1)
                target_Q = all_Q
                target_Q[0, a[0]] = r + gamma * max_Q1

                #Train our network using target and predicted Q values
                _, W1 = sess.run([update_model, W], feed_dict={inputs1:np.identity(layer_0_dim)[s:s+1], next_Q:target_Q})
                r_total += r
                s = s1

                if (done):
                    #Reduce chance of random action as we train the model.
                    epsilon = 1./((i/50) + 10)
                    break

           #append simulation end data
            j_list.append(j)
            r_list.append(r_total)
        
    print("Q-Network success rate: " +  str(sum(r_list)/num_simulations) + "%")
    print("Time taken:", datetime.now() - startTime)

    plt.figure(1)

    plt.subplot(211)
    plt.plot(r_list, linewidth=0.3)

    plt.subplot(212)
    plt.plot(j_list, linewidth=0.3)

    plt.show()

if __name__ == "__main__":
    do_main()
