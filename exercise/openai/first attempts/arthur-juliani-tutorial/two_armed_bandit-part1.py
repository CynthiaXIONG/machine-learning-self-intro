#based on https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
#bandit -> pull slot machine

import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

#List out our bandits. Currently bandit 4 (index#3) is set to most often provide a positive reward.
bandits = [0.2, -0.2, 0, -5]
num_bandits = len(bandits)

def PullBandit(bandit):
    #get rand number
    result = np.random.randn(1)
    if result > bandits[bandit]:
        return 1
    else:
        return -1


#----------------------------------------------------------
def do_main():
    tf.reset_default_graph()

    ##Setting up the Agent
    #These two lines established the feed-forward part of the network. This does the actual choosing.
    weights = tf.Variable(tf.ones([num_bandits]))
    chosen_action = tf.argmax(weights, 0) #chose action that yields the higher reward

    #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
    #to compute the loss, and use it to update the network.
    reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
    action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
    responsible_weight = tf.slice(weights, action_holder, [1]) #extract the weight of the slice/ action_holder
    loss = -(tf.log(responsible_weight) * reward_holder) #policy loss eq: allows increase the   weight of actions with positive reward, and vice-versa
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    update_model = optimizer.minimize(loss)

    ##Training the Agent
    #Learning Parameters
    num_episodes = 1000 #Set number of episodes to train agent on.
    r_total = np.zeros(num_bandits) #Set scoreboard(reward) for bandits to 0.
    epsilon = 0.1 #Set the chance of taking a random action.

    init = tf.global_variables_initializer()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            
            #Choose either a random action or one from our network.
            if np.random.rand(1) < epsilon:
                action = np.random.randint(num_bandits)
            else:
                action = sess.run(chosen_action)
            
            reward = PullBandit(action) #Get our reward from picking one of the bandits.
            
            #Update the network.
            _, resp, w1 = sess.run([update_model, responsible_weight, weights], feed_dict={reward_holder:[reward], action_holder:[action]})
            
            #Update our running tally of scores.
            r_total[action] += reward
            if i % 50 == 0:
                print("Running reward for the " + str(num_bandits) + " bandits: " + str(r_total))

    print("The agent thinks bandit " + str(np.argmax(w1) + 1) + " is the most promising....")
    if np.argmax(w1) == np.argmax(-np.array(bandits)):
        print("...and it was right!")
    else:
        print("...and it was wrong!")

if __name__ == "__main__":
    do_main()
