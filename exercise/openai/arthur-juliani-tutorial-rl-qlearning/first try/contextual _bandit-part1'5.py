#based on https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c
#bandit -> pull slot machine

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class contextual_bandit():
    def __init__(self):
        self.state = 0
        #List out our bandits. Currently arms 4, 2, and 1 (respectively) are the most optimal.
        self.bandits = np.array([[0.2, 0, -0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]
        
    def GetBandit(self):
        self.state = np.random.randint(0, self.num_bandits) #Returns a random state for each episode.
        return self.state
        
    def PullArm(self, action):
        #Get a random number.
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            #return a positive reward.
            return 1
        else:
            #return a negative reward.
            return -1

#----------------------------------------------------------
class agent():
    def __init__(self, lr, s_size, a_size):
        ##Setting up the Agent
        #These two lines established the feed-forward part of the network. This does the actual choosing.
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in, s_size)  #tf-slim, https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim  (simpler setup of tf models)  #one hot encoder: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        output = slim.fully_connected(state_in_OH, a_size, biases_initializer=None, activation_fn=tf.nn.sigmoid, weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder,[1])
        self.loss = -(tf.log(self.responsible_weight) * self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)



#----------------------------------------------------------
def do_main():
    tf.reset_default_graph()

    c_bandit = contextual_bandit() #Load the bandits.
    my_agent = agent(lr=0.001, s_size=c_bandit.num_bandits, a_size=c_bandit.num_actions) #Load the agent
    weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.

    ##Training the Agent
    #Learning Parameters
    num_episodes = 10000 #Set number of episodes to train agent on.
    r_total = np.zeros([c_bandit.num_bandits, c_bandit.num_actions]) #Set scoreboard(reward) for bandits to 0.
    epsilon = 0.1 #Set the chance of taking a random action.

    init = tf.global_variables_initializer()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            s = c_bandit.GetBandit() #Get a state from the environment.

            #Choose either a random action or one from our network.
            if np.random.rand(1) < epsilon:
                action = np.random.randint(c_bandit.num_actions)
            else:
                action = sess.run(my_agent.chosen_action, feed_dict={my_agent.state_in:[s]})
            
            reward = c_bandit.PullArm(action) #Get our reward from picking one of the bandits.
            
            #Update the network.
            feed_dict = {my_agent.reward_holder:[reward], my_agent.action_holder:[action], my_agent.state_in:[s]}
            _, w1 = sess.run([my_agent.update, weights], feed_dict=feed_dict)
            
            #Update our running tally of scores.
            r_total[s, action] += reward
            if i % 500 == 0:
                print("Mean reward for each of the " + str(c_bandit.num_bandits) + " bandits: " + str(np.mean(r_total, axis=1)))


    for a in range(c_bandit.num_bandits):
        print("The agent thinks action " + str(np.argmax(w1[a]) + 1) + " for bandit " + str(a + 1) + " is the most promising....")
        if np.argmax(w1[a]) == np.argmin(c_bandit.bandits[a]):
            print("...and it was right!")
        else:
            print("...and it was wrong!")

if __name__ == "__main__":
    do_main()
