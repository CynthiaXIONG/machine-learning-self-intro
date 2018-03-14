#based on https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
# and https://github.com/breeko/Simple-Reinforcement-Learning-with-Tensorflow/blob/master/Part%202%20-%20Policy-based%20Agents.ipynb

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt


def DiscountRewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward 
     e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801] """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in range(r.size - 1, -1, -1): # iterate in reverse order
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

#----------------------------------------------------------
class Agent():
    def __init__(self, lr, s_size, a_size, h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        #The next lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables() #weights?
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss, tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

#----------------------------------------------------------
def do_main():
    env = gym.make('CartPole-v0')

    tf.reset_default_graph() #Clear the Tensorflow graph.

    num_actions = env.action_space.n
    num_states = 4
    hidden_layer_neurons = 8
    my_agent = Agent(lr=1e-2, s_size=num_states, a_size=num_actions, h_size=hidden_layer_neurons) #Load the agent.
    
    ##Training the Agent
    #Learning Parameters
    gamma = 0.99
    num_episodes = 5000 #Set total number of episodes to train agent on.
    max_sim_steps = 999
    update_frequency = 5

    init = tf.global_variables_initializer()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        total_reward = []
        total_lenght = []
            
        grad_buffer = sess.run(tf.trainable_variables())
        for ix, grad in enumerate(grad_buffer):
            grad_buffer[ix] = grad * 0
        
        for i in range(num_episodes):
            s = env.reset()
            running_reward = 0
            ep_history = []

            for j in range(max_sim_steps):
                env.render()

                #Probabilistically pick an action given our network outputs.
                a_dist = sess.run(my_agent.output, feed_dict={my_agent.state_in:[s]})
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)

                s1, r, done, _ = env.step(a) #Get our reward for taking an action for the given simulation state s.
                ep_history.append([s,a,r,s1])
                s = s1
                running_reward += r

                if done:
                    #Update the network.
                    ep_history = np.array(ep_history)
                    ep_history[:, 2] = DiscountRewards(ep_history[:, 2], gamma)
                    feed_dict = {my_agent.reward_holder:ep_history[:, 2], my_agent.action_holder:ep_history[:, 1], my_agent.state_in:np.vstack(ep_history[:,0])}
                    grads = sess.run(my_agent.gradients, feed_dict=feed_dict)

                    for idx, grad in enumerate(grads):
                        grad_buffer[idx] += grad

                    if i % update_frequency == 0 and i != 0:
                        feed_dict = dict(zip(my_agent.gradient_holders, grad_buffer))
                        _ = sess.run(my_agent.update_batch, feed_dict=feed_dict)

                        for ix, grad in enumerate(grad_buffer):
                            grad_buffer[ix] = grad * 0
                    
                    total_reward.append(running_reward)
                    total_lenght.append(j)
                    break

            
                #Update our running tally of scores.
            if i % 100 == 0:
                print(np.mean(total_reward[-100:]))
   

if __name__ == "__main__":
    do_main()
