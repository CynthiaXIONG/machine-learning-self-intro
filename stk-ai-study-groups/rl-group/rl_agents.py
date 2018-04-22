#code style: https://google.github.io/styleguide/pyguide.html

import numpy as np

class MonteCarloControlAgent():
    # First-Visit MC, based on https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html
    def __init__(self, observation_space, gamma=0.99, hash_s_function=None, exploration_function=None):
        self.observation_space = observation_space
        self.action_space = observation_space[0]
        self.state_space = observation_space[1]
        self.gamma = gamma

        self.hash_s_function = hash_s_function #used to hash s (e.g: if multidimensional or continuous)

        if (exploration_function is not None):
            self.exploration_function = exploration_function
        else:
            self.exploration_function = lambda i : i == 0

        # Initialize Q and Pi with random values
        self.q_matrix = np.zeros([self.action_space, self.state_space]) #np.random.random_sample([self.action_space, self.state_space])
        self.policy_matrix = np.random.randint(low=0, high=self.action_space, size=self.state_space).astype(np.int32)

        # Init visits counter with 1.0e-10 to avoid division by zero
        self.visits_counter_matrix = np.full(self.q_matrix.shape, 1.0e-10)
        pass
        
    def get_action(self, s, get_random=False):
        if (get_random):
            action = np.random.randint(0, self.action_space)
        else:
            # Get from policy
            action = self.policy_matrix[s]
        return action
    
    def get_control_state_return(self, state_list, gamma=None):
        # Returns utility/return using Bellmans discounted reward
        # state_list is tuple (position, action, reward)
        if (gamma is None):
            gamma = self.gamma

        ret_value = 0
        for i in range(len(state_list)):
            reward = state_list[i][2]
            ret_value += reward * np.power(gamma, i)
        return ret_value

    def get_mean_return_matrix(self):
        return self.q_matrix / self.visits_counter_matrix

    def initialize_epoch(self):
        self.episode_list = list()

    def append_episode(self, observation, action, reward):
        self.episode_list.append((observation, action, reward))

    def get_latest_episode(self):
        return self.episode_list[-1]

    def execute_epoch(self, nof_steps, env, s0, render_function=None):
        s = s0
        for i in range(nof_steps):
            if (render_function is not None):
                render_function()

            # Get Hashed state
            if (self.hash_s_function is not None):
                s = self.hash_s_function(s)
            # Get action
            action = self.get_action(s, get_random=self.exploration_function(i))
            # Move one step in the environment and get new state and reward
            s1, reward, done = env.step(action)[:3] #just get the first three, for compatibility with both blog GridWorld and openai gym
            # Append the visit in the episode list
            self.append_episode(s, action, reward)

            s = s1
            if done: break
        
        #episode done, estimate new utility and update policy
        self.estimate_utility()
        self.policy_update()
    
    def estimate_utility(self):
        # This cycle is the implementation of First-Visit MC.
        # This is the Evaluation step of the GPI (Generalized Policy Iteration.
        checkup_matrix = np.zeros(self.q_matrix.shape)

        for i, visit in enumerate(self.episode_list):
            s = visit[0]
            action = visit[1]

            if (checkup_matrix[action, s] == 0):
                # First visit
                return_value = self.get_control_state_return(self.episode_list[i:])
                self.q_matrix[action, s] += return_value
                self.visits_counter_matrix[action, s] += 1
                checkup_matrix[action, s] = 1

    def policy_update(self):
        # Greedy update, selecting the action that yields the highest return
        for visit in self.episode_list:
            s = visit[0]
            self.policy_matrix[s] = np.argmax(self.q_matrix[:, s])

if __name__ == "__main__":
    #TODO
    a = 0