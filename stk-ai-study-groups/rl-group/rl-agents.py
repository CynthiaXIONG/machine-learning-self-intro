#code style: https://google.github.io/styleguide/pyguide.html

import numpy as np

class MonteCarloControlAgent():
    #First-Visit MC, based on https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html
    def __init__(self, observation_space, gamma=0.99):
        self.observation_space = observation_space
        self.action_space = observation_space[0]
        self.state_space = observation_space[1]
        self.gamma = gamma

        #initialize Q and Pi with random values
        self.q_matrix = np.random.random_sample((self.action_space, self.state_space))
        self.policy_matrix = np.random.randint(low=0, high=self.action_space, size=self.state_space).astype(np.int32)

        # init with 1.0e-10 to avoid division by zero
        self.running_mean_matrix = np.full(self.q_matrix.shape, 1.0e-10)
        pass
        
    def get_action(self, s, get_random=False):
        if (get_random):
            action = np.random.randint(0, self.action_space)
        else:
            #get from policy
            action = self.policy_matrix[s]
        return action

    def initialize_epoch(self):
        self.episode_list = list()

    def append_episode(self, observation, action, reward):
        self.episode_list.append((observation, action, reward))

    def 

        


if __name__ == "__main__":
    mc_agent = MonteCarloControlAgent(observation_space=[4, 3*4])
    print(mc_agent.q_matrix)
    print(mc_agent.policy_matrix)
    print(mc_agent.running_mean_matrix)
    print(mc_agent.get_action(0))
    
    
