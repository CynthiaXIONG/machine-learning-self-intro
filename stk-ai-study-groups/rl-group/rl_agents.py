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
        # Get Hashed initial state
        if (self.hash_s_function is not None):
            s0 = self.hash_s_function(s9)

        for i in range(nof_steps):
            if (render_function is not None):
                render_function()

            # Get action
            a0 = self.get_action(s0, get_random=self.exploration_function(i))
            # Move one step in the environment and get new state and reward
            s1, reward, done = env.step(a0)[:3] #just get the first three, for compatibility with both blog GridWorld and openai gym
            if (self.hash_s_function is not None): #Hash new state
                s1 = self.hash_s_function(s1)

            # Append the visit in the episode list
            self.append_episode(s0, a0, reward)

            s0 = s1
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

class SARSAControlAgent():
    # SARSA TD(lambda) control, based on https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html
    def __init__(self, observation_space, alpha=0.1, gamma=0.99, _lambda=0, hash_s_function=None, exploration_function=None):
        self.observation_space = observation_space
        self.action_space = observation_space[0]
        self.state_space = observation_space[1]
        self.alpha = alpha
        self.gamma = gamma
        self._lambda = _lambda

        self.hash_s_function = hash_s_function #used to hash s (e.g: if multidimensional or continuous)

        if (exploration_function is not None):
            self.exploration_function = exploration_function
        else:
            self.exploration_function = lambda : np.random.rand(1) < 0.1

        # Initialize Q and PI with random values
        self.q_matrix = np.zeros([self.action_space, self.state_space]) #np.random.random_sample([self.action_space, self.state_space])
        self.policy_matrix = np.random.randint(low=0, high=self.action_space, size=self.state_space).astype(np.int32)

    def get_action(self, s, get_random=False):
        if (get_random):
            action = np.random.randint(0, self.action_space)
        else:
            # Get from policy
            action = self.policy_matrix[s]
        return action

    def _get_q_prediction_error(self, q_matrix, s0, a0, r, s1, a1, gamma):
        #prediction error = delta
        q_t0 = self.q_matrix[a0, s0] #old estimation
        
        q_t1 = self.q_matrix[a1, s1] #new estimation
        new_estimation = r + gamma * q_t1

        return new_estimation - q_t0

    def update_q_matrix_td0(self, s0, a0, r, s1, a1): #SARSA
        delta = self._get_q_prediction_error(self.q_matrix, s0, a0, r, s1, a1, self.gamma)
        self.q_matrix[a0, s0] += self.alpha * delta

    def update_q_matrix_tdlambda(self, s0, a0, r, s1, a1, trace_matrix): # with eligibilility trace matrix
        delta = self._get_q_prediction_error(self.q_matrix, s0, a0, r, s1, a1, self.gamma)
        self.q_matrix += self.alpha * delta * trace_matrix

    def _update_eligibility_trace(self, trace_matrix):
        trace_matrix *= self.gamma * self._lambda
        return trace_matrix

    def update_policy(self, s):
        #greedy action
        best_action = np.argmax(self.q_matrix[:, s])
        self.policy_matrix[s] = best_action

    def initialize_epoch(self):
        pass

    def execute_epoch(self, nof_steps, env, s0, render_function=None):
        # Get Hashed initial state
        if (self.hash_s_function is not None):
            s0 = self.hash_s_function(s0)

        if (self._lambda > 0):
            trace_matrix = np.zeros(self.q_matrix.shape)

        for i in range(nof_steps):
            if (render_function is not None):
                render_function()

            # Get action
            a0 = self.get_action(s0, get_random=self.exploration_function(i))
            # Move one step in the environment and get new state and reward
            s1, r, done = env.step(a0)[:3] # Just get the first three, for compatibility with both blog GridWorld and openai gym
            if (self.hash_s_function is not None): # Hash new state
                s1 = self.hash_s_function(s1)

            a1 = self.policy_matrix[s1]

            if (self._lambda == 0):
                # Update q_matrix
                self.update_q_matrix_td0(s0, a0, r, s1, a1)
            
            else:
                #Adding +1 in the trace matrix for the state visited
                trace_matrix[a0, s0] += 1
                # Update q_matrix
                self.update_q_matrix_tdlambda(s0, a0, r, s1, a1, trace_matrix)
                # Update the trace matrix decay for the next step
                trace_matrix = self._update_eligibility_trace(trace_matrix)

            # Update policy
            self.update_policy(s0)

            s0 = s1
            if done: break

class QLearningAgent():
    # QLearning TD(0) control, based on https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html
    # TODO On Policy
    def __init__(self, observation_space, alpha=0.1, gamma=0.99, on_policy=False, exploratory_policy=None, hash_s_function=None, exploration_function=None):
        self.observation_space = observation_space
        self.action_space = observation_space[0]
        self.state_space = observation_space[1]
        self.alpha = alpha
        self.gamma = gamma

        self.hash_s_function = hash_s_function #used to hash s (e.g: if multidimensional or continuous)

        if (exploration_function is not None):
            self.exploration_function = exploration_function
        else:
            self.exploration_function = lambda : np.random.rand(1) < 0.1

        self.exploratory_policy = exploratory_policy

        # Initialize Q and PI with random values
        self.q_matrix = np.zeros([self.action_space, self.state_space]) #np.random.random_sample([self.action_space, self.state_space])
        self.optimal_policy_matrix = np.random.randint(low=0, high=self.action_space, size=self.state_space).astype(np.int32)

    def get_action(self, s, get_random=False):
        if (get_random):
            action = np.random.randint(0, self.action_space)
        else:
            # Get from policy
            action = self.exploratory_policy[s]
        return action

    def _get_q_prediction_error(self, q_matrix, s0, a0, r, s1, gamma):
        #prediction error = delta
        q_t0 = self.q_matrix[a0, s0] #old estimation

        q_t1 = np.max(self.q_matrix[:, s1]) #best possible estimation
        new_estimation = r + gamma * q_t1

        return new_estimation - q_t0

    def update_q_matrix(self, s0, a0, r, s1): #SARS
        delta = self._get_q_prediction_error(self.q_matrix, s0, a0, r, s1, self.gamma)
        self.q_matrix[a0, s0] += self.alpha * delta

    def update_optimal_policy(self, s):
        #greedy action
        best_action = np.argmax(self.q_matrix[:, s])
        self.optimal_policy_matrix[s] = best_action

    def initialize_epoch(self):
        pass

    def execute_epoch(self, nof_steps, env, s0, render_function=None):
        # Get Hashed initial state
        if (self.hash_s_function is not None):
            s0 = self.hash_s_function(s0)

        for i in range(nof_steps):
            if (render_function is not None):
                render_function()

            # Get action
            a0 = self.get_action(s0, get_random=self.exploration_function(i))
            # Move one step in the environment and get new state and reward
            s1, r, done = env.step(a0)[:3] # Just get the first three, for compatibility with both blog GridWorld and openai gym
            if (self.hash_s_function is not None): # Hash new state
                s1 = self.hash_s_function(s1)

            # Update q_matrix
            self.update_q_matrix(s0, a0, r, s1)
            # Update policy
            self.update_optimal_policy(s0)

            s0 = s1
            if done: break

    

if __name__ == "__main__":
    #TODO
    a = QLearningAgent([3,4])