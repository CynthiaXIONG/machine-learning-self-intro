import os
import numpy as np

from mpatacchiola_envs.gridworld import GridWorld

### 1 ###
def mdp_kstep_transition_prob(T, k):
    return np.linalg.matrix_power(T, k)

def mdp_state_prob(v, T, k = 1):
    #v = initial state/distribution
    return np.dot(v, mdp_kstep_transition_prob(T, k))

def bellmans_action_utility(v, T, u, r, a, gamma):
    return r + gamma * np.sum(np.multiply(mdp_state_prob(v, T[:, :, a]), u))

def bellmans_state_utility(v, T, u, r, gamma):
    action_space = T.shape[2]
    action_array = np.zeros(action_space)

    for a in range(0, action_space):
        action_array[a] = bellmans_action_utility(v, T, u, r, a, gamma)

    return np.max(action_array)

def bellmans_expected_action(v, T, u):
    action_space = T.shape[2]
    action_array = np.zeros(action_space)

    for a in range(action_space):
        action_array[a] = np.sum(np.multiply(mdp_state_prob(v, T[:, :, a]), u))

    return np.argmax(action_array)

def bellmans_policy_evaluation(p, u, r, T, gamma):
    nof_states = u.size
    for s in range(nof_states):
        if not np.isnan(p[s]):
            v = np.zeros([1, nof_states])
            v[0, s] = 1.0
            action = int(p[s])
            u[s] = bellmans_action_utility(v, T, u, r[s], action, gamma)
    return u

def print_cleaning_robot_policy(p, level_shape):
    """
    Print the policy actions using symbols:
    ^, v, <, > up, down, left, right
    * terminal states
    # obstacles
    """
    counter = 0
    policy_string = ""
    for row in range(level_shape[0]):
        for col in range(level_shape[1]):
            if(p[counter] == -1): policy_string += " *  "            
            elif(p[counter] == 0): policy_string += " ^  "
            elif(p[counter] == 1): policy_string += " <  "
            elif(p[counter] == 2): policy_string += " v  "           
            elif(p[counter] == 3): policy_string += " >  "
            elif(np.isnan(p[counter])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)

def print_grid_world_policy(p):
    """
    Print the policy actions using symbols:
    # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, NaN=Obstacle, -1=NoAction
    * terminal states
    # obstacles
    """
    policy_string = ""
    for row in range(p.shape[0]):
        for col in range(p.shape[1]):
            if(p[row, col] == -1): policy_string += " *  "            
            elif(p[row, col] == 0): policy_string += " ^  "
            elif(p[row, col] == 1): policy_string += " >  "
            elif(p[row, col] == 2): policy_string += " v  "           
            elif(p[row, col] == 3): policy_string += " <  "
            elif(p[row, col] == -2): policy_string += " #  "
            elif(np.isnan(p[row, col])): policy_string += " #  "
        policy_string += '\n'
    print(policy_string)

def setup_34_gridworld():
    # The world has 3 rows and 4 columns
    env = GridWorld(3, 4)
    # Define the state matrix
    # Adding obstacle at position (1,1)
    # Adding the two terminal states
    state_matrix = np.zeros((3,4))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    # Define the reward matrix
    # The reward is -0.04 for all states but the terminal
    reward_matrix = np.full((3,4), -0.04)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = -1
    # Define the transition matrix
    # For each one of the four actions there is a probability
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])

    # Set the matrices 
    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    return env

def get_mc_state_return(state_list, gamma):
    #state_list is type (position, reward)
    ret_value = 0
    for i in range(len(state_list)):
        reward = state_list[i][1]
        ret_value += reward * np.power(gamma, i)
    return ret_value

def get_mc_control_state_return(state_list, gamma):
    #state_list is type (position, action, reward)
    ret_value = 0
    for i in range(len(state_list)):
        reward = state_list[i][2]
        ret_value += reward * np.power(gamma, i)
    return ret_value

def mc_control_update_policy(episode_list, policy_matrix, q_matrix):
    #q_matrix = state_action_matrix
    for visit in episode_list:
        observation = visit[0]
        col = observation[1] + (observation[0]*4) #convert to vector
        if(policy_matrix[observation[0], observation[1]] != -1):
            policy_matrix[observation[0], observation[1]] = np.argmax(q_matrix[:, col])
    return policy_matrix

def snippet_1():
    #Declaring the initial distribution
    v = np.array([[0.5, 0.5]])

    #Declaring the Transition Matrix T
    T = np.array([[0.9, 0.1],
                  [0.5, 0.5]])

    #Printing T after k-iterations
    print("T: " + str(T))
    print("T_3: " + str(mdp_kstep_transition_prob(T, 3)))
    print("T_50: " + str(mdp_kstep_transition_prob(T, 5)))
    print("T_100: " + str(mdp_kstep_transition_prob(T, 100)))

    #Printing the initial distribution
    print("v: " + str(v))
    print("v_1: " + str(mdp_state_prob(v, T, 1)))
    print("v_3: " + str(mdp_state_prob(v, T, 3)))
    print("v_50: " + str(mdp_state_prob(v, T, 50)))
    print("v_100: " + str(mdp_state_prob(v, T, 100)))

def snippet_2():
    #cleaning robot example
    #Starting state vector
    #The agent starts from (1, 1)
    v = np.array([[0.0, 0.0, 0.0, 0.0, 
                   0.0, 0.0, 0.0, 0.0, 
                   1.0, 0.0, 0.0, 0.0]])

    #Transition matrix loaded from the cleaning_robot_T.npy file (too big)
    T = np.load("cleaning_robot_T.npy")

    #Utility vector (given, magically calculated =D)
    u = np.array([[0.812, 0.868, 0.918,   1.0,
                   0.762,   0.0, 0.660,  -1.0,
                   0.705, 0.655, 0.611, 0.388]])

    #Defining the reward for state (1,1). Rewards for all normal states is -0.4 (battery depleating a bit)
    reward = -0.04
    #Assuming that the discount factor is equal to 1.0
    gamma = 1.0

    #Use the Bellman equation to find the utility of state (1,1)
    utility_11 = bellmans_state_utility(v, T, u, reward, gamma)
    print("Utility of state (1,1): " + str(utility_11))

def snippet_3():
    #Bellmans Eq for MDP Value iteration
    #Change as you want
    nof_states = 12
    gamma = 0.999 #Discount factor
    iteration = 0 #Iteration counter
    epsilon = 0.001 #Stopping criteria small value

    graph_list = list() #List containing the data for each iteation

    #Transition matrix loaded from the cleaning_robot_T.npy file (too big)
    T = np.load("cleaning_robot_T.npy")

    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    #Utility vector, initialize to zeros
    u = np.zeros(r.shape)

    while True:
        max_delta = 0
        iteration += 1
        graph_list.append(u)

        for s in range(nof_states):
            reward = r[s]
            v = np.zeros([1, nof_states])
            v[0, s] = 1.0  #assing current state to the state distribution

            prev_u = u[s]
            u[s] = bellmans_state_utility(v, T, u, reward, gamma)

            max_delta = max(max_delta, np.abs(u[s] - prev_u))

        if (max_delta < (epsilon * (1 - gamma) / gamma)):
            print("=================== FINAL RESULT ==================")
            print("Iterations: " + str(iteration))
            print("Delta: " + str(max_delta))
            print("Gamma: " + str(gamma))
            print("Epsilon: " + str(epsilon))
            print("===================================================")
            print(u[0:4])
            print(u[4:8])
            print(u[8:12])
            print("===================================================")
            break

def snippet_4():
    #policy iteration algorithm
    nof_states = 12
    gamma = 0.999
    epsilon = 0.0001
    iteration = 0
    T = np.load("cleaning_robot_T.npy")

    #Generate the first policy randomly
    # NaN=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
    np.random.seed(1)
    p = np.random.randint(0, 4, size=(nof_states)).astype(np.float32)
    p[5] = np.NaN
    p[3] = p[7] = -1

    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    #Utility vector, initialize to zeros
    u = np.zeros(r.shape)

    while True:
        iteration += 1
        #1- Policy evaluation
        u_0 = u.copy()
        u = bellmans_policy_evaluation(p, u, r, T, gamma)
        #Stopping criteria
        delta = np.absolute(u - u_0).max()
        if delta < epsilon * (1 - gamma) / gamma: 
            break

        for s in range(nof_states):
            if not np.isnan(p[s]) and not p[s]==-1:
               v = np.zeros([1, nof_states])
               v[0, s] = 1.0
               #2- Policy improvement
               p[s] = bellmans_expected_action(v, T, u)         

        print_cleaning_robot_policy(p, level_shape=(3,4))
    
    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")
    print_cleaning_robot_policy(p, level_shape=(3,4))
    print("===================================================")

def snippet_5():
    env = setup_34_gridworld()

    #Reset the environment
    observation = env.reset()
    #Display the world printing on terminal
    env.render()
    # Define the policy matrix
    # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, NaN=Obstacle, -1=NoAction
    # This is the optimal policy for world with reward=-0.04
    policy_matrix = np.array([[1,      1,  1,  -1],
                              [0, np.NaN,  0,  -1],
                              [0,      3,  3,   3]])

    for _ in range(1000):
        action = policy_matrix[observation[0], observation[1]]
        observation, reward, done = env.step(action)
        print("")
        print("ACTION: " + str(action))
        print("REWARD: " + str(reward))
        print("DONE: " + str(done))
        env.render()
        if done: break

def snippet_6():
    env = setup_34_gridworld()
    #Reset the environment
    observation = env.reset()

    # Define the policy matrix
    # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, NaN=Obstacle, -1=NoAction
    # This is the optimal policy for world with reward=-0.04
    policy_matrix = np.array([[1,      1,  1,  -1],
                              [0, np.NaN,  0,  -1],
                              [0,      3,  3,   3]])

    # Defining an empty utility matrix
    utility_matrix = np.zeros((3,4))
    # init with 1.0e-10 to avoid division by zero
    running_mean_matrix = np.full((3,4), 1.0e-10)
    
    gamma = 0.999 #discount factor
    nof_epochs = 50000
    print_epoch = 1000
    nof_steps = 100

    for epoch in range(nof_epochs):
        #Starting a new episode
        episode_list = list()
        #Reset and return the first observation
        observation= env.reset(exploring_starts=False)

        for _ in range(nof_steps):
            # Take the action from the policy matrix
            action = policy_matrix[observation[0], observation[1]]
            # Move one step in the environment and get obs and reward
            observation, reward, done = env.step(action)
            # Append the visit in the episode list
            episode_list.append((observation, reward))
            if done: break
        
        # The episode is finished, now estimating the utilities
        # Checkup to identify if it is the first visit to a state -> First Visit MC
        checkup_matrix = np.zeros((3,4))
        # This cycle is the implementation of First-Visit MC.
        # For each state stored in the episode list it checks if it
        # is the first visit and then estimates the return.
        visit_i = 0
        for visit in episode_list:
            observation = visit[0] #observation is tuple(row, col)
            reward = visit[1]

            if (checkup_matrix[observation[0], observation[1]] == 0):
                ret_value = get_mc_state_return(episode_list[visit_i:], gamma)
                running_mean_matrix[observation[0], observation[1]] += 1
                utility_matrix[observation[0], observation[1]] += ret_value
                checkup_matrix[observation[0], observation[1]] = 1
            
            visit_i += 1
        
        if (epoch % print_epoch == 0):
            print("Utility matrix after " + str(epoch+1) + " iterations:") 
            print(utility_matrix / running_mean_matrix)
    
    #Time to check the utility matrix obtained
    print("Utility matrix after " + str(nof_epochs) + " iterations:")
    print(utility_matrix / running_mean_matrix)

def snippet_7():
    env = setup_34_gridworld()
    #Reset the environment
    observation = env.reset()

    # Random policy matrix
    policy_matrix = np.random.randint(low=0, high=4,size=(3, 4)).astype(np.int32)
    policy_matrix[1,1] = -2 #-2 for the obstacle at (1,1)
    policy_matrix[0,3] = policy_matrix[1,3] = -1 #No action (terminal states)

    # Q/State-action matrix (init to zeros or to random values)
    q_matrix = np.random.random_sample((4,12)) # Q

    # init with 1.0e-10 to avoid division by zero
    running_mean_matrix = np.full((4,12), 1.0e-10)

    gamma = 0.999 #discount factor
    nof_epochs = 50000
    print_epoch = 1000
    nof_steps = 100

    for epoch in range(nof_epochs):
        #Starting a new episode
        episode_list = list()
        #Reset and return the first observation
        observation= env.reset(exploring_starts=True)

        for i in range(nof_steps):
            # Take the action from the action matrix
            action = policy_matrix[observation[0], observation[1]]

            if (i == 0): #inital step, take random action (exploring starts)
                action = np.random.randint(0, 4)

            # Move one step in the environment and gets 
            # a new observation and the reward
            new_observation, reward, done = env.step(action)

            #Append the visit in the episode list
            episode_list.append((observation, action, reward))
            observation = new_observation
            if done: break
        
         # The episode is finished, now estimating the utilities
        checkup_matrix = np.zeros((4,12))
        visit_i = 0
        # This cycle is the implementation of First-Visit MC.
        # For each state-action stored in the episode list it checks if 
        # it is the first visit and then estimates the return. 
        # This is the Evaluation step of the GPI.
        for visit in episode_list:
            observation = visit[0]
            action = visit[1]
            col = observation[1] + (observation[0] * 4)
            row = action
            if (checkup_matrix[row, col] == 0):
                return_value = get_mc_control_state_return(episode_list[visit_i:], gamma)
                running_mean_matrix[row, col] += 1
                q_matrix[row, col] += return_value
                checkup_matrix[row, col] = 1
            visit_i += 1

        # Policy Update (Improvement)
        policy_matrix = mc_control_update_policy(episode_list,  policy_matrix, q_matrix/running_mean_matrix)

        # Printing
        if(epoch % print_epoch == 0):
            print("")
            print("Q matrix after " + str(epoch+1) + " iterations:") 
            print(q_matrix / running_mean_matrix)
            print("Policy matrix after " + str(epoch+1) + " iterations:") 
            print(policy_matrix)
            print_grid_world_policy(policy_matrix)
    
    # Time to check the utility matrix obtained
    print("Q " + str(nof_epochs) + " iterations:")
    print(q_matrix / running_mean_matrix)

def main():
    # Change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    #snippet_1()
    #snippet_2()
    #snippet_3()
    #snippet_4()
    #snippet_5()
    #snippet_6()
    snippet_7()

if __name__ == "__main__":
    main()