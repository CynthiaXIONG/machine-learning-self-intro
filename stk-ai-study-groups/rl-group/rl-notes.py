import os
import numpy as np

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


def main():
    # Change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    #snippet_1()
    #snippet_2()
    #snippet_3()
    snippet_4()

if __name__ == "__main__":
    main()