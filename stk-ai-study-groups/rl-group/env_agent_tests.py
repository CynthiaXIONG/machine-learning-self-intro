import os
import numpy as np

import gym
from gym import wrappers, logger

from mpatacchiola_envs.gridworld import GridWorld

import rl_agents

# Mpatacchiola GridWorld Utils
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

def hash_gridworld_state(s, grid_size=(3,4)):
    return (s[0] * grid_size[1]) + s[1]

def print_grid_world_policy(p, grid_size=(3,4)):
    policy_string = ""
    i = 0
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            if(p[i] == -1): policy_string += " *  " #end state, both the good (recharging station) and the bad (stairs) one        
            elif(p[i] == 0): policy_string += " ^  "
            elif(p[i] == 1): policy_string += " >  "
            elif(p[i] == 2): policy_string += " v  "           
            elif(p[i] == 3): policy_string += " <  "
            elif(p[i] == -2): policy_string += " #  " #wall, cant move here
            i += 1

        policy_string += '\n'
    print(policy_string)

# OpenAi Frozenlake Utils
def print_frozenlake_policy(p, grid_size=(4,4)):
    policy_string = ""
    i = 0
    for _ in range(grid_size[0]):
        for _ in range(grid_size[1]):
            if(p[i] == 0): policy_string += "<"            
            elif(p[i] == 1): policy_string += "v"
            elif(p[i] == 2): policy_string += ">"
            elif(p[i] == 3): policy_string += "^"
            i += 1         

        policy_string += '\n'
    print(policy_string)

# Trials
def mc_control_gridworld():
    np.random.seed(1)
    env = setup_34_gridworld()
    nof_epochs = 50000
    nof_steps = 100
    exploration_function = lambda i : np.random.rand(1) < (1./((i/100) + 1))
    mc_agent = rl_agents.MonteCarloControlAgent(observation_space=[4, 3*4], hash_s_function=hash_gridworld_state, exploration_function=exploration_function)

    for epoch in range(nof_epochs):
        #Reset the environment
        s0 = env.reset(exploring_starts=False)
        mc_agent.initialize_epoch()

        mc_agent.execute_epoch(nof_steps, env, s0)

        # Printing
        if(epoch % 5000 == 0):
            print("")
            print("Q matrix after " + str(epoch+1) + " iterations:") 
            print(mc_agent.get_mean_return_matrix())
            print("Policy matrix after " + str(epoch+1) + " iterations:") 
            print(mc_agent.policy_matrix)
            print_grid_world_policy(mc_agent.policy_matrix)
    
    # Time to check the utility matrix obtained
    print("Q " + str(nof_epochs) + " iterations:")
    print(mc_agent.get_mean_return_matrix())

def mc_control_frozenlake():
    env = gym.make("FrozenLake-v0")
    env.seed(0)

    exploration_function = lambda i : np.random.rand(1) < (1./((i/100) + 2))
    mc_agent = rl_agents.MonteCarloControlAgent(observation_space=[env.action_space.n, env.observation_space.n], exploration_function=exploration_function)
    nof_epochs = 100000
    print_epoch = 10000

    r_all_epochs = 0

    mc_agent.policy_matrix = [1,0,1,0, 1,0,1,0, 2,2,1,0, 0,2,2,0]

    for epoch in range(nof_epochs):
        #Reset the environment
        s0 = env.reset()
        mc_agent.initialize_epoch()

        mc_agent.execute_epoch(100, env, s0, render_function=env.render if (epoch % print_epoch == 0) else None)
        r_all_epochs += mc_agent.get_latest_episode()[2]

        # Printing
        if(epoch % print_epoch == 0):
            print("")
            #print("Q matrix after " + str(epoch+1) + " iterations:") 
            #print(mc_agent.get_mean_return_matrix())
            print("Policy matrix after " + str(epoch+1) + " iterations:") 
            print_frozenlake_policy(mc_agent.policy_matrix)

    print ("Score over time: " +  str(r_all_epochs/nof_epochs))

def sarsa_control_gridworld():
    np.random.seed(1)
    env = setup_34_gridworld()
    gamma = 0.999
    alpha = 0.001
    nof_epochs = 100000
    nof_steps = 1000
    print_step = 100
    exploration_function = lambda i : np.random.rand(1) < (1./((i/100) + 1))
    sarsa_agent = rl_agents.SARSAControlAgent(observation_space=[4, 3*4], alpha=alpha, gamma=gamma, _lambda=0.5,\
                                           hash_s_function=hash_gridworld_state, exploration_function=exploration_function)

    for epoch in range(nof_epochs):
        #Reset the environment
        s0 = env.reset(exploring_starts=False)
        sarsa_agent.initialize_epoch()

        sarsa_agent.execute_epoch(nof_steps, env, s0)

        # Printing
        if(epoch % print_step == 0):
            print("")
            print("Q matrix after " + str(epoch+1) + " iterations:") 
            print(sarsa_agent.policy_matrix)
            print("Policy matrix after " + str(epoch+1) + " iterations:") 
            print_grid_world_policy(sarsa_agent.policy_matrix)
    
    # Time to check the utility matrix obtained
    print("Q " + str(nof_epochs) + " iterations:")
    print(sarsa_agent.policy_matrix)
    print("Policy matrix after " + str(epoch+1) + " iterations:") 
    print_grid_world_policy(sarsa_agent.policy_matrix)

def sarsa_control_frozenlake():
    env = gym.make("FrozenLake-v0")
    env.seed(0)

    gamma = 0.99
    alpha = 0.85
    nof_epochs = 50000
    nof_steps = 1000
    print_step = 1000
    exploration_function = lambda i : np.random.rand(1) < (1./((i/100) + 1))
    sarsa_agent = rl_agents.SARSAControlAgent(observation_space=[env.action_space.n, env.observation_space.n], alpha=alpha, gamma=gamma, _lambda=0)#,\
                                              #exploration_function=exploration_function)

    r_all_epochs = 0

    for epoch in range(nof_epochs):
        #Reset the environment
        s0 = env.reset()
        sarsa_agent.initialize_epoch()

        sarsa_agent.execute_epoch(nof_steps, env, s0, render_function=env.render if (epoch % print_step == 0) else None)
        r_all_epochs += sarsa_agent.get_latest_episode()[2]

        # Printing
        if(epoch % print_step == 0):
            print("")
            print("Q matrix after " + str(epoch+1) + " iterations:") 
            print(sarsa_agent.policy_matrix)
            print("Policy matrix after " + str(epoch+1) + " iterations:") 
            print_frozenlake_policy(sarsa_agent.policy_matrix)
    
    # Time to check the utility matrix obtained
    print("Q " + str(nof_epochs) + " iterations:")
    print(sarsa_agent.policy_matrix)
    print("Policy matrix after " + str(epoch+1) + " iterations:") 
    print_frozenlake_policy(sarsa_agent.policy_matrix)

    print ("Score over time: " +  str(r_all_epochs/nof_epochs))

def qlearning_gridworld():
    np.random.seed(1)
    env = setup_34_gridworld()
    gamma = 0.999
    alpha = 0.001
    nof_epochs = 100000
    nof_steps = 1000
    print_step = 100
    exploration_function = lambda i : np.random.rand(1) < (1./((i/100) + 1))

    exploratory_policy_matrix = np.array([1, 1, 1, 0, 
                                          0, 0, 0, 0,
                                          0, 1, 0, 3])

    qlearn_agent = rl_agents.QLearningAgent(observation_space=[4, 3*4], alpha=alpha, gamma=gamma, \
                                            exploratory_policy=exploratory_policy_matrix, hash_s_function=hash_gridworld_state,\
                                            exploration_function=exploration_function)

    for epoch in range(nof_epochs):
        #Reset the environment
        s0 = env.reset(exploring_starts=False)
        qlearn_agent.initialize_epoch()

        qlearn_agent.execute_epoch(nof_steps, env, s0)

        # Printing
        if(epoch % print_step == 0):
            print("")
            print("Q matrix after " + str(epoch+1) + " iterations:") 
            print(qlearn_agent.q_matrix)
            print("Policy matrix after " + str(epoch+1) + " iterations:") 
            print_grid_world_policy(qlearn_agent.optimal_policy_matrix)
    
    # Time to check the utility matrix obtained
    print("Q " + str(nof_epochs) + " iterations:")
    print(qlearn_agent.q_matrix)
    print("Policy matrix after " + str(epoch+1) + " iterations:") 
    print_grid_world_policy(qlearn_agent.optimal_policy_matrix)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change dir to this script location
    os.system('') #enable ansi colors

    #mc_control_gridworld()
    #mc_control_frozenlake()

    #sarsa_control_gridworld()
    sarsa_control_frozenlake()

    #qlearning_gridworld()


if __name__ == "__main__":
    main()