import os
import numpy as np

import gym
from gym import wrappers, logger

import rl_agents

def print_frozenlake_policy(p):
    grid_size = (4,4)
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

def mc_control_frozenlake():
    env = gym.make("FrozenLake-v0")
    env.seed(0)

    #logger.set_level(logger.INFO)
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


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change dir to this script location
    os.system('') #enable ansi colors

    mc_control_frozenlake()

if __name__ == "__main__":
    main()