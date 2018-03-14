import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#----------------------------------------------------------
class CoinFlipMachine():
    def __init__(self, initial_balance_exp):
        self.initial_balance = 1 << initial_balance_exp
        self.Reset()
        pass

    def Flip(self, action):
        bet = 1 << action
        if bet > self.balance:
            return self.HashState(self.balance, self.winning_streak), -999999 #return low reward to make it bad!!


        result = np.random.randn(1)

        win = result > 0.5

        if win:
            reward = bet
           
            if self.winning_streak > 0:
                self.winning_streak = self.winning_streak + 1
            else:
                self.winning_streak = 1
        else:
            reward = -bet

            if self.winning_streak < 0:
                self.winning_streak = self.winning_streak - 1
            else:
                self.winning_streak = -1

        self.balance = self.balance + reward
        return self.HashState(self.balance, self.winning_streak), reward
            
    def Reset(self):
        self.balance = self.initial_balance
        self.winning_streak = 0

        return self.HashState(self.balance, self.winning_streak)

    def HashState(self, balance, winning_streak):
        #hashes as SSBBBBBB  (S-streak, B- balance)
        hash = balance
        hash = hash + winning_streak * 1000000
        return hash

#----------------------------------------------------------
class QTable():
    def __init__(self, s_size, a_size, lr, gamma):
        self.s_size = s_size
        self.a_size = a_size

        self.Q = pd.DataFrame({0:np.zeros([a_size])})

        self.lr = lr
        self.gamma = gamma
        self.epsilon = 0.99

    def SelectAction(self, s, iteration):
        #Choose an action by greedily (with noise, so we dont get stuck on local maximums) picking from Q table
        state_exists = s in self.Q
        if state_exists: 
            a = np.argmax(self.Q[s])
        
        if (not state_exists) or (np.random.rand(1) < self.epsilon): #random change is reduced per iteration later on
            a = np.random.randint(self.a_size)
        
        #Slowly reduce the random action change as the Q-Table improves
        self.epsilon = 1. / ((iteration/50) + 10)
        
        return a

    def Update(self, s, s1, r, a):
        #Update/Improve Q-Table with new knowledge
        max_future_reward = 0
        if s1 in self.Q: 
            max_future_reward = self.Q[s1].max() # get the max reward using the QTable data for the next state s1
        
        improved_estimation = r + self.gamma * max_future_reward #improve estimation reward, adding the future reward parameter as 
        
        self.InitializeValue(s, a)
        self.InitializeValue(s1, a)
        self.Q[s][a] = (1-self.lr) * self.Q[s][a] + self.lr *  improved_estimation #Update q_table with the new reward
        
        return 

    def InitializeValue(self, s, a):
        if not s in self.Q:
            self.Q[s] = np.zeros([self.a_size])
            

#----------------------------------------------------------
class DoubleTheLossMethod():
    def __init__(self):
        pass

    def SelectAction(self, s, iteration):
        winning_streak = s // 1000000

        if (winning_streak < 0):
            winning_streak = winning_streak + 1
            return winning_streak * -1
        else:
            return 0 

    

#----------------------------------------------------------
def DoMain():
    num_of_2_exp = 10
    coin_flip_machine = CoinFlipMachine(num_of_2_exp)


    ##QLearning - Table Implementation
    # Setup the table
    lr = .8 #learning rate, 
    gamma = .95  #discount rate, how much we value future reward
    num_episodes = 10000
    num_sim_steps = 50

    q_table = QTable(s_size=2, a_size=num_of_2_exp, lr=lr, gamma=gamma)
    #double_method = DoubleTheLossMethod()

    #create lists to contain total rewards and steps per simulation
    j_list = []
    r_list = []

    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = coin_flip_machine.Reset()
        r_total = 0
        j = 0

        #The Q-Table learning algorithm
        #for j in range(num_sim_steps):
        while (j < num_sim_steps or r_total < 0):
            j = j + 1
            #select action
            a = q_table.SelectAction(s, j)
            #a = double_method.SelectAction(s, j)
            
            #Get new state and reward from environment
            s1, r = coin_flip_machine.Flip(a)

            #Update/Improve Q-Table with new knowledge
            q_table.Update(s, s1, r, a)

            #Update state
            r_total += r
            s = s1

            if (r_total < -1024):
                break
        
        #append simulation end data
        j_list.append(j)
        if (r_total < 0):
            r_total = -11
        r_list.append(r_total)
    
    print("Q-Table success rate: " +  str(sum(r_list) / num_episodes) + "%")

    plt.figure(1)

    plt.subplot(211)
    plt.plot(r_list, linewidth=0.3)

    plt.subplot(212)
    plt.plot(j_list, linewidth=0.3)

    plt.show()

if __name__ == "__main__":
    DoMain()
