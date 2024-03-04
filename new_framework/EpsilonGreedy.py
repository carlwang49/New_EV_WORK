import random
import numpy
import json
import itertools

class EpsilonGreedy():


    def __init__(self, epsilon):
        
        self.epsilon = epsilon 
        self.q_table = self.genetrate_q_table()


    def genetrate_q_table(self):

        facility_percent = ['0%', '20%', '40%', '60%', '80%', '100%']
        time_difference = list(range(0, 24))
        incentive = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        combinations = list(itertools.product(facility_percent, time_difference, incentive))

        q_table = {}

        for combo in combinations:
            q_table[str(combo)] = 0

        return q_table
        

    def initialize(self):
        
        self.index_pick_dict = {k: 0 for k in self.q_table.keys()} # index 被選了幾次
        
        return


    def ind_max(self, x):
        m = max(x)
        return x.index(m)


    def select_arm(self, recommend_q, score_max_recommend):

        if random.random() > self.epsilon:
            return recommend_q, 1
        else:
            return score_max_recommend, 1


    def user_decision(self, prob_accept=0.5):

        if random.random() < float(prob_accept):
            return True
        else:
            return False


    def updateEpsilon(self):
        '''
        更新 epsilon
        '''
        self.epsilon  -= (numpy.power(self.epsilon, 10))


    def update(self, chosen_arm, reward):
        '''
        更新 Q-table
        '''
        self.index_pick_dict[str(chosen_arm)] = self.index_pick_dict[str(chosen_arm)] + 1 # index 被選的次數加1
        n = self.index_pick_dict[str(chosen_arm)] # 取出此 index 被選的次數

        value = self.q_table[chosen_arm] # 此 index 目前的 q value
        new_value = ((n-1) / float(n)) * value + (1 / float(n)) * reward # update 之後的 q_value
		# weighted average of the previously estimated value and the reward we just received
        self.q_table[chosen_arm] = new_value
        return