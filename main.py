__author__ = 'gilbert'


"""
Based off of q controller from here:
http://pages.cs.wisc.edu/~finton/qcontroller.html
"""

import gym
import random
import numpy as np

class QAgent:

    def __init__(self):

        #Constants
        self.W_INIT = 0.0
        self.NUM_BOXES = 162

        self.ALPHA = 0.5 #Learning Rate Parameter
        self.BETA = 0.0 #Magnitude of noise added to choice
        self.GAMMA = 0.999 #Discount factor for future reinforcement

        self.random_seed = 1
        random.seed(self.random_seed)


        self.q_val = np.array([[self.NUM_BOXES],[self.NUM_BOXES]], np.float) #state-action values

        self.first_time = False

        self.cur_action = -1
        self.prev_action = -1
        self.cur_state = 0
        self.prev_state = 0

    def print_controller_information(self):
        print("Firing up controller with ALPHA to %f" % self.ALPHA)
        print("BETA: %f" % self.BETA)
        print("GAMMA: %f" % self.GAMMA)
        print("Random Seed: %d" % self.random_seed)

    def reset_controller(self):
        self.cur_state, self.prev_state = 0;
        self.cur_action, self.prev_action = -1

    def get_box(self):
        return 0


    def get_action(self, observation, reward):
        return 0








agent = QAgent()
agent.print_controller_information()

#env = gym.make('CartPole-v0')

