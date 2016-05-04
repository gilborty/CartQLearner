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

    def get_box(self, observation):
        """
        The following routine was written by Rich Sutton and Chuck Anderson,
        with translation from FORTRAN to C by Claude Sammut


        Given the current state, returns a number from 0 to 161
        designating the region of the state space encompassing the current state.
        Returns a value of -1 if a failure state is encountered.
        """
        ONE_DEGREE = 0.0174532
        SIX_DEGREES = 0.1047192
        TWELVE_DEGREES = 0.2094384
        FIFTY_DEGREES = 0.87266

        box = 0 #Return value

        x = observation[0]
        x_dot = observation[1]
        theta = observation[2]
        theta_dot = observation[3]

        if( x < -2.4 || x > 2.4 || theta < -TWELVE_DEGREES || theta > TWELVE_DEGREES ):
            return -1 #Signal a failure

        if( x < -0.8 ):
            box = 0
        elif( x < 0.8 ):
            box = 1
        else:
            box = 2

        if(x_dot < -0.5):
            #Do nothing
        elif(x_dot < 0.5):
            box += 3
        else:
            box += 6

        if(theta < -SIX_DEGREES):
            #Do nothing
        elif(theta < -ONE_DEGREE):
            box +=9
        elif(theta < 0):
            box += 18
        elif(theta < ONE_DEGREE):
            box += 27
        elif(theta < SIX_DEGREES):
            box += 36
        else:
            box += 45

        if( theta_dot < -FIFTY_DEGREES):
            #Do nothing
        elif( theta_dot < FIFTY_DEGREES):
            box += 54
        else:
            box += 108

        return box


    def get_action(self, observation, reward):
        return 0








agent = QAgent()
agent.print_controller_information()

#env = gym.make('CartPole-v0')

