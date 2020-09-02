import time
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from enum import Enum
from random import shuffle as rshuffle
import tensorflow.compat.v2 as tf
from tensorflow import keras
import tensorflow as tf1

import deal_cards as dc
import cards_env as ce
import discard_q_net_keras_3 as q_net

#
def best_6_cards(y_true, y_pred):
    oindex = tf1.argsort(y_pred, direction='DESCENDING', axis=-1)
    top_1 = tf1.math.top_k(oindex, 1)
    return top_1 #it is NOT a list. give up


class RandomAgent:
    def __init__(self):
        self.decide_his={}
        return
        
    def decide(self, state2):
        oindex = np.where(state2 > 0)[0]
        discard_candiates_len = len(oindex)

        action_index = np.random.choice(discard_candiates_len)
        action = oindex[action_index]
        
        if action not in self.decide_his.keys():
            self.decide_his[action] = 1
        else:
            self.decide_his[action] += 1

        behavior = 1. / discard_candiates_len     # why ?? A: 概率为1/n=b(A|S). 这是异策的关键
        return action, behavior


class DiscardAgent_net1_PI(q_net.DiscardAgent_net6_base): #1 step
    #input = onehot*54; output=54*p()
    def __init__(self, gamma=0.0):
        self.gamma = gamma

        input_size=54
        hidden_layers = [[256, 0.2], [64, 0.2]]
        output_size=54
        activation=tf.nn.relu
        loss=tf.losses.categorical_crossentropy  # tf.losses.mse
        output_activation=tf.nn.softmax
        learning_rate = 0.001
        self.policy_net = self.build_network(input_size, hidden_layers, output_size,
                                             activation, loss, output_activation, learning_rate)

        input_size=54
        hidden_layers = [[128, 0.2]]
        output_size=1
        activation=tf.nn.relu
        loss=tf.losses.mse
        output_activation=None
        learning_rate = 0.001
        self.v_net = self.build_network(input_size, hidden_layers, output_size,
                                        activation, loss, output_activation, learning_rate)
        return
    
    def decide(self, state2):
        oindex = np.where(state2 > 0)[0]
        state = state2[np.newaxis]  #full_cards_onehot_like, batch=1
        actions0 = self.policy_net.predict(state).reshape(-1)  #oindex(54,), action0(1,54)->(54)
        actions1 = actions0[oindex]
        actions_index0 = np.argsort(-actions1)
        actions_index = actions_index0[:6]
        actions = oindex[actions_index]
        return actions

    def pre_learn(self, state2, action, reward, done, best_discards_oindex):
        #loss = mse MUST
        discard_possibility = np.zeros(54)

        discard_possibility[best_discards_oindex] = 1/6
        y = discard_possibility[np.newaxis]
        
        state = state2[np.newaxis]  #full_cards_onehot_like, batch=1
        self.policy_net.fit(state, y, verbose=0)
        return

    def learn(self, state2, action, reward, done, best_discards_oindex):
        discard_possibility = np.zeros(54)

        discard_possibility[best_discards_oindex] = 1/6
        G = reward # 回报
        gamma = self.gamma  # 1 round
        y = gamma * G * discard_possibility
        y = y[np.newaxis]
        
        state = state2[np.newaxis]  #full_cards_onehot_like, batch=1
        self.policy_net.fit(state, y, verbose=0)
        return


class DiscardAgent_net6_PI(q_net.DiscardAgent_net6_base):  #6 steps
    #input = onehot*54; output=54*p()
    def __init__(self, hidden_layers_pi, hidden_layers_v, learning_rate, filename_pi, filename_v, reload=False, 
                 gamma=0.2, loss1=tf.losses.categorical_crossentropy, epsilon=0.0, flash_t=False ):
        super().__init__(learning_rate, epsilon=epsilon, gamma=gamma, flash_t=flash_t)
        
        self.filename_pi = filename_pi
        self.filename_v  = filename_v

        if ( reload == True ):
            self.policy_net = self.load_model(filename_pi, loss1)
            self.v_net = self.load_model(filename_v)
            self.policy_net.summary()
            self.v_net.summary()
        else:
            input_size=54
            #hidden_layers = [[256, 0.2], [64, 0.2]]
            #hidden_layers = [[4096, 0.2], [512, 0.2], [128, 0.2]]
            hidden_layers = hidden_layers_pi
            output_size=54
            activation=tf.nn.relu
            loss=loss1 #tf.losses.categorical_crossentropy  # tf.losses.mse
            output_activation=tf.nn.softmax
            learning_rate = learning_rate #0.00001
            self.policy_net = self.build_network(input_size, hidden_layers, output_size,
                                                 activation, loss, output_activation, learning_rate)
    
            input_size=54
            #hidden_layers = [[1024, 0.2], [128, 0.2]]
            hidden_layers = hidden_layers_v
            output_size=1
            activation=tf.nn.relu
            loss=tf.losses.mse
            output_activation=None
            learning_rate = learning_rate #0.00001
            self.v_net = self.build_network(input_size, hidden_layers, output_size,
                                            activation, loss, output_activation, learning_rate)
        return

    def decide(self, state2, train=True):
        b = 1 #dummy for behavior
        if np.random.rand() < self.epsilon and train==True:
            #TBD: here is off-policy like. should not epsilon in in-policy alg !!
            oindex = np.where(state2 > 0)[0]  # '!=' => '>' due to -1, 10, 100
            q_max_index = np.random.choice(len(oindex))
            q_max_oindex = oindex[q_max_index]
            #b = 1.0/len(oindex) #in-policy never use it (b)
        else:
            q_max_oindex, action0_index, action0 = super().decide_6(self.policy_net, state2)
        
        if True == train :
            return q_max_oindex, b
        else:
            return q_max_oindex, b, action0_index, action0
        

    def decide_onego(self, state2):  #collect 6 cards once. state2 size MUST be 18
        q_max_oindex, action0_index, action0 = super().decide_1(self.policy_net, state2)
        return q_max_oindex, action0_index, action0

    def pre_learn_G(self, trajectory0): #G support batch=1 only
        #manual reward
        Gs = []
        Gs_gamma = []
        G = 0
        trajectory = np.array(trajectory0)
        state2s = trajectory[:,0].tolist()
        state2s = np.array(state2s)
        
        vs = self.v_net.predict(state2s)
        vs = vs.reshape(-1)
        T = trajectory.shape[0]  #6
        
        reward = 0
        for t, step_list in enumerate(reversed(trajectory0)): #[::-1]: 0-5
            state2, action, reward0, behavior = step_list[0], step_list[1], step_list[2], step_list[5]

            #if t == 0: #last R=1. have to use the 1 step reward rather than 6 step reward
            #    reward = reward0   #extend the last R to all 6 steps. 
            reward = reward0
            cards_counter = state2.sum()
            G =  reward + self.gamma * G
            G_gamma = G * self.gamma **(T-t-1)
            vs[T-t-1] = vs[T-t-1] * self.gamma **(T-t-1)
            Gs.insert(0, G)
            G_gamma -= vs[T-t-1]   #baseline
            G_gamma /= behavior  #cum_behavior  # **t: diff to ch7.py
            Gs_gamma.insert(0, G_gamma)
            
        Gs = np.array(Gs)[:, np.newaxis]
        Gs_gamma = np.array(Gs_gamma)[:, np.newaxis]
        actions = trajectory[:,1].tolist()
        
        discard_possibility0 = np.eye(54)[actions]
        discard_possibility = discard_possibility0 * Gs_gamma
        
        #print(Gs.shape, state2s.shape, discard_possibility.shape)
        his = self.policy_net.fit(state2s, discard_possibility, verbose=0)
        self.v_net.fit(state2s, Gs, verbose=0)
        return his

    def save_models(self):
        super().save_model(self.policy_net, self.filename_pi)
        super().save_model(self.v_net, self.filename_v)


class OffPolicyDiscardAgent_net6_PI(DiscardAgent_net6_PI):  #6 steps
    def __init__(self, hidden_layers_pi, hidden_layers_v, learning_rate, filename_pi, filename_v, 
                 reload=False, gamma=0.2):
        self.agent_b = RandomAgent()
        
        def my_loss(y_true, y_pred): #YDL: 就是对loss=dot(),对theta求梯度 
            # y_true = y = (df['psi'] / df['behavior']) = (gamma^t * Gt) / b(A|S)
            # y_pred = pi(A|S,theta)
            # - y_true * y_pred 就是ch7.3的theta更新公式. 这样用tf的loss函数， miao!!!
            # loss(theta),对theta求梯度 = -(gamma^t * Gt) / b(A|S) * delta(pi(A|S,theta))
            loss_b = -tf.reduce_sum(y_true * y_pred, axis=-1)
            return loss_b
        
        super().__init__(hidden_layers_pi, hidden_layers_v, learning_rate, filename_pi, filename_v, 
                         reload=reload, gamma=gamma, loss1=my_loss)
        
    def decide(self, state2, train=True):
        b = 1
        if True == train :
            action, b = self.agent_b.decide(state2)
        else:
            action, _, action0_index, action0 = super().decide(state2, train=False)

        if True == train :
            return action, b
        else:
            return action, b, action0_index, action0
        
    
