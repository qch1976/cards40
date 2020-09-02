import time
import numpy as np
import pandas as pd
import random as rd
from enum import Enum
from random import shuffle as rshuffle
import tensorflow.compat.v2 as tf
from tensorflow import keras

import deal_cards as dc
import cards_env_3 as ce


def ydl_measure(y_true, y_pred):
    loss_b = -tf.reduce_sum(y_true * y_pred, axis=-1)
    return loss_b



class DiscardAgent_net6_base:  #6 steps
    def __init__(self, learning_rate, epsilon=0.2, gamma=0.0, flash_t=False):
        self.gamma = gamma
        self.epsilon = epsilon
        self.flash_t = flash_t

    def build_network(self, input_size, hidden_layers, output_size,
                             activation, loss, output_activation=None, learning_rate=0.01, metrics=['accuracy']): # 构建网络

        model = keras.Sequential()
        model.add(keras.layers.Dense(units=input_size, input_shape=(input_size,), activation=activation))

        for layer, [hidden_size, dropout] in enumerate(hidden_layers):
            model.add(keras.layers.Dense(units=hidden_size, activation=activation))
            model.add(keras.layers.Dropout(dropout))

        model.add(keras.layers.Dense(units=output_size, activation=output_activation)) # 输出层
        optimizer = tf.optimizers.Adam(lr=learning_rate)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics) #['accuracy']), his['acc']
        model.summary()
        return model

    def decide_1(self, net, state2):  #collect 6 cards once. state2 size MUST be 18
        oindex = np.where(state2 > 0)[0]
        state = state2[np.newaxis]  #full_cards_onehot_like, batch=1

        qs0 = net.predict(state).reshape(-1)
        qs = qs0[oindex]  #oindex is action
        q_max_index = np.argsort(-qs)  #(-):bigger -> smaller
        q0_max_index = np.argsort(-qs0)  #(-):bigger -> smaller

        q_max_oindex = oindex[q_max_index[0:6]]
        return q_max_oindex, q0_max_index[0:8], qs0[q0_max_index[0:8]]

    def decide_6(self, net, state2):
        oindex = np.where(state2 > 0)[0]  # '!=' => '>' due to -1, 10, 100
        state = state2[np.newaxis]  #full_cards_onehot_like, batch=1
        actions0 = net.predict(state).reshape(-1)  #oindex(54,), action0(1,54)->(54)
        actions1 = actions0[oindex]
        action_index = np.argmax(actions1)
        action0_max_index = np.argsort(-actions0)  #(-):bigger -> smaller

        action = oindex[action_index]
        return action, action0_max_index[0:8], actions0[action0_max_index[0:8]]
    

    def save_model(self, net, filename):
        net.save(filename)
        ydl = net.get_weights()

    def load_model(self, filename, my_loss=0):
        net = keras.models.load_model(filename, custom_objects={'my_loss': my_loss}) #{'ydl_loss': ydl_loss}, NAME MUST BE SAME
        ydl = net.get_weights()
        return net


class DiscardAgent_net6_Qmax2(DiscardAgent_net6_base):  #6 steps
    def __init__(self, hidden_layers, learning_rate, filename_e, filename_t, epsilon=0.2, gamma=0.0, reload=False, flash_t=False):
        super().__init__(learning_rate, epsilon=epsilon, gamma=gamma, flash_t=flash_t)

        self.filename_e = filename_e
        self.filename_t = filename_t
        
        if ( reload == True ):
            self.qe_net = self.load_model(filename_e)
            self.qt_net = self.load_model(filename_t)
            self.qe_net.summary()
            self.qt_net.summary()
        else:                
            input_size=54
            #hidden_layers = [[512, 0.2], [128, 0.2]]
            output_size=54
            activation=tf.nn.relu
            loss=tf.losses.mse
            output_activation=None
            #learning_rate = 0.01
            self.qe_net = self.build_network(input_size, hidden_layers, output_size,
                                            activation, loss, output_activation, learning_rate)
            self.qt_net = self.build_network(input_size, hidden_layers, output_size,
                                            activation, loss, output_activation, learning_rate)
            self.qt_net.set_weights(self.qe_net.get_weights())
        return


    def decide(self, state2, train=True):
        b = 1 #b=dummy for behavior
        if np.random.rand() < self.epsilon and train==True:
            oindex = np.where(state2 > 0)[0]  # '!=' => '>' due to -1, 10, 100
            q_max_index = np.random.choice(len(oindex))
            q_max_oindex = oindex[q_max_index]
        else:
            q_max_oindex, action0_index, action0 = super().decide_6(self.qe_net, state2)
            
        if True == train :
            return q_max_oindex, b
        else:
            return q_max_oindex, b, action0_index, action0

    def decide_onego(self, state2):  #collect 6 cards once. state2 size MUST be 18
        q_max_oindex, action0_index, action0 = super().decide_1(self.qe_net, state2)
        return q_max_oindex, action0_index, action0

    def pre_learn_diff(self, trajectory0): #diff support batch>1
        trajectory = np.array(trajectory0)
        
        state2s = np.array(trajectory[:,0].tolist())
        actions = np.array(trajectory[:,1].tolist())
        rewards = np.array(trajectory[:,2].tolist())
        next_state2s = np.array(trajectory[:,3].tolist())
        dones = np.array(trajectory[:,4].tolist())
        
        state2s_1 = np.where(state2s>0, 1, 0)  #>0 mean in-hand or trump
        next_state2s_1 = np.where(next_state2s>0, 1, 0)  #>0 mean in-hand or trump

        ###用target model计算U价值
        #next_state2 = (6, 54)
        #next_qs = (6, 54), non-oindex position=0, in predict and train
        #U算法: U=R+Q(next_s,): target model
        #对比之前的迭代算法: q(s,a) += [U - q(s,a)]
        next_qs0 = self.qt_net.predict(next_state2s)  #t
        next_qs = next_qs0 * next_state2s_1  #clear the position that oindex not existing
        
        next_max_qs = next_qs.max(axis=-1)  # =>Q
        #Us里包括了qt_net的变化，不要用qt_net计算梯度 此例子里，没有明显影响
        #如果用tensor定义U和predict的关系，就可能受影响
        Us = rewards + self.gamma * (1. - dones) * next_max_qs  

        ###用evaluate model训练，用U更新 predict出的经验数据
        #evaluate model生成state的q(s,)
        #targets=[6， 54], actions=[6， 54]
        if True == self.flash_t:
            targets = np.zeros([6,54])  #backgroud with 0
        else:
            targets0 = self.qe_net.predict(state2s)
            targets = targets0 * state2s_1  #clear the position that oindex not existing
        targets[np.arange(state2s.shape[0]), actions] = Us

        #update评估网络的w only.
        #用evaluate model的perdict + U update,训练evaluate model
        history = self.qe_net.fit(state2s, targets, verbose=0, batch_size=6*128, epochs=1) #would not be >1
        
        #every episode
        self.qt_net.set_weights(self.qe_net.get_weights())
        return history


    def pre_learn_G(self, trajectory0): #G support batch=1 only
        trajectory = np.array(trajectory0)

        state2s = np.array(trajectory[:,0].tolist())
        actions = np.array(trajectory[:,1].tolist())
        
        state2s_1 = np.where(state2s>0, 1, 0)  #>0 mean in-hand or trump

        Gs = np.array([])
        G = 0
        reward = 0
        for t, [state2, _, reward0, _, _, _] in enumerate(reversed(trajectory0)): #[::-1]: 0-5
            #if t == 0: #last R=1, means win or loss. env.auto_judge_return_estimation(). have to use the 1 step reward rather than 6 step reward
            #    reward = reward0   #extend the last R to all 6 steps. 
            reward = reward0 #step reward
            
            ydl_cards_checker = state2.sum()
            G =  reward + self.gamma * G
            Gs = np.insert(Gs, 0, G)
            #batch=1 MUST be
            
        #Gs = np.array(Gs)

        if True == self.flash_t:
            targets = np.zeros([6,54])  #backgroud with 0
        else:
            targets0 = self.qe_net.predict(state2s) #backgroud with previous predict
            targets = targets0 * state2s_1  #clear the position that oindex not existing
            
        targets[np.arange(state2s.shape[0]), actions] = Gs
        
        history = self.qe_net.fit(state2s, targets, verbose=0, batch_size=6*128)
        return history

    def pre_learn_dump(self, trajectory0):
        state2s = np.array(trajectory0[0][0]).reshape(1,54)
        best_discards_oindex = np.array(trajectory0[0][1])
        rewards = np.array(trajectory0[0][2])

        state2s_1 = np.where(state2s>0, 1, 0)  #>0 mean in-hand or trump
        
        if True == self.flash_t:
            targets = np.zeros([1,54])  #backgroud with 0
        else:
            targets0 = self.qe_net.predict(state2s)
            targets = targets0 * state2s_1  #clear the position that oindex not existing
        targets[np.arange(state2s.shape[0]), best_discards_oindex] = rewards
        
        history = self.qe_net.fit(state2s, targets, verbose=0, batch_size=6*128)
        return history

    def learn(self, state2, action, reward, done, best_discards_oindex):
        return

    def save_models(self):
        super().save_model(self.qe_net, self.filename_e)
        super().save_model(self.qt_net, self.filename_t)


