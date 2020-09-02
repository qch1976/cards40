
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
import cards_env_3 as ce
import discard_q_net_keras_3 as q_net
import discard_pi_net_keras_3 as pi_net





def test_MC_result(env, agent, state2, best_discards_oindex, train=True):
        #best_discards_oindex = [16, 18, 34, 41, 42, 43]  #oindex
        #net only predict 1 card as best candidate in corss-entropy mode.
        #onego works only in mse mode. but uncompatible to auto-playing
        metrix = []
        interim_result = []
        ydl_cards_checker = state2.sum()
        #print(ydl_cards_checker)
        while len(interim_result) < 6 :
            interim_result0, _, action0_index, action0 = agent.decide(state2, train=False)
            action = interim_result0
            next_state, _, _, _ = env.discard_step2(action)
            interim_result.append(interim_result0)

            if False == train:
                if len(interim_result) in [1,6]:
                    set3 = set(action0_index)
                    covered_by_18 = len(set3 & env.oindex_full)
                    metrix.append(action0_index)
                    metrix.append(covered_by_18)
                    metrix.append(action0)  #only collect the 1st and last status
                if len(interim_result) == 6:
                    set4 = set(np.where(state2 > 0)[0]) #last state2 with 13 cards
                    covered_by_13 = len(set3 & set4) #within 13
                    metrix.append(covered_by_13)

            state2 = next_state

        interim_result.sort()

        set1 = set(best_discards_oindex)
        set2 = set(interim_result)
        covered_by_best = len(set1 & set2)
        
        return interim_result, covered_by_best, metrix


def test_MC_result_onego(env, agent, state2, best_discards_oindex, train=True):
        #best_discards_oindex = [16, 18, 34, 41, 42, 43]  #oindex
        #net only predict 1 card as best candidate in corss-entropy mode.
        #onego works only in mse mode. but uncompatible to auto-playing
        metrix = []
        ydl_cards_checker = state2.sum()
        #print(ydl_cards_checker)

        interim_result, action0_index, action0 = agent.decide_onego(state2)
        interim_result.sort()

        set1 = set(best_discards_oindex)
        set2 = set(interim_result)
        covered_by_best = len(set1 & set2)

        if False == train:
            set3 = set(action0_index)
            covered_by_18 = len(set3 & env.oindex_full)
            metrix = [action0_index, covered_by_18, action0, [], [], [], []]
            
        return interim_result, covered_by_best, metrix


def play_qlearning_diff(agent, trajectory):
    history = agent.pre_learn_diff(trajectory)
    trajectory.clear()
    return history
    
def play_qmax_MC(agent, trajectory):
    history = agent.pre_learn_G(trajectory)
    trajectory.clear()
    return history

def play_dump(agent, trajectory): #need onego as test and exam method
    history = agent.pre_learn_dump(trajectory)
    trajectory.clear()
    return history

def play_pi_G(agent, trajectory):  #with baseline
    history = agent.pre_learn_G(trajectory)
    trajectory.clear()
    return history

def play_pi_G_behavior(agent, trajectory):  #with baseline
    history = agent.pre_learn_G(trajectory)
    trajectory.clear()
    return history


ticks_d_1_diff = 0
ticks_d_2_diff = 0
ticks_d_3_diff = 0
ticks_d_4_diff = 0
def play_MC_alg(env, agent, learning_f, test_f, epoch=500, to_best=6, batches=1):
    global ticks_d_1_diff, ticks_d_2_diff, ticks_d_3_diff, ticks_d_4_diff
    trajectory = []
    his_matrix = []
    covered_by_best = 0
    for e in range(epoch):

        ticks1 = time.time()

        state2, best_discards_oindex, _, reward = env.reset(keep=True, render=False)
        if learning_f == play_dump:
            #train by 'best' directly
            trajectory.append([state2, best_discards_oindex, [reward]*6 ])
        else:
            while True:
                ticks11 = time.time()
                action, behavior = agent.decide(state2)  #6 steps only, no onego in train
                ticks2 = time.time()
                ticks_d_3_diff += ticks2 - ticks11
                
                ticks12 = time.time()
                next_state2, reward, done, _ = env.discard_step2(action)
                ticks2 = time.time()
                ticks_d_4_diff += ticks2 - ticks12
    
                trajectory.append([state2, action, reward, next_state2, done, behavior])
                state2 = next_state2
                if done:
                    break

        ticks2 = time.time()
        ticks_d_1_diff += ticks2 - ticks1
        
        if e%200 == 0:
            ydl_stop = 1
        if e % batches == 0 or e >= (epoch-1):
            ticks1 = time.time()
            history = learning_f(agent, trajectory)
            #print(history.history)
            his_matrix.append(history.history["loss"])
            #his_matrix.append(history.history["acc"]) #ydl_measure"])

            ticks2 = time.time()
            ticks_d_2_diff += ticks2 - ticks1
            
            if to_best > 0 :
                ticks1 = time.time()
                state2, best_discards_oindex, _, _ = env.reset(keep=True, render=False)
                interim_result, covered_by_best, _ = test_f(env, agent, state2, best_discards_oindex)
                ticks2 = time.time()
                ticks_d_2_diff += ticks2 - ticks1
                print("inner epoch ", e, interim_result, covered_by_best)
                if covered_by_best >= to_best :
                    break #e

    print("episod steps to complance:", e, covered_by_best, to_best)
    return interim_result, e, covered_by_best, np.mean(np.array(his_matrix)), history.history["loss"][0]  #last just has 1 value




class TestBench:
    def __init__(self, parameters, reload0=False):
#0: id,  train-method,   test-method,   exam-method,  net-conf_pi,
#5: lr,  R(*10), to-best, batch, net_input_format, 
#10: separator, flash_t,  epsilon, net-conf_v,  gamma,
        
        self.para_id = parameters[0]
        self.train_f = parameters[1]
        self.test_f  = parameters[2]
        self.exam_f  = parameters[3]
        self.hidden_layers_1 = parameters[4]
        self.learning_rate   = parameters[5]
        self.reward_times = parameters[6]
        self.be_best      = parameters[7]
        self.batches      = parameters[8]
        self.net_input_format = parameters[9]
        self.flash_t      = parameters[11]
        self.epsilon      = parameters[12]
        self.hidden_layers_2 = parameters[13]
        self.gamma       = parameters[14]
        self.outer_epoch = parameters[15]
        self.inner_epoch = parameters[16]
        
        #"checkpoints/route-rnn-{}.ckpt".format(index)
        
        if self.train_f == play_qmax_MC :
            filename_1 = 'q_MC_e'
            filename_2 = 'q_MC_t'
        elif self.train_f == play_qlearning_diff :
            filename_1 = 'q_diff_e'
            filename_2 = 'q_diff_t'
        elif self.train_f == play_dump:
            filename_1 = 'q_dump_e'
            filename_2 = 'q_dump_t'
        elif self.train_f == play_pi_G :
            filename_1 = 'pi_G_pi'
            filename_2 = 'pi_G_v'
        elif self.train_f == play_pi_G_behavior :
            filename_1 = 'pi_G_b_pi'
            filename_2 = 'pi_G_b_v'
        else:
            filename_1 = 'ydl_1'
            filename_2 = 'ydl_2'

        if self.test_f == test_MC_result :
            filename_1 += '_t6'
            filename_2 += '_t6'
        elif self.test_f == test_MC_result_onego:
            filename_1 += '_t1'
            filename_2 += '_t1'
        else:
            filename_1 += '_ydl1'
            filename_2 += '_ydl1'

        if self.exam_f == test_MC_result:
            filename_1 += '_e6_'
            filename_2 += '_e6_'
        elif self.exam_f == test_MC_result_onego:
            filename_1 += '_e1_'
            filename_2 += '_e1_'
        else:
            filename_1 += '_ydl2_'
            filename_2 += '_ydl2_'

        filename_1 += str(self.para_id) + '.h5'
        filename_2 += str(self.para_id) + '.h5'

        if self.train_f in [play_qmax_MC, play_qlearning_diff, play_dump]:
            self.env = ce.PokerEnvironment_6_1(reward_times=self.reward_times, input_format=self.net_input_format )
            self.agent = q_net.DiscardAgent_net6_Qmax2(self.hidden_layers_1, self.learning_rate, filename_1, filename_2, epsilon=self.epsilon, reload=reload0, flash_t=self.flash_t)
            
        elif self.train_f == play_pi_G :
            self.env = ce.PokerEnvironment_6_1(reward_times=self.reward_times)
            self.agent = pi_net.DiscardAgent_net6_PI(self.hidden_layers_1, self.hidden_layers_2, self.learning_rate, filename_1, filename_2, gamma=self.gamma, reload=reload0, epsilon=self.epsilon)
            
        elif self.train_f == play_pi_G_behavior :
            self.env = ce.PokerEnvironment_6_1(reward_times=self.reward_times)
            self.agent = pi_net.OffPolicyDiscardAgent_net6_PI(self.hidden_layers_1, self.hidden_layers_2, self.learning_rate, filename_1, filename_2, gamma=self.gamma, reload=reload0)
            
        else:
            print("wrong train_f", self.train_f)
            return
            
        print("param id and ID env and agent ", self.para_id, id(self.env), id(self.agent))

        
    def update_parameters(self, parameters):
        self.__init__(parameters)
    
    def train(self, param_id, episodes=10):
        matrix = []
        for e in range(self.outer_epoch):
            np.random.seed(13)  #keep same random sequence in every epoch
            rd.seed(13) 
            print("train outer epoch ", param_id, e)
            for i in range(episodes): #玩多少盘; inner_epoch=为达到best,同一盘重复
                state2, best_discards_oindex, _, _ = self.env.reset(keep=False) #new card deal
                discard_cards, convergency, covered_by_best, his_mean, his_last = play_MC_alg(self.env, self.agent, self.train_f, self.test_f, epoch=self.inner_epoch, to_best=self.be_best, batches=self.batches)
                matrix.append([best_discards_oindex, discard_cards, covered_by_best, convergency, his_mean, his_last])
        self.agent.save_models()
        np_matrix = np.array(matrix)
        print("train result: paramid episode ", param_id, episodes, "averag covered_by_best ", np.mean(np_matrix[:, 2]))
        return matrix

        
    def test(self, param_id, episodes=10):
        matrix = []
        for i in range(episodes):
            state2, best_discards_oindex, _, _ = self.env.reset(keep=False) #new card deal
            discard_cards, covered_by_best, top6 = self.test_f(self.env, self.agent, state2, best_discards_oindex, train=False)
            matrix.append([best_discards_oindex, discard_cards, covered_by_best, top6])
        np_matrix = np.array(matrix)
        print("test result: paramid episode ", param_id, episodes, "averag best ", np_matrix[:,2].mean())
        return matrix

        
    def exam(self, param_id, episodes=1):
        matrix = []
        for i in range(episodes):
            state2, best_discards_oindex, _, _ = self.env.reset(keep=False) #new card deal
            interim_result, covered_by_best, top6 = self.exam_f(self.env, self.agent, state2, best_discards_oindex, train=False)
            matrix.append([best_discards_oindex, interim_result, covered_by_best, top6])
        np_matrix = np.array(matrix)
        print("exam result: paramid episode ", param_id, episodes, "averag best ", np_matrix[:,2].mean())
        return matrix


parameters_set=[]
#note1: net_input_format: 'not in-hand' MUST BE: <=0
#note2: G alg: batch MUST BE 1

'''
#Q                   #id,  train-method,        test-method,          exam-method,           net-conf_pi,                            lr,       R(*10), to-best, batch, net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer, inner
parameters_set.append([0,  play_qlearning_diff, test_MC_result,       test_MC_result,        [[256, 0.2]],                           0.01,     5,      6,       1,     [0, -1, 1,   2],  '#',       True,     0.2,     [],                         0,     1,     100 ])
parameters_set.append([1,  play_qlearning_diff, test_MC_result,       test_MC_result,        [[1024, 0.2], [256, 0.2]],              0.01,     5,      6,       1,     [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     1,     100 ])
parameters_set.append([2,  play_qmax_MC,        test_MC_result,       test_MC_result,        [[1024, 0.2], [256, 0.2]],              0.01,     5,      6,       1,     [0, -1, 1,   2],  '#',       True,     0.2,     [],                         0,     1,     100 ])
parameters_set.append([3,  play_qlearning_diff, test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.01,     5,      6,       1,     [0,  0, 0.5, 1],  '#',       False,    0.2,     [],                         0,     1,     100 ])
parameters_set.append([4,  play_qmax_MC,        test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     5,      6,       1,     [0,  0, 1,   1],  '#',       True,     0.2,     [],                         0,     1,     100 ])
parameters_set.append([5,  play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.01,     5,      6,       1,     [0, -1, 2,   1],  '#',       False,    0.2,     [],                         0,     1,     100 ])
#PI                  #id,  train-method,        test-method,          exam-method,           net-conf_pi,                            lr,       R(*10), to-best, batch, net_input_format, separator, NNNNNNN,  epsilon, net-conf_v,                 gamma, outer, inner
parameters_set.append([6,  play_pi_G,           test_MC_result,       test_MC_result,        [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.00001,  5,      6,       1,     [0,  0, 1,   1],  '#',       False,    0.5,     [[1024, 0.2], [128, 0.2]],  0.2,   1,     10  ])
parameters_set.append([7,  play_pi_G_behavior,  test_MC_result,       test_MC_result,        [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.00001,  5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0,       [[1024, 0.2], [128, 0.2]],  0.2,   1,     500  ])
parameters_set.append([8,  play_pi_G_behavior,  test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.00001,  5,      6,       1,     [0,  0, 0.5, 1],  '#',       False,    0,       [[1024, 0.2], [128, 0.2]],  0.2,   1,     100 ])
'''

########################################
# 1. net scale impact
#Q                   #id,    train-method,        test-method,          exam-method,           net-conf_pi,                            lr,       R(*10), to-best, batch, net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer, inner
parameters_set.append([0,    play_qlearning_diff, test_MC_result,       test_MC_result,        [[256, 0.2]],                           0.01,     5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     100,     1 ])
parameters_set.append([1,    play_qlearning_diff, test_MC_result,       test_MC_result,        [[1024, 0.2], [256, 0.2]],              0.01,     5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     100,     1 ])
parameters_set.append([2,    play_qlearning_diff, test_MC_result,       test_MC_result,        [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.01,     5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     100,     1 ])
parameters_set.append([3,    play_qmax_MC,        test_MC_result,       test_MC_result,        [[256, 0.2]],                           0.01,     5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     100,     1 ])
parameters_set.append([4,    play_qmax_MC,        test_MC_result,       test_MC_result,        [[1024, 0.2], [256, 0.2]],              0.01,     5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     100,     1 ])
parameters_set.append([5,    play_qmax_MC,        test_MC_result,       test_MC_result,        [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.01,     5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     100,     1 ])
parameters_set.append([6,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[256, 0.2]],                           0.01,     5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     100,     1 ])
parameters_set.append([7,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     100,     1 ])
parameters_set.append([8,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.01,     5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     100,     1 ])

#PI                  #id,    train-method,        test-method,          exam-method,           net-conf_pi,                            lr,       R(*10), to-best, batch, net_input_format, separator, NNNNNNN,  epsilon, net-conf_v,                 gamma, outer, inner
parameters_set.append([9,    play_pi_G,           test_MC_result,       test_MC_result,        [[256, 0.2]],                           0.00001,  5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.0,     [[256,  0.2]],              0.2,   100,     1 ])
parameters_set.append([10,   play_pi_G,           test_MC_result,       test_MC_result,        [[1024, 0.2], [256, 0.2]],              0.00001,  5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.0,     [[512,  0.2], [32,  0.2]],  0.2,   100,     1 ])
parameters_set.append([11,   play_pi_G,           test_MC_result,       test_MC_result,        [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.00001,  5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0.0,     [[1024, 0.2], [128, 0.2]],  0.2,   100,     1 ])
parameters_set.append([12,   play_pi_G_behavior,  test_MC_result,       test_MC_result,        [[256, 0.2]],                           0.00001,  5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0,       [[256,  0.2]],              0.2,   100,     1 ])
parameters_set.append([13,   play_pi_G_behavior,  test_MC_result,       test_MC_result,        [[1024, 0.2], [256, 0.2]],              0.00001,  5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0,       [[512,  0.2], [32,  0.2]],  0.2,   100,     1 ])
parameters_set.append([14,   play_pi_G_behavior,  test_MC_result,       test_MC_result,        [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.00001,  5,      6,       1,     [0, -1, 1,   2],  '#',       False,    0,       [[1024, 0.2], [128, 0.2]],  0.2,   100,     1 ])


                       
def save_csv(param_id, his, overwrite):
    train_measures = his["train"]
    test_measures  = his["test"]
    exam_measures  = his["exam"]
    #np_tt = np.concatenate((np_train, np_test),axis=1)
    #df_train = pd.DataFrame(np_train, columns=['convergency', 'best', 'loss_mean', 'loss_last', 'test_inphase'])
    
    ###########################
    # train storing
    df_train = pd.DataFrame(train_measures, columns=['expect best', 'last epoch', 'covered', 'convergency', 'loss mean', 'last loss'])

    
    ###########################
    # test storing
    line = []
    test_to_df = []
    for test_meas in test_measures:
        line = test_meas[0:3]

        test_meas_3 = test_meas[3]
        line.append(test_meas_3[0]) #1st 8
        line.append(test_meas_3[1]) #1st 8, covered by 18
        
        ydl = np.round(test_meas_3[2], 7)
        line.append(ydl) #1st 8, possibility
        
        if len(test_meas_3[3]) > 0:
            line.append(test_meas_3[3]) #6th 8
            line.append(test_meas_3[4]) #6th 8, covered by 18
            line.append(test_meas_3[6]) #6th 8, covered by 13
        else:
            line.append("[]")
            line.append("[]")
            line.append("[]")
            
        if len(test_meas_3[5]) > 0:
            ydl = np.round(test_meas_3[5], 7)
            line.append(ydl) #6th 8, possibility
        else:
            line.append("[]")
            
        test_to_df.append(line)
        line = []
        
    df_test = pd.DataFrame(test_to_df, columns=['expect best', 'result', 'covered', '1st 8 cards', '1st covered 18', '1st 8 possibilty', 'last 8 cards', 'last cover 18', 'last cover 13', 'last 8 possibility'])

    ###########################
    # exam storing
    line = []
    exam_to_df = []
    for exam_meas in exam_measures:
        line = exam_meas[0:3]

        exam_meas_3 = exam_meas[3]
        line.append(exam_meas_3[0]) #1st 8
        line.append(exam_meas_3[1]) #1st 8, covered by 18
        
        ydl = np.round(exam_meas_3[2], 7)
        line.append(ydl) #1st 8, possibility
        
        if len(exam_meas_3[3]) > 0:
            line.append(exam_meas_3[3]) #6th 8
            line.append(exam_meas_3[4]) #6th 8, covered by 18
            line.append(exam_meas_3[6]) #6th 8, covered by 13
        else:
            line.append("[]")
            line.append("[]")
            line.append("[]")
            
        if len(exam_meas_3[5]) > 0:
            ydl = np.round(exam_meas_3[5], 7)
            line.append(ydl) #6th 8, possibility
        else:
            line.append("[]")
            
        exam_to_df.append(line)
        line = []

    df_exam = pd.DataFrame(exam_to_df, columns=['expect best', 'result', 'covered', '1st 8 cards', '1st covered 18', '1st 8 possibilty', 'last 8 cards', 'last cover 18', 'last cover 13', 'last 8 possibility'])



    train_filename = 'train-' + str(param_id) + '.csv'
    test_filename = 'test-' + str(param_id) + '.csv'
    exam_filename = 'exam-' + str(param_id) + '.csv'
    
    if True == overwrite:
        df_train.to_csv(train_filename, index=False)
        df_test.to_csv(test_filename, index=False)
        df_exam.to_csv(exam_filename, index=False)
    else:
        df_csv_train = pd.read_csv(train_filename)
        df_csv_test = pd.read_csv(test_filename)
        df_csv_exam = pd.read_csv(exam_filename)
        print("df_csv_t/t, df_csv_exam shape ", df_csv_test.shape, df_csv_exam.shape)
        df_csv_train = df_csv_train.append(df_train)
        df_csv_test = df_csv_test.append(df_test)
        df_csv_exam = df_csv_exam.append(df_exam)
        df_csv_train.to_csv(train_filename, index=False)
        df_csv_test.to_csv(test_filename, index=False)
        df_csv_exam.to_csv(exam_filename, index=False)

       
#initial
def initial_h5(self_test, exam_test, selected_p_set):
    #measurements={}
    print("initial H5 creation\ninitial H5 creation\ninitial H5 creation\ninitial H5 creation\n")
    for i, parameters in enumerate(parameters_set):
        if parameters[0] not in selected_p_set:
            continue;
            
        his = {}
        q_init_test_bench = TestBench(parameters, reload0=False)
        print("ID q_init_test_bench ", i, parameters[0], id(q_init_test_bench))
        
        np.random.seed(13)  #keep same random sequence
        rd.seed(13) 
        matrix = q_init_test_bench.train(parameters[0], episodes=self_test)
        print("CPU cost: ", i, ticks_d_1_diff, ticks_d_2_diff, ticks_d_3_diff, ticks_d_4_diff)
        his["train"] = matrix
    
        np.random.seed(13)  #keep same random sequence
        rd.seed(13) #test set=train set
        matrix = q_init_test_bench.test(parameters[0], episodes=self_test)
        his["test"] = matrix
    
        #no re-seed() needed in exam. generate diff sequence
        np.random.seed(179)  #use diff random sequence
        rd.seed(179) 
        matrix = q_init_test_bench.exam(parameters[0], episodes=exam_test)
        his["exam"] = matrix
        
        #measurements[parameters[0]] = his
        save_csv(parameters[0], his, True)


#pause
#continuous
def resume_h5(self_test, exam_test, selected_p_set):
    measurements={}
    print("resume H5 ... \nresume H5 ... \nresume H5 ... \nresume H5 ... \nresume H5 ... \n")
    for i, parameters in enumerate(parameters_set):
        if parameters[0] not in selected_p_set:
            continue;
            
        his = {}
        q_infinit_test_bench = TestBench(parameters, reload0=True)  #reload .h5
        print("ID q_infinit_test_bench ", i, parameters[0], id(q_infinit_test_bench))
        
        np.random.seed(13)  #keep same random sequence
        rd.seed(13) 
        matrix = q_infinit_test_bench.train(parameters[0], episodes=self_test)
        print("CPU cost: ", i, ticks_d_1_diff, ticks_d_2_diff, ticks_d_3_diff, ticks_d_4_diff)
        his["train"] = matrix
        
        np.random.seed(13)  #keep same random sequence
        rd.seed(13) #test set=train set
        matrix = q_infinit_test_bench.test(parameters[0], episodes=self_test)
        his["test"] = matrix
    
        #no re-seed() needed in exam. generate diff sequence
        np.random.seed(179)  #use diff random sequence
        rd.seed(179) 
        matrix = q_infinit_test_bench.exam(parameters[0], episodes=exam_test)
        his["exam"] = matrix
        
        #measurements[parameters[0]] = his
        save_csv(parameters[0], his, False)



    
import sys, getopt

def main(argv):
    print("argv ", argv)

    try:
        opts, args = getopt.getopt(argv,"r:s:e:p:")
    except getopt.GetoptError:
        print("wrong input")
        return;

    try:
        for opt, arg in opts:
            print("arg ",opt, arg)
            if opt == '-r':
                if arg == 'init' :
                    h5_func_f = initial_h5
                elif arg == 'resume' :
                    h5_func_f = resume_h5
                else:
                    print("wrong -r input", opt, arg)
                    return;
            
            if opt == '-s':
                self_test = int(arg)
            
            if opt == '-e':
                exam_test = int(arg)

            if opt == '-p':
                selected_p_set0 = arg.split(',')
                selected_p_set = [int(c) for c in selected_p_set0]
                
    except  ValueError:
            print("wrong input", opt, arg)
            return

    h5_func_f(self_test, exam_test, selected_p_set)
    return


if __name__ == "__main__":
   main(sys.argv[1:])