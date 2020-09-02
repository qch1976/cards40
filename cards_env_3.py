import numpy as np
import pandas as pd
import deal_cards as dc
import collections as cllt

import time
import copy


card_importancy = np.array([0,      #0-invalid
                            13*2,   #1
                            1,      #2
                            2,      #3
                            3,      #4
                            4,      #5
                            5,      #6
                            6,      #7
                            7,      #8
                            8,      #9
                            9*1.2,  #10
                            10*1.5, #11
                            11*2,   #12
                            12*2,   #13
                            15*3,   #BJ
                            16*3    #RJ 
                            ])  #reserve 14 for round trump

master_trump_importancy = { 'master'  : 14*3,
                            'regular' : 14*2 }

player_importancy_scope = [[487.2, 527.7], #max
                           [ 36,   68   ]  #min 
                           ]#12,  18 cards
player_deviation_scope = [ 0.65333, #log(4.501252),  #(1,0,0)~4.501252 (+1), max
                          -2.62468  #log(0.002373)]  #(3,3,3)~0.002373 (-1), min
                          ]

#偏离度， ~3=12/4, 实测结果 x=normal(2.88902, 1.35)的正态分布， log(x)
#每种花色的张数                   normal()        log()
norm_p_entropy_table = { 0: [0.029930572,   1.523884976],
                         1: [0.111021978,   0.954591037],
                         2: [0.237906818,   0.623593111],
                         3: [0.294515939,   0.530891196],
                         4: [0.210627322,   0.676485294],
                         5: [0.087021106,   1.060375403],
                         6: [0.020770095,   1.682561525],
                         7: [0.00286389,    2.543043658],
                         8: [0.000228128,   3.641821804],
                         9: [1.04979e-05,   4.978895961],
                         10: [2.79083e-07,  6.554266131],
                         11: [4.28615e-09,  8.367932312],
                         12: [3.80282e-11,  10.41989451],
                         13: [1.94916e-13,  12.71015271] }


class PokerEnvironment:
    def __init__(self):
        self.full_poker = dc.FullPokers()
        self.state = ''
        self.state2 = np.array([])
        self.banker = dc.Players.NONE
        self.df_saver = pd.DataFrame()
        self.cards_status = {}  #trump rating, non-trump rating, uneven of non-trump rating
        self.oindex_pattern = np.array([])
        self.allowed_discarded_score = 0
        self.best_discards_oindex = []   #oindex ordered
        self.best_discards_priority_sorted = np.array([])  #pri ordered
        self.time_cost = np.zeros(8, dtype='float')  #for performance measure
        self.oindex_full = []


    def reset(self, start_player=dc.Players.SOUTH, keep=True, render=False): #keep: stay in same deal for initial debug
        ######################
        #  clean cards, shuffle, set trump and round and banker
        ######################
        if True == keep : #stay in same card deal
            if dc.Players.NONE == self.banker : # first running
                self.full_poker.__init__()
                dc.deal_first_round(self.full_poker, render)
                self.banker = start_player
                self.full_poker.save_cards_status()
                #self.full_poker.render()  #stop for optm
            else: # repeat running
                #self.full_poker.reset_df_cards()   #stop for optm
                _ = self.full_poker.load_cards_status()
        else: # free run
            self.banker = start_player
            self.full_poker.__init__()
            dc.deal_first_round(self.full_poker, render)
            self.full_poker.save_cards_status()

        ######################
        #  optm for player card bufferring
        ######################
        #example, only south = banker
        if self.df_saver.empty == True or False == keep: #for optm
            self.df_start_player_cards = self.full_poker.get_player_cards_for_discard(start_player)
            self.df_start_player_cards.sort_values(by=["suit", "name"] , ascending=[True, True], inplace=True)
            self.df_saver = self.df_start_player_cards.copy(deep=True)
        else:
            self.df_start_player_cards = self.df_saver.copy(deep=True)

        ######################
        #  initial state by string
        ######################
        #stop for optm #self.df_start_player_cards.sort_values(by=["suit", "name"] , ascending=[True, True], inplace=True)
        #print(self.df_start_player_cards)
        self.state = "" #self.full_poker.generate_pi_str_key(self.df_start_player_cards)
        
        ######################
        #  rating
        ######################
        if False == keep or 0 == len(self.cards_status): #stay in same card deal
            player_importency, player_deviation, suit_length, priorities = self.create_cards_status_evaluation()
            self.cards_status['importency'] = player_importency  #regularity player_importency, (0,1]
            self.cards_status['deviation'] = player_deviation
            self.cards_status['length'] = suit_length
            self.cards_status['priority'] = priorities  #data in memory in correct

        ######################
        # pattern for search AKQJ
        ######################
        if False == keep or 0 == len(self.oindex_pattern): #stay in same card deal
            self.oindex_pattern = self.create_AKQJ_pattern()
    
        ######################
        # max score can be discarded based on card status
        ######################
        if False == keep or 0 == self.allowed_discarded_score: #stay in same card deal
            self.allowed_discarded_score = self.create_allowed_score(self.cards_status['importency'], self.cards_status['deviation'])
        
        ######################
        # create best discard cards
        ######################
        if False == keep or 0 == len(self.best_discards_oindex): #stay in same card deal
            self.best_discards_oindex = self.create_best_discards()
            self.best_discards_priority_sorted = self.priority_ordered_best_discards()
        
        ######################
        #  initial state by onehot
        ######################
        full_cards_onehot_like = np.zeros(54)
        oindex = self.df_start_player_cards['oindex'].values.astype('int32')
        self.oindex_full = set(oindex)
        #print(id(oindex), id(self.oindex_full))
        full_cards_onehot_like[oindex] = self.net_input_format[2]
        #select trump
        df_trumps = self.df_start_player_cards.loc[self.df_start_player_cards['trumps']==True]
        oindex = df_trumps['oindex'].values.astype('int32')
        full_cards_onehot_like[oindex] = self.net_input_format[3]
        
        self.state2 = full_cards_onehot_like
        return self.state2, self.best_discards_oindex, self.df_start_player_cards, self.reward_times_10
        
    def create_cards_status_evaluation(self):
        ######################
        # importency 
        ######################
        master_trump = self.full_poker.trump  #suit
        total_importancy = 0
        total_deviation = 0  #non trump only
        total_cards = 0  #non trump only
        suit_lens = {}  #
        
        df_trumps = self.df_start_player_cards.loc[self.df_start_player_cards['trumps']==True]
        total_importancy += self.sum_cards_importancy(df_trumps)*1.5 #added weight for trump cards
        suit_lens['trump'] = df_trumps.shape[0]
        
        for suit in dc.CardSuits :
            if suit == master_trump or (suit in [dc.CardSuits.NONE, dc.CardSuits.BLACKJOKER, dc.CardSuits.REDJOKER]) :
                continue

            df_suit = self.df_start_player_cards.loc[(self.df_start_player_cards['suit']==suit) & (self.df_start_player_cards['trumps']==False)]
            total_importancy += self.sum_cards_importancy(df_suit)
            suit_lens[suit] = df_suit.shape[0]
            
        distance18 = player_importancy_scope[0][1] - player_importancy_scope[1][1]
        regularity18 = total_importancy - player_importancy_scope[1][1]  
        total_importancy = regularity18/distance18
        
        ######################
        # deviation. sigmoid()效果不好,x趋近0太多
        ######################
        len_list= []
        for suit in suit_lens.keys():
            if 'trump' == suit :
                continue  #skip trump
            len_list.append(suit_lens[suit])
        
        np_lengths = np.array(len_list)
        total_deviation0 = np.sqrt(np.sum(np.square((np_lengths - 2.88902))))
        try:
            total_deviation1 = total_deviation0/np.sum(np_lengths)/np.sum(np_lengths)  #non trump越长，值越小。 ^2突出了副牌长度的影响
        except ZeroDivisionError:
            print("YDL： PokerEnvironment::cards_status_evaluation(). didived by 0. total_deviation1 = +inf")
            total_deviation1 = float("inf")
        total_deviation2 = np.log10(total_deviation1)
        distance18 = player_deviation_scope[0] - player_deviation_scope[1]
        regularity18 = total_deviation2 - player_deviation_scope[1] #min
        total_deviation = regularity18/distance18
        
        #total_entropy = total_entropy*100/np.exp(total_cards)  #有问题!! TBS: regular (?,?)  突出副牌数量的影响。 (1,2,3)=0.523, (2,3,4)=0.023, (0,0,12)=0.8.27e-2, (0,1,11)=0.666e-2
        
        ######################
        # discard priority
        ######################
        #sort by 牌的数量. difficulty is the key types are not same (str, CardSuits)
        suit_lens_keys0 = list(suit_lens.keys())
        suit_lens_keys = []
        np_suit_lens_values = np.array(list(suit_lens.values()))
        sorted_length_index = np.argsort(-np_suit_lens_values)  #bigger --> smaller
        for i in range(4): #suit_lens_keys0 can't be np.array(). neither be slice indexed, such as suit_lens_keys0[sorted_length_index] is failed
            suit_lens_keys.append(suit_lens_keys0[sorted_length_index[i]]) #have to loop

        sorted_suit_lens = zip(suit_lens_keys, np_suit_lens_values[sorted_length_index])

        #doesn't work!!? keys are not in same type! sorted_suit_lens = sorted(suit_lens_items, key=lambda s:s[1], reverse = True)
        
        priorities = self.assign_discard_prority_to_card(sorted_suit_lens)
        
        return total_importancy, total_deviation, suit_lens, priorities #dict没有顺序，即使用sorted_suit_lens重建dict，也不行

    def sum_cards_importancy(self, df_suited_cards):  #only one suit in the df here
        total_importancy = 0
        master_round = self.full_poker.round  #name
        master_trump = self.full_poker.trump  #suit
        
        np_suited_cards_suit = df_suited_cards['suit'].values
        np_suited_cards_name = df_suited_cards['name'].values

        # 提取主2， 其他的无论是否trump, 都用regular计算. return后再加权
        trumps_master_index = np.where(np_suited_cards_name == master_round)  #arg， 所有的2
        trumps_master_len = len(trumps_master_index[0])  #unwrap tuple
        if trumps_master_len > 0 :
            #有2
            trumps_master_suits = np_suited_cards_suit[trumps_master_index]  # 2的suit
            total_importancy += master_trump_importancy['regular'] * trumps_master_len
            if master_trump in trumps_master_suits :
                total_importancy += master_trump_importancy['master'] - master_trump_importancy['regular']

        regular_cards_index = np.where(np_suited_cards_name != master_round)
        regular_cards = np_suited_cards_name[regular_cards_index]
        total_importancy += np.sum(card_importancy[regular_cards])

        return total_importancy

    def assign_discard_prority_to_card(self, suit_length): # priority=higher ==> dont discard
        player_cards_priority = {}
        player_suit_priority = {}
        priority_class = 200
        master_round = self.full_poker.round  #name
        master_trump = self.full_poker.trump  #suit
        
        for suit, length in suit_length:
            player_suit_priority = {}

            if 'trump' == suit:
                df_suit = self.df_start_player_cards.loc[(self.df_start_player_cards['trumps']==True)]
                df_oindex = df_suit[['oindex', 'suit', 'name']].values
                for oindex, suit2, name in df_oindex:
                    if name == master_round :
                        if suit2 == master_trump:
                            player_suit_priority[oindex] = master_trump_importancy['master'] + 300
                        else:
                            player_suit_priority[oindex] = master_trump_importancy['regular'] + 300
                    else:
                        player_suit_priority[oindex] = card_importancy[name] + 300
                #for oindex
            else: #non-trump
                df_suit = self.df_start_player_cards.loc[(self.df_start_player_cards['suit']==suit) & (self.df_start_player_cards['trumps']==False)]
                df_oindex = df_suit[['oindex', 'suit', 'name']].values
                for oindex, suit2, name in df_oindex:
                    player_suit_priority[oindex] = card_importancy[name] + priority_class
                
                priority_class -= 100
                
            player_cards_priority[suit] = player_suit_priority #dict不保证length排序
            #for length

        #print(player_cards_priority) #spyder显示出错，打印结果正确
        return player_cards_priority
    
    def create_AKQJ_pattern(self):
        master_round = self.full_poker.round  #name
        
        ################ pattern in AKQJ1098
        default_oindex_pattern0 = np.array([0, 12, 11, 10, 9, 8, 7])
        default_oindex_pattern = np.vstack((default_oindex_pattern0, 
                                            default_oindex_pattern0+13,
                                            default_oindex_pattern0+13*2,
                                            default_oindex_pattern0+13*3))
        
        
        #主2的oindex[4]
        master_trump_oindex = self.full_poker.df_cards.loc[self.full_poker.df_cards['name']==master_round, 'oindex'].values
        master_trump_oindex = np.sort(master_trump_oindex)  
        master_trump_oindex = master_trump_oindex[:,np.newaxis]  #1d -> 2d [4,1]
        
        #test K
        #master_trump_oindex = np.array([[0], [13], [26], [39]])  #A
        #master_trump_oindex = np.array([[12], [25], [38], [51]])  #K
        #master_trump_oindex = np.array([[1], [14], [27], [40]])  #2
        
        index_in_default_oindex_pattern0 = np.where(default_oindex_pattern==master_trump_oindex)  #where() output = tuple([axis=0], [axis=1])
        if 0 == len(index_in_default_oindex_pattern0[1]):
            #不在前7个大牌中
            oindex_pattern = default_oindex_pattern
        else:
            index_in_pattern = index_in_default_oindex_pattern0[1][0]
            oindex_pattern = np.delete(default_oindex_pattern, np.s_[index_pattern:index_pattern+1:1], axis=1)  #s_[起点：终点：间隔]

        return oindex_pattern
    
    def create_allowed_score(self, cards_status_importency, cards_status_deviation):
        max_score = 0
        coeff = cards_status_importency * cards_status_deviation    
        max_score = 100 * coeff + 10  #total 100 score in a full poker, 10 personal compensation
        return max_score
    
    def create_best_discards(self):
        priority_list = []
        np_priorities = np.array([])
#columns=['oindex', 'suit', 'name', 'score', 'trumps', 'who', 'played', 'discarded'])
        np_start_player_cards = self.df_start_player_cards[['oindex', 'suit', 'name', 'score']].values
        
        priority_dict = self.cards_status['priority']
        for key_suit in priority_dict.keys():
            priority_suit_dict = priority_dict[key_suit]
            pri0 = np.array(list(priority_suit_dict.items()))
            if len(pri0) > 0:
                priority_list.append(pri0)

        #[oindex, pri, skip]
        np_priorities = np.vstack(priority_list)
        np_priorities = np.array(sorted(np_priorities, key=lambda s: s[1]))
        skip_added = np.zeros([np_priorities.shape[0],1])
        np_priorities = np.hstack([np_priorities, skip_added])

        #test temp , [[4, 9, 12+13+13+13], [0, 12, 11, 13, 25, 26], [9+13]
        #np_priorities[:, 0] = np.array([4, 9, 48, 26, 25, 13, 11, 12, 0, 43, 49, 51, 34, 8, 16, 47, 14, 1])

        while True:
            #search pattern A,K,Q ....
            remove_list = self.search_AKQJ_pattern(np_priorities) #after here, lowest 6 is the discards choice
            #print("AKQJ pattern remove: ", remove_list)
            if len(remove_list) == 0:
                break;

            remove_list = self.search_score_in_discard(np_priorities, np_start_player_cards)
            #print("5,10,K pattern remove: ", remove_list)
            if len(remove_list) == 0:
                break;

        selected_discards0 = np_priorities[np_priorities[:,2]==False] # w/o skip
        selected_discards = selected_discards0[:6]  #lowest 6 w/o skip
        best_discards_oindex = selected_discards[:,0].astype(int).tolist()  #oindex
        best_discards_oindex.sort()
        #print(best_discards_oindex)

        return best_discards_oindex
        
    
    def auto_judge_return_estimation(self): #assume: win or loss
        reward = np.random.choice(2)  #win or loss
        #DiscardAgent_net6_MC::pre_learn()
        return reward

    def manual_return_estimation(self): #for episode. 没有具体牌，用策略原则打分
        static_reward = [-6, -5, -4, -3, -2, -1, 6]
        #static_reward = [-6, -2, -1, 0, 1, 2, 6]
        #static_reward = [0, 0, 0, 0, 0, 0, 1]
        #best_discards_oindex = [16, 18, 34, 41, 42, 43]  #oindex

        df_discarded_player_cards = self.df_start_player_cards.loc[(self.df_start_player_cards['discarded'] == True), 'oindex']
        returned_discarded_oindex = df_discarded_player_cards.values.tolist()
        
        #print(type(returned_discarded_oindex), returned_discarded_oindex)
        set1 = set(self.best_discards_oindex)
        set2 = set(returned_discarded_oindex)
        covered_by_best = len(set1 & set2)
        #print(ydl)
        
        reward = static_reward[covered_by_best]
        #refresh the discard status to self.full_poker
        return reward

    def manual_reward_estimation(self, discarded_card_oindex): # for step
        #[16, 18, 34, 41, 42, 43]  #oindex
        best_discards_oindex = self.best_discards_oindex
        '''
        discarded_card_oindex = discarded_card_oindex.tolist()
        set1 = set(best_discards_oindex)
        set2 = set(discarded_card_oindex)
        covered_by_best = len(set1 & set2)
        reward = covered_by_best
        '''
        
        if discarded_card_oindex in best_discards_oindex :
            reward = self.reward_times_10 #50
        else:
            reward = self.reward_times_1  #5
        return reward

    def search_AKQJ_pattern(self, np_priorities):
        length = 0
        remove_list = []
        remove_list2 = []
        oindex_pattern_flat = self.oindex_pattern.reshape(-1)

        #################### discarding priority
        selected_discards0 = np_priorities[np_priorities[:,2]==False] # w/o skip
        selected_discards = selected_discards0[:6]  #lowest 6 w/o skip
        selected_oindex0 = selected_discards[:,0]  #oindex
        selected_oindex = selected_oindex0[::-1]   #oindex
        
        #################### pattern mutaching
        biggest_card_oindex = self.oindex_pattern[:, 0].reshape(-1)
        for oindex_pattern in biggest_card_oindex:
            biggest_in_select = np.where(selected_oindex == oindex_pattern)[0]
            if len(biggest_in_select) == 0 :
                continue
            else:
                biggest_in_pattern = np.where(oindex_pattern_flat == oindex_pattern)[0]
                start_index = biggest_in_select
                length = 0
                while length <= 6:
                    try:
                        if selected_oindex[start_index+length] == oindex_pattern_flat[biggest_in_pattern+length]:
                            remove_list.append(selected_oindex[start_index+length][0]) #selected_oindex is array with only 1 item
                            length += 1
                            continue
                        else:
                            break
                    except IndexError: # selected_oindex[] may be out of boundary, as well as oindex_pattern_flat[]
                        print("YDL: PokerEnvironment：：search_AKQJ_pattern()： selected_oindex or oindex_pattern_flat out of bounds")
                        break
                
                remove_start = np.where(np_priorities[:,0]==selected_oindex[start_index])[0][0]  #tuple -> list -> scalar
                np_priorities[remove_start-length+1:remove_start+1, 2] = True  #原因：反顺序
        
        if 0 < length :  #ever happen
            remove_list2 = self.search_AKQJ_pattern(np_priorities)
        
        return remove_list + remove_list2

    def search_score_in_discard(self, np_priorities, np_start_player_cards):
        remove_list = []
        remove_list2 = []
        
        #################### discarding priority
        selected_discards0 = np_priorities[np_priorities[:,2]==False] # w/o skip
        selected_discards = selected_discards0[:6]  #lowest 6 w/o skip
        selected_oindex0 = selected_discards[:,0][::-1]  #oindex, 0=lowest pri, 5=highest pri
        selected_oindex = selected_oindex0[:,np.newaxis] #1d->2d
        
#['oindex', 'suit', 'name', 'score']
        selected_cards0 = np.where(np_start_player_cards[:,0]==selected_oindex) #每个selected_oindex item会生成一行[18], 共6行
        selected_cards1 = np_start_player_cards[selected_cards0[1]]  #priority sorted
        selected_cards = selected_cards1[selected_cards1[:,3]>0]  #score cards
        discarded_score = np.sum(selected_cards[:,3])
        
        if discarded_score > self.allowed_discarded_score:  #15
            for oindex_score, _, _, score in selected_cards:
                remove_list.append(oindex_score)
                discarded_score -= score
                remove_index = np.where(np_priorities[:,0]==oindex_score)
                np_priorities[remove_index[0],2] =True  #should be only one item in remove_index
                if discarded_score <= self.allowed_discarded_score:
                    break;
                #for score
            #reduce

        if 0 < len(remove_list) :  #ever happen
            remove_list2 = self.search_score_in_discard(np_priorities, np_start_player_cards)
        
        return remove_list + remove_list2

    def priority_ordered_best_discards(self):
        pri_values = list(self.cards_status['priority'].values())  #dict{} by suit and oindex

        #simplified index by 0/1/2/3
        player_priority_list_dict = {**pri_values[0], **pri_values[1], **pri_values[2], **pri_values[3]}

        best_discards_priority = np.array([player_priority_list_dict[oindex] for oindex in self.best_discards_oindex])
        best_discards_index = np.argsort(best_discards_priority)
        best_discards_sorted = np.array(self.best_discards_oindex)[best_discards_index]
        
        return best_discards_sorted   #smaller -> bigger

    def get_best_discards_priority_sorted(self):
        oindex_pri_sorted = self.best_discards_priority_sorted.copy()
        return oindex_pri_sorted
            
class PokerEnvironment_6_1(PokerEnvironment): #6 steps
    def __init__(self, reward_times=5, input_format=[0,0,0.5,1]):
        super().__init__()
        self.reward_times_10 = 10*reward_times
        self.reward_times_1  = 1 *reward_times
        #input_format: [not existing, discarded/played, in-hand, trump]
        self.net_input_format = input_format # not in-hand MUST BE: <=0
        
    def discard_step(self, discarded_card_oindex):  # np optm demo here
#columns=['oindex', 'suit', 'name', 'score', 'trumps', 'who', 'played', 'discarded'])
        np_start_player_cards = self.df_start_player_cards.values
        np_player_cards = self.df_start_player_cards.values
        
        for i in range(np_start_player_cards.shape[0]):
            if np_start_player_cards[i,0] == discarded_card_oindex:
                np_start_player_cards[i,7] = True
                break   #this it the only action

        to_del = []
        for i in range(np_player_cards.shape[0]):
            if np_start_player_cards[i,7] == True:
                to_del.append(i)
        np_player_cards = np.delete(np_player_cards, to_del, axis=0)


        self.df_start_player_cards = pd.DataFrame(np_start_player_cards, columns=['oindex', 'suit', 'name', 'score', 'trumps', 'who', 'played', 'discarded'])
        df_player_cards = pd.DataFrame(np_player_cards, columns=['oindex', 'suit', 'name', 'score', 'trumps', 'who', 'played', 'discarded'])

        #need not. done in reset(). df_player_cards.sort_values(by=["suit", "name"] , ascending=[True, True], inplace=True)
        #print(df_player_cards)
        self.state = self.full_poker.generate_pi_str_key(df_player_cards)

        df_discarded_player_cards = self.df_start_player_cards[self.df_start_player_cards['discarded'] == True]
        if df_discarded_player_cards.shape[0] < 6 :
            reward = self.manual_reward_estimation(discarded_card_oindex) #0
            done = False
        else:
            reward = self.manual_reward_estimation(discarded_card_oindex) #self.auto_judge_return_estimation() #self.manual_return_estimation()
            done = True

        return self.state, reward, done, df_player_cards, df_discarded_player_cards

    def discard_step2(self, discarded_card_oindex):
        #apply state2 (54) onehot
        _, reward, done, df_player_cards, df_discarded = self.discard_step(discarded_card_oindex)

        #no existing
        full_cards_onehot_like = np.zeros(54) #self.net_input_format[0]

        #input_format: [not existing, discarded/played, in-hand, trump]
        #[0, -1, 1, 2]
        #in-hand
        oindex = df_player_cards['oindex'].values.astype('int32')
        full_cards_onehot_like[oindex] = self.net_input_format[2]

        #trump
        df_trumps = df_player_cards.loc[df_player_cards['trumps']==True]
        oindex = df_trumps['oindex'].values.astype('int32')
        full_cards_onehot_like[oindex] = self.net_input_format[3]

        #discarded
        oindex = df_discarded['oindex'].values.astype('int32')
        full_cards_onehot_like[oindex] = self.net_input_format[1]

        self.state2 = full_cards_onehot_like

        return self.state2, reward, done, df_player_cards  #exclude discarded
    
class PokerEnvironment_1_6(PokerEnvironment):  #6 cards in one go
    def __init__(self):
        super().__init__()

    def discard_onego_step(self, discarded_card_oindexs):
        # 有问题！！！ 目前只能判定一个牌局，不能同时判定多个牌局形式, 
        # 扩展性差，理论上需要C(54,18)个状态
        self.df_start_player_cards.loc[(self.df_start_player_cards['oindex'].isin(discarded_card_oindexs)), 'discarded'] = True
        df_player_cards = self.df_start_player_cards[self.df_start_player_cards['discarded'] == False]

        self.state2 = np.array([])   #next state, but need not in 6 in onego mode
        reward = self.manual_return_estimation()
        done = True

        return self.state2, reward, done, df_player_cards  #exclude discarded

    