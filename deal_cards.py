import time
import numpy as np
import pandas as pd
import random as rd
from enum import Enum, IntEnum
from random import shuffle as rshuffle
import collections as cllt

ticks = time.time()
seed0 = int(ticks*100000) % (2**32-1)
np.random.seed(seed0)
rd.seed(seed0)

np.random.seed()  #empty input param = seeds from current time
rd.seed()  #empty input param = seeds from current time

np.random.seed(13)  #keep same random sequence
rd.seed(13)  #keep same random sequence


class CardSuits(IntEnum):
    NONE       = 0
    SPADES     = 1
    HEARTS     = 2
    CLUBS      = 3
    DIAMONDS   = 4
    BLACKJOKER = 5  #must be this place
    REDJOKER   = 6  #must be this place. otherwise, the ‘T' print is wrong

cardsuits_value_to_name = {0: CardSuits.NONE,
                           1: CardSuits.SPADES,
                           2: CardSuits.HEARTS,
                           3: CardSuits.CLUBS,
                           4: CardSuits.DIAMONDS,
                           5: CardSuits.BLACKJOKER,
                           6: CardSuits.REDJOKER }
class Players(IntEnum):
    NONE   = 0
    NORTH  = 1
    WEST   = 2
    EAST   = 3
    SOUTH  = 4

players_value_to_name = {0: Players.NONE,
                         1: Players.NORTH,
                         2: Players.WEST,
                         3: Players.EAST,
                         4: Players.SOUTH }

playing_order = [Players.SOUTH, Players.WEST, Players.NORTH, Players.EAST]

card_print_name = {1: 'A',
                   2: '2',
                   3: '3',
                   4: '4',
                   5: '5',
                   6: '6',
                   7: '7',
                   8: '8',
                   9: '9',
                   10: '10',
                   11: 'J',
                   12: 'Q',
                   13: 'K',
                   14: 'BJ',
                   15: 'RJ' }
brief_suit = {CardSuits.SPADES: 'S',
              CardSuits.HEARTS: 'H',
              CardSuits.CLUBS:  'C',
              CardSuits.DIAMONDS: 'D',
              CardSuits.BLACKJOKER: 'T',
              CardSuits.REDJOKER: 'T'}

perfix_space = {Players.NORTH: ' '*22,
                Players.WEST : ' ',
                Players.EAST : ' '*40,
                Players.SOUTH: ' '*22 }

class FullPokers :
    def sync_df_to_list(self):
        self.cards = self.df_cards.values.tolist()

    def sync_list_to_df(self):
        self.df_cards = pd.DataFrame(self.cards, columns=['oindex', 'suit', 'name', 'score', 'trumps', 'who', 'played', 'discarded'])

    def print_all_cards(self):
        #print("list cards ", self.cards)
        print("df cards ", self.df_cards)

    def __init__(self):
        self.cards = []
        trumps = False
        who = Players.NONE
        played = False
        discarded = False
        card_index = 0
        for suit in CardSuits:
            if CardSuits.BLACKJOKER == suit:
                name = 14
                score = 0
                trumps = True
                self.cards.append([card_index, suit, name, score, trumps, who, played, discarded])  #?=state?
                card_index += 1
                trumps = False
            elif CardSuits.REDJOKER == suit:
                name = 15
                score = 0
                trumps = True
                self.cards.append([card_index, suit, name, score, trumps, who, played, discarded])  #?=state?
                card_index += 1
                trumps = False
            elif CardSuits.NONE == suit:
                continue
            
            else:
                for name in np.arange(1,14,1):
                    if 5 == name:
                        score = 5
                    elif 10 == name or 13 == name:
                        score = 10
                    else:
                        score = 0
                    self.cards.append([card_index, suit, name, score, trumps, who, played, discarded])  #?=state?
                    card_index += 1
                
        self.sync_list_to_df()
        #self.print_all_cards()
        self.trump = CardSuits.NONE   #主牌花色
        self.round = 0   #打2，3，4 ...
        self.banker = Players.NONE
        self.start_player = Players.SOUTH
        self.scores = [10, 5]  #0: banker; 1: non-banker
        self.saver = pd.DataFrame()


    def cards_shuffle(self):
        l_cards = self.df_cards.values.tolist()
        n_cards = self.df_cards.values
        np.random.shuffle(n_cards)

        rshuffle(self.cards)
        self.sync_list_to_df()
        #self.print_all_cards()
    
    def cards_assign_to_player(self):
        self.df_cards.loc[0:11,'who']  = Players.WEST
        self.df_cards.loc[12:23,'who'] = Players.NORTH
        self.df_cards.loc[24:35,'who'] = Players.EAST
        self.df_cards.loc[36:,'who']   = Players.SOUTH  #banker=south, final version should be exclude last 6
        self.sync_df_to_list()
        #self.print_all_cards()

    def cards_assign_to_player_2(self): #ignore last 6 cards
        self.df_cards.loc[0:11,'who']  = Players.WEST
        self.df_cards.loc[12:23,'who'] = Players.NORTH
        self.df_cards.loc[24:35,'who'] = Players.EAST
        self.df_cards.loc[36:47,'who']   = Players.SOUTH  #banker=south, final version should be exclude last 6

    def statistic_cards_length(self):
        players_card_len = []
        for player in Players:
            if player == Players.NONE:
                continue
            df_player_cards = self.df_cards.loc[(self.df_cards['who']==player)]
            suit_cards_len = []
            
            for suit in CardSuits:
               if suit in [CardSuits.NONE, CardSuits.BLACKJOKER, CardSuits.REDJOKER] :
                   continue
        
               df_player_suit_cards = df_player_cards.loc[df_player_cards['suit']==suit] 
               suit_cards_len.append(df_player_suit_cards.shape[0])

            players_card_len.append(suit_cards_len)
        return players_card_len
                
    def set_trumps_banker(self, suit_trump, name_trump, banker):
        self.df_cards.loc[self.df_cards['suit']==suit_trump, 'trumps'] = True
        self.df_cards.loc[self.df_cards['name']==name_trump, 'trumps'] = True
        self.sync_df_to_list()
        #self.print_all_cards()
        
        self.trump = suit_trump
        self.round = name_trump
        self.banker = banker
        
    def render(self):
        # https://www.jianshu.com/p/7d7e7e160372
        '''
        print("显示方式：")
        print("\033[0mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[1mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[4mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[5mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[7mSuixinBlog: https://suixinblog.cn\033[0m")
        print("字体色：")
        print("\033[30mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[31mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[32mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[4;33mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[34mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[1;35mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[4;36mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37mSuixinBlog: https://suixinblog.cn\033[0m")
        print("背景色：")
        print("\033[1;37;40m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;41m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;42m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;43m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;44m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;45m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;46m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[1;30;47m\tSuixinBlog: https://suixinblog.cn\033[0m")
        '''
        
        str_W = {'S': ' ', 'H': ' ', 'C': ' ', 'D': ' ', 'T': ' '}
        str_N = {'S': ' ', 'H': ' ', 'C': ' ', 'D': ' ', 'T': ' '}
        str_E = {'S': ' ', 'H': ' ', 'C': ' ', 'D': ' ', 'T': ' '}
        str_S = {'S': ' ', 'H': ' ', 'C': ' ', 'D': ' ', 'T': ' '}

        str_mapping = {Players.WEST: str_W,
                       Players.NORTH: str_N,
                       Players.EAST: str_E,
                       Players.SOUTH: str_S}
        
        for player in Players :
            if Players.NONE == player :
                continue
            #df_ydl1 = self.df_cards[(self.df_cards['suit']==suit) & (self.df_cards['name']==name)]
            group_player = self.df_cards[(self.df_cards['who'] == player) & (self.df_cards['discarded'] == False)]
            group_player = group_player.sort_values(by="name" , ascending=True)
        
            for card in group_player.itertuples():
                self.build_player_string_per_card(str_mapping[card.who], card)
                
            
        for player in Players :
            print(end='\n')

            if Players.NONE == player or Players.EAST == player:
                continue

            for suit in CardSuits:
                if CardSuits.NONE == suit:
                    continue
                
                print_strs = []
                print_str = str_mapping[player][brief_suit[suit]]
                if Players.WEST == player :   #combine WEST and EAST in one row
                    print_str2 = str_mapping[Players.EAST][brief_suit[suit]]
                    len_str  = len(print_str)
                    c33 = print_str.count('\033')
                    len_str -= c33//2*(8+4)
                    print_str2_sapce = ' '*(len(perfix_space[Players.EAST]) - len_str - len(perfix_space[Players.WEST]) - 3)

                    print_strs.append([print_str, perfix_space[Players.WEST]])
                    print_strs.append([print_str2, print_str2_sapce])
                else:
                    print_strs.append([print_str, perfix_space[player]])

                if self.trump == suit :
                    T_perfix = 'T'
                else :
                    T_perfix = ' '
                for print_str, spaces in print_strs :
                    print(spaces, T_perfix, brief_suit[suit], ':', print_str, end='') #str_mapping[player][brief_suit[suit]])
                print(end='\n')
                
                #ensure the BJ and RJ are in last position of 'players'
                if CardSuits.BLACKJOKER == suit or CardSuits.REDJOKER == suit :
                    break
            continue
        
        print('\n', self.start_player, self.scores)
        return
            
    def build_player_string_per_card(self, str_dict, card, to_print=True):
        #print("\033[37;45m\tSuixinBlog: https://suixinblog.cn\033[0m")
        played_patten_perfix = ''
        played_patten_postfix = ''
        if True == to_print :
            if  True == card.played :
                played_patten_perfix = '\033[37;45m'  #len = 8
                played_patten_postfix = '\033[0m'     #len = 4

        if card.name == self.round :
            added_str= played_patten_perfix + brief_suit[card.suit] + card_print_name[card.name] + played_patten_postfix + ' '
            str_dict['T'] += added_str
            return
        
        added_str = played_patten_perfix + card_print_name[card.name] + played_patten_postfix + ' '
        str_dict[brief_suit[card.suit]] += added_str
        ''' replaced by above
        if card.suit == CardSuits.SPADES :
            str_dict['S'] += added_str
        elif card.suit == CardSuits.HEARTS :
            str_dict['H'] += added_str
        elif card.suit == CardSuits.CLUBS :
            str_dict['C'] += added_str
        elif card.suit == CardSuits.DIAMONDS :
            str_dict['D'] += added_str
        elif card.suit == CardSuits.BLACKJOKER or card.suit == CardSuits.REDJOKER :
            str_dict['T'] += added_str
        '''
        
        return 0
        
    def generate_pi_str_key(self, df_player_cards):
        str_keys = {'S': ' ', 'H': ' ', 'C': ' ', 'D': ' ', 'T': ' '}
        for card in df_player_cards.itertuples():
            self.build_player_string_per_card(str_keys, card, to_print=False)
        #print(str_keys)
        
        str_state = ''
        for suit in str_keys.keys():
            if brief_suit[self.trump] == suit :
                leading = 'T' + suit
            else:
                leading = suit
                
            str_keys_splited = str_keys[suit].split(' ')
            for name in str_keys_splited:
                if ' ' == name or '' == name:
                    continue
                else:
                    str_state += leading + name
                    
        #print(str_state)
        return str_state
        
    def discard_cards(self, player, dropping_cards): #dropping_cards[], list of index
        for suit, name in dropping_cards:
            self.df_cards.loc[(self.df_cards['suit']==suit) & (self.df_cards['name']==name),'discarded'] = True
        self.sync_df_to_list()
        self.print_all_cards()
        
    def decide_discard_cards(self, player):
        #temp, simplified example
        dropping_cards = self.df_cards[self.df_cards['who']==player].values
        dropping_cards1 = dropping_cards[-1]
        dropping_cards2 = dropping_cards[-6:-1]
        dropping_cards = np.vstack((dropping_cards2, dropping_cards1))
        self.print_all_cards()
        
        df_dropping_cards = pd.DataFrame(dropping_cards, columns=['index', 'suit', 'name', 'score', 'trumps', 'who', 'played', 'discarded'])        
        df_ydl1 = df_dropping_cards[['suit', 'name']]
        dropping_cards = df_ydl1.values
        #print(dropping_cards)
        #print(df_dropping_cards)
        
        return dropping_cards
        
    def play_card_decision(self, player, played_cards):
        # temp, example
        player_cards = self.df_cards[(self.df_cards['who']==player) & (self.df_cards['played']==False)]
        selected_card = player_cards.iloc[0]
        suit = selected_card['suit']
        name = selected_card['name']
        self.df_cards.loc[(self.df_cards['who']==player) & (self.df_cards['suit']==suit) & (self.df_cards['name']==name),'played'] = True #?stupid, how to reference(view) to it?
        
        return selected_card[['suit', 'name']]
    
    def play_one_round(self, start_player):
        start_player_index = playing_order.index(start_player)
        rotated_players = playing_order[start_player_index:] + playing_order[:start_player_index]
        print(rotated_players)
        played_cards = []
        for player in rotated_players:
            played = self.play_card_decision(player, played_cards)
            played_cards.append([played])
            #print(played)
            #print(played_cards)
        
        self.sync_df_to_list()
        return

    #have to convert ENUM to int
    def save_cards_status(self):  # only support singletone deal
        if self.saver.empty == True :
            self.saver = self.df_cards.copy(deep=True)
            write_df = self.df_cards.copy(deep=True)
            
            ''' #bypass if IntEnum
            w_suit = write_df['suit']
            w_player = write_df['who']
            for i in range(54):
                w_suit[i] = w_suit[i].value
                w_player[i] = w_player[i].value
            '''
            #write_df.to_csv('round0.csv', index=False) #, dtype={'suit':int, 'who':int})
        
    #have to recover int to ENUM
    def load_cards_status(self): # only support singletone deal
        if self.saver.empty == False :
            self.df_cards = self.saver.copy(deep=True) #ignore sync to list
        else:
            
            read_df = pd.read_csv('round0.csv') 
            ''' #bypass if IntEnum
            r_suit = read_df['suit']
            r_player = read_df['who']
    
            for i in range(54):
                r_suit[i] = cardsuits_value_to_name[r_suit[i]]
                r_player[i] = players_value_to_name[r_player[i]]
    
            '''
            #self.df_cards = read_df.copy(deep=True) 
            
            self.sync_df_to_list()
        return self.df_cards

    def reset_df_cards(self):
        self.df_cards = self.df_cards.drop(index=self.df_cards.index)
        self.sync_df_to_list()
        
    def get_player_cards_for_discard(self, player):
        group_player = self.df_cards[(self.df_cards['who'] == player)]
        #group_player = group_player[['oindex', 'suit', 'name', 'score', 'trumps', 'discarded']] #master trump card need to be added later
        return group_player
        
def test_example():
    full_poker = FullPokers()
    full_poker.cards_shuffle()
    full_poker.cards_assign_to_player()
    full_poker.set_trumps_banker(CardSuits.SPADES, 2, Players.SOUTH)
    
    dropping_cards_example = full_poker.decide_discard_cards(Players.SOUTH)
    full_poker.discard_cards(Players.SOUTH, dropping_cards_example)
    full_poker.play_one_round(Players.NORTH)
    full_poker.play_one_round(Players.SOUTH)
    full_poker.play_one_round(Players.EAST)
    full_poker.play_one_round(Players.EAST)
    full_poker.play_one_round(Players.WEST)
    full_poker.play_one_round(Players.WEST)
    full_poker.render()
    
    
    full_poker.save_cards_status()
    full_poker.reset_df_cards()
    reloaded = full_poker.load_cards_status()
    full_poker.render()

def test_example_2(): #统计每个player手上，各个花色牌的数量的分布， 均值=3，标准差=1.35
    full_poker = FullPokers()
    ydl_c = {}
    for i in range(10000):
        full_poker.cards_shuffle()
        full_poker.cards_assign_to_player_2() #ignore last 6 cards
        ydl = full_poker.statistic_cards_length()
        np_ydl = np.array(ydl).reshape(-1)
        ydl2 = cllt.Counter(np_ydl)
        for length in ydl2.keys():
            ydl3 = ydl2[length]
            if length not in ydl_c.keys():
                ydl_c[length] = ydl3
            else:
                ydl_c[length] += ydl3
        if i % 100 == 0 :
            print(i, ydl_c)
    print(ydl_c)
    
def deal_first_round(full_poker, render=False):
    full_poker.cards_shuffle()
    full_poker.cards_assign_to_player()
    full_poker.set_trumps_banker(CardSuits.SPADES, 2, Players.SOUTH)
    if render == True:
        full_poker.render()
    


#test_example()
#test_example_2()

'''
np.random.shuffle(arr)
np.random.permutation([1, 4, 9, 12, 15])
'''
