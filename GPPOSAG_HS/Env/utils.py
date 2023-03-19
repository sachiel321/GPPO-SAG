import numpy as np
import clr
import random
import os
import System
from System.Collections import *
clr.AddReference(
    os.getcwd() + "/Env/DllSimulator/SabberStoneCore.dll")
clr.AddReference(
    os.getcwd() + "/Env/DllSimulator/SabberStoneBasicAI.dll")
import SabberStoneCore
import SabberStoneBasicAI
from SabberStoneBasicAI.Score import *
from SabberStoneBasicAI.Nodes import *
from SabberStoneBasicAI.Meta import *
from SabberStoneCore.Enums import *
from SabberStoneCore.Config import *
from SabberStoneCore.Model import *
from SabberStoneCore.Tasks.PlayerTasks import ChooseTask

def check_race(card_entity):
    """
        Get the minion race
    """
    race_list = [0, 18, 15, 20, 14, 21, 23, 17, 24]
    race_id = None
    for (i, id) in enumerate(race_list):
        if card_entity.IsRace(id):
            race_id = i
            break
    return race_id

def check_type(card_type):
    """
        Get the card type
    """
    type_list = [4, 5, 7]
    return type_list.index(card_type)
    # [0: minion, 1: spell, 2: weapon]

def DeckList(deck_list, random_cards=False):
    deck = System.Collections.Generic.List[Card]()
    for card_name in deck_list:
        card = Cards.FromName(card_name)
        if random_cards == True:
            # To decouple the description and corresponding scalar vector, randomize the Attack, Health, Cost of a card.
            if card.Type == 4:
                card.ATK = random.randint(0, 12)
                card.Health = random.randint(1, 12)
                card.Cost = random.randint(0, 10)
            elif card.Type == 5:
                card.Cost = random.randint(0, 10)
            elif card.Type == 7:
                card.Cost = random.randint(0, 10)
                card.ATK = random.randint(0, 10)
        if card is None:
            raise Exception("Card Is None Exception {}".format(card_name))
        else:
            deck.Add(Cards.FromName(card_name))

    return deck

# def get_id(name):
#     return card_name.index(name)
        
def modify_cards():
    # Some cards has been updated by the official game, while SabberStone has not yet implemented
    soulfire = Cards.FromName("Soulfire")
    soulfire.Cost = 0

    knifeujuggler = Cards.FromName("Knife Juggler")
    knifeujuggler.ATK = 3

    leeroy = Cards.FromName("Leeroy Jenkins")
    leeroy.Cost = 4

    hunter_mark = Cards.FromName("Hunter's Mark")
    hunter_mark.Cost = 0

    flare = Cards.FromName("Flare")
    flare.Cost = 1

    starving_buzzard = Cards.FromName("Starving Buzzard")
    starving_buzzard.Cost = 2
    starving_buzzard.ATK = 2
    starving_buzzard.Health = 1

    mana_wyrm = Cards.FromName("Mana Wyrm")
    mana_wyrm.Cost = 1

    equality = Cards.FromName("Equality")
    equality.Cost = 2

    return

def validate_card(card_name):
    card = Cards.FromName(card_name)
    if card is not None and card.Implemented:
        return True
    else:
        return False

import torch
def get_action_mask(action_list):
    '''
    Action type:
        0       EndTurn
        1-11    select card in hand
        12-18   select minion in desk
        19      hero attack task
        20   discovery select

    TargetCardHead:
        0~2:    card selection in card discovery

    TargetEntityHead:
        0:      None target
        1~7:    our minions
        8~14 :  enemy minions
        15:     our hero
        16:     enemy hero
    
    TargetPositionHead:
        0~6:    minion position


    obs:
        hand_card_namrs:    list, 11
        minion_names:       list, 14
        weapon_names:       list, 2
        secret_names:       list, 2
        hand_card_scalar:   tensor
        minion_scalar:      tensor
        hero_scalar:        tensor
    '''
    #TODO:
    mask = {}
    mask['action_type'] = torch.tensor([True,False,False,False,False,
                                    False,False,False,False,False,
                                    False,False,False,False,False,
                                    False,False,False,False,False,
                                    False
                                    ]).reshape(1,21)
    
    mask['target_card'] = torch.tensor([False,False,False
                                    ]).reshape(1,3)
    
    mask['target_entity'] = torch.tensor([False,False,False,False,False,
                                        False,False,False,False,False,
                                        False,False,False,False,False,
                                        False,False
                                        ]).reshape(1,17)

    mask['target_position'] = torch.tensor([False,False,False,False,False,
                                        False,False
                                        ]).reshape(1,7)
    
    if 'PlayCardTask' in action_list:
        temp_idx = [j+1 for j in list(action_list['PlayCardTask'].keys())]
        mask['action_type'][:,temp_idx] = True

        for dict_i in action_list['PlayCardTask'].values():
            mask['target_entity'][:,list(dict_i.keys())] = True

            for temp in dict_i.values():
                if isinstance(temp, dict):
                    mask['target_position'][:,list(temp.keys())] = True
    
    if 'MinionAttackTask' in action_list:
        temp_idx = [j+12 for j in list(action_list['MinionAttackTask'].keys())]
        mask['action_type'][:,temp_idx] = True

        for dict_i in action_list['MinionAttackTask'].values():
            mask['target_entity'][:,list(dict_i.keys())] = True
    
    if 'HeroAttackTask' in action_list:
        mask['action_type'][:,19] = True
        mask['target_entity'][:,list(action_list['HeroAttackTask'].keys())] = True
    
    if 'Discovery' in action_list:
        mask['target_card'][:] = True
    
    return mask

def get_action_dict(options, game):


    def target2id(option, game):

        '''
            0: no target
            1 - 7: friendly minions
            8 - 14: opponent minions
            15: my hero
            16: opponent hero
        '''

        if option.HasTarget:
            if option.Target.Zone is not None:
                if option.Target.Zone.Controller.Name == game.CurrentPlayer.Name:
                    target_id = option.Target.ZonePosition + 1
                elif option.Target.Zone.Controller.Name == game.CurrentOpponent.Name:
                    target_id = option.Target.ZonePosition + 8
            elif option.Target == game.CurrentPlayer.Hero:
                target_id = 15
            elif option.Target == game.CurrentOpponent.Hero:
                target_id = 16
        else:
            target_id = 0
        return target_id


    action_dict = {}
    for option_id, option in enumerate(options):
        target_id = target2id(option, game)
        option_name = type(option).__name__
        if option_name not in action_dict:
            action_dict[option_name] = {}
        if option_name == 'EndTurnTask':
            action_dict[option_name] = option_id
        elif option_name == 'HeroPowerTask':
            action_dict[option_name][target_id] = option_id
        elif option_name == 'PlayCardTask':
            # currently no ChooseOne keyword
            if option.Source.ZonePosition not in action_dict[option_name]:
                action_dict[option_name][option.Source.ZonePosition] = {}
            if target_id not in action_dict[option_name][option.Source.ZonePosition]:
                action_dict[option_name][option.Source.ZonePosition][target_id] = {}
            if option.ZonePosition != -1:  # play a minion to position: 'option.ZonePosition'
                # action_dict['PlayCardTask']['HandCardPositionID']['TargetID']['MinionZonePosition']
                action_dict[option_name][option.Source.ZonePosition][target_id][option.ZonePosition] = option_id
            else:
                action_dict[option_name][option.Source.ZonePosition][target_id] = option_id
        elif option_name == 'MinionAttackTask':
            if option.Source.ZonePosition not in action_dict[option_name]:
                action_dict[option_name][option.Source.ZonePosition] = {}
            action_dict[option_name][option.Source.ZonePosition][target_id] = option_id
            
        elif option_name == 'HeroAttackTask':
            action_dict[option_name][target_id] = option_id
    if 'HeroPowerTask' in action_dict:
        if 'PlayCardTask' not in action_dict:
            action_dict['PlayCardTask'] = {}
        action_dict['PlayCardTask'][game.CurrentPlayer.HandZone.Count] = action_dict['HeroPowerTask']
        del action_dict['HeroPowerTask']
    
    return action_dict, get_action_mask(action_dict)