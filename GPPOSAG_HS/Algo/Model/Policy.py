import torch
import torch.nn as nn

from .head import ActionTypeHead, TargetEntityHead, TargetCardHead, TargetPositionHead
from typing import List, Dict, Optional
from torch import Tensor

def update_action_mask(action_list, action_type, mask_action_type):
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
    action_type = int(action_type)
    mask = {}
    mask_head = {}
    mask['action_type'] = mask_action_type
    mask_head['action_type'] = torch.tensor([True])
    
    mask['target_card'] = torch.tensor([False,False,False
                                    ]).reshape(1,3)
    mask_head['target_card'] = torch.tensor([False])
    
    mask['target_entity'] = torch.tensor([False,False,False,False,False,
                                        False,False,False,False,False,
                                        False,False,False,False,False,
                                        False,False
                                        ]).reshape(1,17)
    mask_head['target_entity'] = torch.tensor([False])

    mask['target_position'] = torch.tensor([False,False,False,False,False,
                                        False,False
                                        ]).reshape(1,7)
    mask_head['target_position'] = torch.tensor([False])
    
    if action_type == 0:
        pass

    elif action_type < 12 and action_type > 0:
        if 'PlayCardTask' not in action_list.keys():
            return mask, mask_head
        dict_i = action_list['PlayCardTask'][action_type-1]

        mask['target_entity'][:,list(dict_i.keys())] = True

        for item in dict_i.values():
            if isinstance(item, dict):
                mask['target_position'][:,list(item.keys())] = True
    
    elif action_type>=12 and action_type <=18:
        if 'MinionAttackTask' not in action_list.keys():
            return mask, mask_head
        dict_i = action_list['MinionAttackTask'][action_type-12]
        mask['target_entity'][:,list(dict_i.keys())] = True
    
    elif action_type == 19:
        mask['target_entity'][:,list(action_list['HeroAttackTask'].keys())] = True
    
    elif action_type == 20:
        mask['target_card'][:] = True
    
    if mask['target_entity'].sum()>1:
        mask_head['target_entity'][0] = True
    if mask['target_position'].sum()>1:
        mask_head['target_position'][0] = True
    return mask, mask_head

class Policy(nn.Module):
    def __init__(self, cfg):
        super(Policy, self).__init__()
        self.whole_cfg = cfg
        self.action_type_head = ActionTypeHead(self.whole_cfg)
        self.target_entity_head = TargetEntityHead(self.whole_cfg)
        self.target_card_head = TargetCardHead(self.whole_cfg)
        self.location_head = TargetPositionHead(self.whole_cfg)

    def forward(self,entities_feat, hand_card_feat, minions_feat, heros_feat, secret_value, mask):
        action = torch.jit.annotate(Dict[str, Tensor], {})
        logit = torch.jit.annotate(Dict[str, Tensor], {})

        # action type
        logit['action_type'], action['action_type'], embeddings = self.action_type_head(entities_feat, heros_feat.reshape(heros_feat.shape[0],-1), mask['action_type']) # entities_feat:(batch_size,320) heros_feat:(batch_size,2,320)

        logit['target_entity'], action['target_entity'], embeddings = self.target_entity_head(embeddings, hand_card_feat, mask['target_entity'])

        logit['target_card'], action['target_card'] = self.target_card_head(embeddings, hand_card_feat, mask['target_card'])

        logit['target_position'], action['target_position'] = self.location_head(embeddings, minions_feat, mask['target_position'])
        
        for k in logit.keys():
            logit[k] = logit[k].tanh()

        return action, logit

    def select_forward(self, entities_feat, hand_card_feat, minions_feat, heros_feat, secret_value, available_actions, mask):
        action = torch.jit.annotate(Dict[str, Tensor], {})
        logit = torch.jit.annotate(Dict[str, Tensor], {})

        # action type
        logit['action_type'], action['action_type'], embeddings = self.action_type_head(entities_feat, heros_feat.reshape(heros_feat.shape[0],-1), mask['action_type']) # entities_feat:(batch_size,320) heros_feat:(batch_size,2,320)

        mask, mask_head = update_action_mask(available_actions, action['action_type'].cpu().item(), mask['action_type'])

        logit['target_entity'], action['target_entity'], embeddings = self.target_entity_head(embeddings, available_actions, mask['target_entity'])

        logit['target_card'], action['target_card'] = self.target_card_head(embeddings, available_actions, mask['target_card'])

        logit['target_position'], action['target_position'] = self.location_head(embeddings, available_actions, mask['target_position'])
        
        for k in logit.keys():
            logit[k] = logit[k].tanh()

        return action, logit, mask_head, mask

    def train_forward(self, entities_feat, hand_card_feat, minions_feat, heros_feat, secret_value, mask, action_info):
        action = torch.jit.annotate(Dict[str, Tensor], {})
        logit = torch.jit.annotate(Dict[str, Tensor], {})

        if heros_feat.dim() == 4:
            heros_feat = heros_feat.reshape(heros_feat.shape[0],heros_feat.shape[1],-1)
        elif heros_feat.dim() == 3:
            heros_feat = heros_feat.reshape(heros_feat.shape[0],-1)
        # action type
        logit['action_type'], action['action_type'], embeddings = self.action_type_head(entities_feat, heros_feat, mask['action_type'], action_info['action_type'])

        logit['target_entity'], action['target_entity'], embeddings = self.target_entity_head(embeddings, hand_card_feat, mask['target_entity'], action_info['target_entity'])

        logit['target_card'], action['target_card'] = self.target_card_head(embeddings, hand_card_feat, mask['target_card'], action_info['target_card'])

        logit['target_position'], action['target_position'] = self.location_head(embeddings, minions_feat, mask['target_position'], action_info['target_position'])
        
        for k in logit.keys():
            logit[k] = logit[k].tanh()
            
        return action, logit
