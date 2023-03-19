'''
SelectedCardHead:
    0~9:    10 cards in hand
    10:     1 hero power
    11:     1 hero weapon
    12~18:  7 minions on desk

TargetCardHead:
    0~2:    card selection in card discovery

TargetPositionHead:
    0:      None target
    1~7:    our minions
    8~14 :  enemy minions
    15:     our hero
    16:     enemy hero
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from distar.ctools.torch_utils import fc_block, build_activation
from distar.ctools.torch_utils.network.rnn import sequence_mask
from typing import Optional, List, Tuple

class TargetEntityHead(nn.Module):
    def __init__(self, cfg):
        super(TargetEntityHead, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.policy.head.target_entity_head
        self.act = build_activation(self.cfg.activation)
        self.fc1 = fc_block(self.cfg.input_dim, self.cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(self.cfg.decode_dim, self.cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc3 = fc_block(self.cfg.decode_dim, self.cfg.targetentity_dim, activation=None, norm_type=None)  # regression
        self.embed_fc1 = fc_block(self.cfg.targetentity_dim, self.cfg.targetentity_map_dim, activation=self.act, norm_type=None)
        self.embed_fc2 = fc_block(self.cfg.targetentity_map_dim, self.cfg.input_dim, activation=None, norm_type=None)

        # self.tanh_act = nn.Tanh()
        
        self.targetentity_dim = self.cfg.targetentity_dim
        self.use_mask = True

    def forward(self, embedding, additional_feature, mask, targetentity: Optional[torch.Tensor] = None):
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.tanh_act(x)

        if self.use_mask and mask is not None:
            mask = mask.to(x.device)
            x = x.masked_fill(~mask, -1e9)
        if targetentity is None:
            p = F.softmax(x, dim=-1)
            targetentity = torch.multinomial(p, 1)[:, 0]
        else:
            targetentity = targetentity.squeeze()

        targetentity_encode = torch.nn.functional.one_hot(targetentity.long(), self.targetentity_dim).float()
        embedding_targetentity = self.embed_fc1(targetentity_encode)
        embedding_targetentity = self.embed_fc2(embedding_targetentity)  # get autoregressive_embedding

        return x, targetentity, embedding + embedding_targetentity

class TargetCardHead(nn.Module):
    def __init__(self, cfg):
        super(TargetCardHead, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.policy.head.target_card_head
        self.act = build_activation(self.cfg.activation)
        self.fc1 = fc_block(self.cfg.input_dim, self.cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(self.cfg.decode_dim, self.cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc3 = fc_block(self.cfg.decode_dim, self.cfg.discovery_dim, activation=None, norm_type=None)  # regression
        
        # self.tanh_act = nn.Tanh()
        
        self.discovery_dim = self.cfg.discovery_dim
        self.use_mask = True

    def forward(self, embedding, additional_feature, mask, targetcard: Optional[torch.Tensor] = None):
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.tanh_act(x)
        if self.use_mask and mask is not None:
            mask = mask.to(x.device)
            x = x.masked_fill(~mask, -1e9)
        if targetcard is None:
            p = F.softmax(x, dim=-1)
            targetcard = torch.multinomial(p, 1)[:, 0]
        else:
            targetcard = targetcard.squeeze()

        return x, targetcard

class TargetPositionHead(nn.Module):
    def __init__(self, cfg):
        super(TargetPositionHead, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.policy.head.target_position_head
        self.act = build_activation(self.cfg.activation)
        self.fc1 = fc_block(self.cfg.input_dim, self.cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(self.cfg.decode_dim, self.cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc3 = fc_block(self.cfg.decode_dim, self.cfg.position_dim, activation=None, norm_type=None)  # regression
        
        # self.tanh_act = nn.Tanh()
        
        self.position_dim = self.cfg.position_dim
        self.use_mask = True

    def forward(self, embedding, additional_feature, mask, targetposition: Optional[torch.Tensor] = None):
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.tanh_act(x)

        if self.use_mask and mask is not None:
            mask = mask.to(x.device)
            x = x.masked_fill(~mask, -1e9)

        if targetposition is None:
            p = F.softmax(x, dim=-1)
            targetposition = torch.multinomial(p, 1)[:, 0]
        else:
            targetposition = targetposition.squeeze()

        return x, targetposition