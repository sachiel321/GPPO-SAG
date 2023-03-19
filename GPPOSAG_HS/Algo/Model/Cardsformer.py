from os import truncate
import os.path as osp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import sqrt as sqrt
from Env.GameStats import GameStats
from Algo.Model.net import ReverseGRD, GPTConfig
from Algo.Model.Policy import Policy
from distar.ctools.utils import read_config, deep_merge_dicts
from distar.ctools.torch_utils.detach import detach_grad
from .value import ValueBaseline

card_description_dict = GameStats().card_text_dict

card_name = [k for k in card_description_dict]

InitDict = {
        'state_dim': 320,
        'embeddingT' : 320,
        'embeddingS':320,
        'atthead': 8,
        'attlayer': 1,
        'action_dim': None,
        'use_TS':True,
        'use_GTrXL':True,
        'use_attbias':False,
        'init_gru_gate_bias':2.0
    }
hs_model_default_config = read_config(osp.join(osp.dirname(__file__), "actor_critic_default_config.yaml"))

class Cardsformer(nn.Module):
	def __init__(self, cfg={}):
		super().__init__()
		self.whole_cfg = deep_merge_dicts(hs_model_default_config, cfg)
		self.cfg = self.whole_cfg.model
		config = GPTConfig(block_size=100, 
                            state_dim=InitDict['state_dim'],
                            n_layer=InitDict['attlayer'],
                            n_head=InitDict['atthead'],
                            mask=False,
                            n_embd=InitDict['embeddingT'],
                            n_embdS=InitDict['embeddingS'],
                            init_gru_gate_bias=InitDict['init_gru_gate_bias'],
                            use_TS=InitDict['use_TS'],
                            use_GTrXL=InitDict['use_GTrXL'])
		self.mpnet_embedding = nn.Linear(self.cfg.bert_dim, self.cfg.embed_dim)
		self.card_dim = self.cfg.card_dim
		self.embed_dim = self.cfg.embed_dim
		self.entity_dim  = self.card_dim + self.embed_dim
		self.secret_embedding  = nn.Linear(self.cfg.bert_dim, self.entity_dim)
		self.hand_card_feat_embed = nn.Linear(20, self.cfg.card_dim)
		self.minion_embeding = nn.Linear(23, self.cfg.card_dim)
		self.hero_embedding = nn.Linear(29, self.cfg.card_dim)
		self.transformer = ReverseGRD(config)

		self.policy = Policy(self.whole_cfg)
		self.value_network = ValueBaseline(self.cfg.value.param, False)

		self.only_update_baseline = False

		self.device = None

	def forward(self, hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_cards, minions, heros, num_options, mask, actor = True):
		
		hand_card_value = self.mpnet_embedding(hand_card_embed)
		minion_value = self.mpnet_embedding(minion_embed)

		secret_value = self.secret_embedding(secret_embed)
		weapon_value = self.mpnet_embedding(weapon_embed)

		if actor:
			hand_card_value = hand_card_value.repeat(1, 1, 1)
			minion_value = minion_value.repeat(1, 1, 1)
			secret_value = secret_value.repeat(1, 1, 1)
			weapon_value = weapon_value.repeat(1, 1, 1)
			hand_cards = hand_cards.repeat(1, 1, 1)	
			minions = minions.repeat(1, 1, 1)
			heros = heros.repeat(1, 1, 1)

		hand_card_feat = self.hand_card_feat_embed(hand_cards)
		hand_card_feat = torch.cat((hand_card_feat, hand_card_value), dim=-1)

		minions_feat = self.minion_embeding(minions)
		minions_feat = torch.cat((minions_feat, minion_value), dim=-1)

		heros_feat_ = self.hero_embedding(heros)
		heros_feat = torch.cat((heros_feat_, weapon_value), dim=-1)

		entities = torch.cat((hand_card_feat, minions_feat, heros_feat, secret_value), dim = -2)
		if not actor:
			entities = entities.reshape(-1, 32, self.entity_dim)

		entities_feat = self.transformer(entities) # entities_feat: (batch_size,320)

		action_info, logit = self.policy.forward(entities_feat, hand_card_feat, minions_feat, heros_feat, secret_value, mask)

		return action_info, logit

	def select_forward(self, hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_cards, minions, heros, num_options, available_actions, mask, actor = True):
		
		hand_card_value = self.mpnet_embedding(hand_card_embed)
		minion_value = self.mpnet_embedding(minion_embed)

		secret_value = self.secret_embedding(secret_embed)
		weapon_value = self.mpnet_embedding(weapon_embed)

		if actor:
			hand_card_value = hand_card_value.repeat(1, 1, 1)
			minion_value = minion_value.repeat(1, 1, 1)
			secret_value = secret_value.repeat(1, 1, 1)
			weapon_value = weapon_value.repeat(1, 1, 1)
			hand_cards = hand_cards.repeat(1, 1, 1)	
			minions = minions.repeat(1, 1, 1)
			heros = heros.repeat(1, 1, 1)

		hand_card_feat = self.hand_card_feat_embed(hand_cards)
		hand_card_feat = torch.cat((hand_card_feat, hand_card_value), dim=-1)

		minions_feat = self.minion_embeding(minions)
		minions_feat = torch.cat((minions_feat, minion_value), dim=-1)

		heros_feat_ = self.hero_embedding(heros)
		heros_feat = torch.cat((heros_feat_, weapon_value), dim=-1)

		entities = torch.cat((hand_card_feat, minions_feat, heros_feat, secret_value), dim = -2)
		if not actor:
			entities = entities.reshape(-1, 32, self.entity_dim)

		entities_feat = self.transformer(entities) # entities_feat: (batch_size,320)

		action_info, logit, mask, mask_head = self.policy.select_forward(entities_feat, hand_card_feat, minions_feat, heros_feat, secret_value, available_actions, mask)

		return action_info, logit, mask, mask_head
	
	def train_forward(self, 
						hand_card_embed, 
						minion_embed, 
						secret_embed, 
						weapon_embed, 
						hand_cards, 
						minions, 
						heros, 
						behaviour_logp,
                        teacher_logprob,
						reward,
						mask, 
						mask_head,
						action_info,
						done):
		T = hand_card_embed.shape[0]
		B = hand_card_embed.shape[1]
		hand_card_value = self.mpnet_embedding(hand_card_embed)
		minion_value = self.mpnet_embedding(minion_embed)

		

		secret_value = self.secret_embedding(secret_embed)
		weapon_value = self.mpnet_embedding(weapon_embed)

		hand_card_feat = self.hand_card_feat_embed(hand_cards)
		hand_card_feat = torch.cat((hand_card_feat, hand_card_value), dim=-1)

		minions_feat = self.minion_embeding(minions)
		minions_feat = torch.cat((minions_feat, minion_value), dim=-1)

		heros_feat_ = self.hero_embedding(heros)
		heros_feat = torch.cat((heros_feat_, weapon_value), dim=-1)

		entities = torch.cat((hand_card_feat, minions_feat, heros_feat, secret_value), dim = -2)

		entities_feat = self.transformer(entities) # entities_feat: (T,B,320)

		if self.only_update_baseline:
			entities = entities_feat.detach()

		value = self.value_network(entities_feat).squeeze()

		_, logit = self.policy.train_forward(entities_feat, hand_card_feat, minions_feat, heros_feat, secret_value, mask_head, action_info)

		outputs = {}
		outputs['batch_size'] = B
		outputs['value'] = value
		outputs['mask'] = mask
		outputs['mask_head'] = mask_head
		outputs['target_logit'] = logit
		outputs['action_log_prob'] = behaviour_logp
		outputs['teacher_logprob'] = teacher_logprob
		outputs['reward'] = reward
		outputs['action'] = action_info
		outputs['done'] = done

		return outputs
	
	def sl_forward(self, hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_cards, minions, heros, mask, action_info):
		
		hand_card_value = self.mpnet_embedding(hand_card_embed)
		minion_value = self.mpnet_embedding(minion_embed)	

		secret_value = self.secret_embedding(secret_embed)
		weapon_value = self.mpnet_embedding(weapon_embed)

		hand_card_feat = self.hand_card_feat_embed(hand_cards)
		hand_card_feat = torch.cat((hand_card_feat, hand_card_value), dim=-1)

		minions_feat = self.minion_embeding(minions)
		minions_feat = torch.cat((minions_feat, minion_value), dim=-1)

		heros_feat_ = self.hero_embedding(heros)
		heros_feat = torch.cat((heros_feat_, weapon_value), dim=-1)

		entities = torch.cat((hand_card_feat, minions_feat, heros_feat, secret_value), dim = -2)

		entities = entities.reshape(-1, 32, self.entity_dim)

		entities_feat = self.transformer(entities) # entities_feat: (batch_size,320)


		action_info, logit = self.policy.train_forward(entities_feat, hand_card_feat, minions_feat, heros_feat, secret_value, mask, action_info)

		return action_info, logit

class Cardsformer_naive(nn.Module):
	def __init__(self, card_dim = 64, bert_dim = 768, embed_dim = 256, dim_ff = 512):
		super().__init__()
		self.mpnet_embedding = nn.Linear(bert_dim, embed_dim)
		self.card_dim = card_dim
		self.embed_dim = embed_dim
		self.entity_dim  = self.card_dim + self.embed_dim
		self.secret_embedding  = nn.Linear(bert_dim, self.entity_dim)

		self.hand_card_feat_embed = nn.Linear(19, card_dim)
		self.minion_embeding = nn.Linear(26, card_dim)
		self.hero_embedding = nn.Linear(34, card_dim)
		transformer_layer = nn.TransformerEncoderLayer(d_model=self.entity_dim, nhead=8, dim_feedforward=dim_ff, dropout=0.0)
		self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)
		
		self.out_ln = nn.Linear(self.entity_dim, 64)
		self.scale_out = nn.Sequential(
			nn.Linear(self.entity_dim, 1),
			nn.Softmax(dim=-2)
			)
		self.fn_ln = nn.Linear(64, 1)

		self.device = None

	def forward(self, hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_cards, minions, heros, num_options, actor = True):
		
		hand_card_value = self.mpnet_embedding(hand_card_embed)
		minion_value = self.mpnet_embedding(minion_embed)

		secret_value = self.secret_embedding(secret_embed)
		weapon_value = self.mpnet_embedding(weapon_embed)

		if actor:
			hand_card_value = hand_card_value.repeat(num_options, 1, 1)
			minion_value = minion_value.repeat(num_options, 1, 1)
			secret_value = secret_value.repeat(num_options, 1, 1)
			weapon_value = weapon_value.repeat(num_options, 1, 1)
		
		hand_card_feat = self.hand_card_feat_embed(hand_cards)
		hand_card_feat = torch.cat((hand_card_feat, hand_card_value), dim=-1)

		minions_feat = self.minion_embeding(minions)
		minions_feat = torch.cat((minions_feat, minion_value), dim=-1)

		heros_feat = self.hero_embedding(heros)
		heros_feat = torch.cat((heros_feat, weapon_value), dim=-1)
		
		entities = torch.cat((hand_card_feat, minions_feat, heros_feat, secret_value), dim = -2)
		if not actor:
			entities = entities.reshape(-1, 32, self.entity_dim)
		temp_out = self.transformer(entities.permute(1, 0, 2)).permute(1, 0, 2)
		out = self.out_ln(temp_out)
		out_scale = self.scale_out(temp_out)
		out = out * out_scale
		out = torch.sum(out, dim=-2)
		out = self.fn_ln(out).squeeze()

		return out

class Encoder:
    def __init__(self, model, tokenizer):
        self.encoder = model
        self.tokenizer = tokenizer
        self.cache = {}
        self.game_stats = GameStats()

    def to(self, device):
        self.device = device
        self.encoder = self.encoder.to(device)

    def tokens_to_device(self, tokens):
        tok_device = {}
        for key in tokens:
            tok_device[key] = tokens[key].to(self.device)
        return tok_device


    def encode(self, names, no_use=None):
        txt = []
        for name in names:
            if name is None:
                txt.append(None)
            else:
                description = self.game_stats.card_text_dict[name]
                txt.append(description)

        encoded = []
        for sent in txt:

            if sent in self.cache.keys():
                encoded.append(self.cache[sent])
            elif sent is None:
                encoded.append(torch.zeros((1, 768)).to(self.device))
            else:
                encoded_input = self.tokenizer(sent, padding=True, truncation=True, return_tensors='pt')
                encoded_input = self.tokens_to_device(encoded_input)
                with torch.no_grad():
                    model_output = self.encoder(**encoded_input)
                sent_embed = mean_pooling(model_output, encoded_input['attention_mask'])
                sent_embed = F.normalize(sent_embed, p=2, dim=1)
                encoded.append(sent_embed)
                self.cache[sent] = sent_embed
        if len(encoded) == 0:
            return None
        else:
            return torch.cat(encoded, dim=0)  # n * max_length * 768


def mean_pooling(model_output, attention_mask):
	token_embeddings = model_output[0] #First element of model_output contains all token embeddings
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
